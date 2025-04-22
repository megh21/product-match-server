import os
import tensorrt as trt
import onnx
import logging
import numpy as np
from PIL import Image
from transformers import AutoProcessor

logging.basicConfig(level=logging.INFO)
TRT_LOGGER = trt.Logger(trt.Logger.INFO)


def build_engine(onnx_path, engine_path, precision="fp16", calibrator=None):
    """
    Build TensorRT engine from ONNX file

    Args:
        onnx_path: Path to ONNX file
        engine_path: Path to save TensorRT engine
        precision: Precision mode ('fp32', 'fp16', or 'int8')
        calibrator: Calibrator for INT8 quantization

    Returns:
        TensorRT engine
    """
    logging.info(f"Building {precision} TensorRT engine from {onnx_path}")

    # Create builder and config
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    config = builder.create_builder_config()

    # Set workspace size (8GB)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 8 << 30)

    # Parse ONNX file
    parser = trt.OnnxParser(network, TRT_LOGGER)
    with open(onnx_path, "rb") as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                logging.error(f"ONNX parse error: {parser.get_error(error)}")
            raise RuntimeError("Failed to parse ONNX model")

    # Set precision flags
    if precision == "fp16" and builder.platform_has_fast_fp16:
        logging.info("Enabling FP16 mode")
        config.set_flag(trt.BuilderFlag.FP16)
    elif precision == "int8" and builder.platform_has_fast_int8:
        logging.info("Enabling INT8 mode")
        config.set_flag(trt.BuilderFlag.INT8)
        if calibrator:
            config.int8_calibrator = calibrator
        else:
            logging.warning("INT8 mode requires a calibrator, but none was provided.")
            return None

    # Build and serialize engine
    serialized_engine = builder.build_serialized_network(network, config)
    if not serialized_engine:
        logging.error("Failed to build TensorRT engine")
        return None

    # Save engine to file
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)

    logging.info(f"Successfully built and saved TensorRT engine to {engine_path}")

    # Create runtime and deserialize engine
    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)

    return engine


# INT8 Calibrator for the vision model
class VisionEntropyCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(
        self, calibration_data_dir, batch_size=1, cache_file="calibration.cache"
    ):
        super().__init__()
        self.cache_file = cache_file
        self.batch_size = batch_size
        self.current_index = 0

        # Find image files in calibration directory
        self.image_files = []
        if os.path.exists(calibration_data_dir):
            for filename in os.listdir(calibration_data_dir):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.image_files.append(
                        os.path.join(calibration_data_dir, filename)
                    )

        logging.info(f"Found {len(self.image_files)} images for calibration")

        # Load processor
        self.processor = AutoProcessor.from_pretrained("OpenGVLab/InternVL3-2B")

        # Allocate host memory for inputs
        self.pixel_values = None

    def get_batch_size(self):
        return self.batch_size

    def get_batch(self, names):
        if self.current_index + self.batch_size > len(self.image_files):
            return None

        batch_images = []

        # Process batch_size images
        for i in range(self.batch_size):
            if self.current_index + i < len(self.image_files):
                image_path = self.image_files[self.current_index + i]
                try:
                    image = Image.open(image_path).convert("RGB")
                    inputs = self.processor(images=image, return_tensors="pt")
                    batch_images.append(inputs["pixel_values"][0].numpy())
                except Exception as e:
                    logging.error(f"Error processing image {image_path}: {e}")

        self.current_index += self.batch_size

        if not batch_images:
            return None

        # Stack images into a batch
        batch = np.stack(batch_images)

        # Allocate memory if needed
        if self.pixel_values is None or self.pixel_values.shape[0] != batch.shape[0]:
            self.pixel_values = np.ascontiguousarray(batch)
        else:
            np.copyto(self.pixel_values, batch)

        return [int(self.pixel_values.ctypes.data)]

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        with open(self.cache_file, "wb") as f:
            f.write(cache)


def main():
    # Base directories
    model_dir = "models/internvl3"
    trt_dir = "models/internvl3_trt"
    os.makedirs(trt_dir, exist_ok=True)

    # Vision model
    vision_onnx_path = os.path.join(model_dir, "internvl3_vision.onnx")
    vision_fp16_path = os.path.join(trt_dir, "internvl3_vision_fp16.engine")
    vision_int8_path = os.path.join(trt_dir, "internvl3_vision_int8.engine")

    # Text model
    text_onnx_path = os.path.join(model_dir, "internvl3_text.onnx")
    text_fp16_path = os.path.join(trt_dir, "internvl3_text_fp16.engine")

    # Convert vision model to FP16
    if os.path.exists(vision_onnx_path):
        try:
            # Verify ONNX model
            onnx_model = onnx.load(vision_onnx_path)
            onnx.checker.check_model(onnx_model)
            logging.info("Vision ONNX model is valid")

            # Convert to TensorRT FP16
            engine = build_engine(vision_onnx_path, vision_fp16_path, precision="fp16")
            logging.info("Successfully converted vision model to TensorRT FP16")

            # Convert to TensorRT INT8 with calibration
            calib_data_dir = "data/calibration_images"
            if os.path.exists(calib_data_dir) and len(os.listdir(calib_data_dir)) > 0:
                calibrator = VisionEntropyCalibrator(
                    calib_data_dir,
                    batch_size=1,
                    cache_file=os.path.join(trt_dir, "vision_calibration.cache"),
                )
                engine = build_engine(
                    vision_onnx_path,
                    vision_int8_path,
                    precision="int8",
                    calibrator=calibrator,
                )
                logging.info("Successfully converted vision model to TensorRT INT8")
            else:
                logging.warning(
                    f"Calibration data directory {calib_data_dir} not found or empty. Skipping INT8 quantization."
                )
        except Exception as e:
            logging.error(f"Failed to convert vision model: {e}")

    # Convert text model to FP16
    if os.path.exists(text_onnx_path):
        try:
            # Verify ONNX model
            onnx_model = onnx.load(text_onnx_path)
            onnx.checker.check_model(onnx_model)
            logging.info("Text ONNX model is valid")

            # Convert to TensorRT
            engine = build_engine(text_onnx_path, text_fp16_path, precision="fp16")
            print(type(engine))
            logging.info("Successfully converted text model to TensorRT FP16")
        except Exception as e:
            logging.error(f"Failed to convert text model: {e}")


if __name__ == "__main__":
    main()
