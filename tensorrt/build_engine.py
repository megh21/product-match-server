# (Add to the end of calibrate_clip_vision.py or create a new build script)
import tensorrt as trt
import os
import logging
from calibrate_clip_vision import ClipVisionCalibrator, ONNX_PATH, CACHE_FILE, calib_dataset # Reuse definitions

# --- Configuration ---
ENGINE_PATH_FP16 = "models/clip_vision/clip_vision_fp16_py.engine"
ENGINE_PATH_INT8 = "models/clip_vision/clip_vision_int8_py.engine"
WORKSPACE_MB = 4096

def build_engine(onnx_path, engine_path, precision="fp16", calib_dataset=None, cache_file=None):
    logger = trt.Logger(trt.Logger.VERBOSE if os.environ.get("TRT_VERBOSE") else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    logging.info(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                logging.error(f"ONNX Parser Error: {parser.get_error(error)}")
            raise ValueError("Failed to parse the ONNX file.")
    logging.info("ONNX model parsed successfully.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * (1024**2))
    logging.info(f"Workspace size set to {WORKSPACE_MB} MB")

    # --- Set Precision ---
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logging.info("FP16 mode enabled.")
        else:
            logging.warning("FP16 not supported on this platform, using FP32.")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logging.info("INT8 mode enabled.")
            if calib_dataset is None or cache_file is None:
                raise ValueError("Calibration dataset and cache file needed for INT8.")
            logging.info("Setting up INT8 calibrator...")
            config.int8_calibrator = ClipVisionCalibrator(calib_dataset, cache_file) # Use the class from calibrate script
            # Optional: Set calibration profile if needed (usually automatic)
            # profile = builder.create_optimization_profile()
            # ... set shapes for profile ...
            # config.add_optimization_profile(profile)

        else:
            logging.warning("INT8 not supported on this platform, using FP32.")
            precision = "fp32" # Fallback

    # --- Build Engine ---
    logging.info(f"Building TensorRT engine ({precision})... This may take a while.")
    # Note: build_serialized_network is preferred for saving/loading
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")
    logging.info("TensorRT engine built successfully.")

    # --- Save Engine ---
    logging.info(f"Saving engine to: {engine_path}")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    logging.info("Engine saved.")


if __name__ == "__main__":
    # Build FP16 Engine
    # build_engine(ONNX_PATH, ENGINE_PATH_FP16, precision="fp16")

    # Build INT8 Engine (Requires calibration dataset loaded)
    logging.info("Loading calibration dataset for INT8 build...")
    from datasets import load_dataset
    # Ensure this matches the calibrator script's config
    DATASET_NAME = "restufiqih/fashion-product"
    DATASET_SPLIT = "train"
    NUM_CALIBRATION_IMAGES = 200 # Match calibrator
    calib_dataset_main = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=False)
    # If streaming=True, convert to list or adapt calibrator:
    # calib_dataset_main = list(calib_dataset_main.take(NUM_CALIBRATION_IMAGES))

    build_engine(ONNX_PATH, ENGINE_PATH_INT8, precision="int8", calib_dataset=calib_dataset_main, cache_file=CACHE_FILE)

    logging.info("Engine building process finished.")