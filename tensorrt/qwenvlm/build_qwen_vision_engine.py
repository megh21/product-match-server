import tensorrt as trt
import os
import logging
from dotenv import load_dotenv
from calibrate.calibrate_qwen_vision import get_qwen_vision_calibrator  # Import factory

logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Configuration ---
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "Qwen/Qwen-VL-Chat")
ONNX_PATH = f"models/{VLM_MODEL_ID.split('/')[-1]}_vision/qwen_vision_encoder.onnx"
ENGINE_DIR = f"models/{VLM_MODEL_ID.split('/')[-1]}_vision"
ENGINE_PATH_FP16 = os.path.join(ENGINE_DIR, "qwen_vision_fp16.plan")
ENGINE_PATH_INT8 = os.path.join(ENGINE_DIR, "qwen_vision_int8.plan")
WORKSPACE_MB = 4096  # Qwen vision tower might be large

# (batch_size, channels, height, width) - Must match calibration/export
IMAGE_SIZE = 448  # From calibration script
MIN_SHAPE = (1, 3, IMAGE_SIZE, IMAGE_SIZE)
OPT_SHAPE = (8, 3, IMAGE_SIZE, IMAGE_SIZE)  # Optimal batch size
MAX_SHAPE = (16, 3, IMAGE_SIZE, IMAGE_SIZE)  # Max batch size

INPUT_NAME = "pixel_values"  # From export script


def build_engine(precision="fp16"):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    engine_path = ENGINE_PATH_FP16 if precision == "fp16" else ENGINE_PATH_INT8

    logging.info(f"Parsing ONNX model: {ONNX_PATH}")
    if not os.path.exists(ONNX_PATH):
        raise FileNotFoundError(
            f"ONNX file not found: {ONNX_PATH}. Run export script first."
        )
    with open(ONNX_PATH, "rb") as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                logging.error(f"ONNX Parser Error: {parser.get_error(error)}")
            raise ValueError("Failed to parse the ONNX file.")
    logging.info("ONNX model parsed successfully.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * (1024**2))
    logging.info(f"Workspace size set to {WORKSPACE_MB} MB")

    # --- Create Optimization Profile ---
    profile = builder.create_optimization_profile()
    profile.set_shape(INPUT_NAME, min=MIN_SHAPE, opt=OPT_SHAPE, max=MAX_SHAPE)
    config.add_optimization_profile(profile)
    logging.info(
        f"Added optimization profile: MIN={MIN_SHAPE}, OPT={OPT_SHAPE}, MAX={MAX_SHAPE}"
    )

    # --- Set Precision ---
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logging.info("FP16 mode enabled.")
        else:
            logging.warning("FP16 not supported, using FP32.")
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            config.set_flag(trt.BuilderFlag.INT8)
            logging.info("INT8 mode enabled. Setting up calibrator...")
            calibrator = get_qwen_vision_calibrator()
            if calibrator is None:
                raise ValueError("Failed to get Qwen Vision INT8 calibrator.")
            config.int8_calibrator = calibrator
            logging.info("INT8 calibrator set.")
        else:
            logging.warning("INT8 not supported. Cannot build INT8 engine.")
            return  # Exit
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    # --- Build Engine ---
    logging.info(
        f"Building TensorRT engine ({precision})... This may take considerable time."
    )
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(f"Failed to build TensorRT engine ({precision}).")
    logging.info(f"TensorRT engine ({precision}) built successfully.")

    # --- Save Engine ---
    logging.info(f"Saving engine to: {engine_path}")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    logging.info(f"Engine ({precision}) saved.")


if __name__ == "__main__":
    logging.info("--- Building FP16 Engine for Qwen Vision ---")
    build_engine(precision="fp16")

    logging.info("\n--- Building INT8 Engine for Qwen Vision ---")
    try:
        build_engine(precision="int8")
    except Exception as e:
        logging.error(f"Failed to build INT8 Qwen Vision engine: {e}", exc_info=True)

    logging.info("\nQwen Vision engine building process finished.")
