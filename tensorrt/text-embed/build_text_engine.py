import tensorrt as trt
import os
import logging
from dotenv import load_dotenv
from calibrate.calibrate_text import get_calibrator  # Import calibrator factory

logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Configuration ---
ONNX_PATH = "models/text_encoder/text_encoder.onnx"
ENGINE_DIR = "models/text_encoder"
ENGINE_PATH_FP16 = os.path.join(ENGINE_DIR, "text_encoder_fp16.plan")
ENGINE_PATH_INT8 = os.path.join(ENGINE_DIR, "text_encoder_int8.plan")
WORKSPACE_MB = 2048  # Adjust based on available GPU memory

# Optimization Profile (Min, Optimal, Max shapes for dynamic axes)
# (batch_size, sequence_length)
MIN_SHAPE = (1, 32)  # min batch 1, min seq len 32
OPT_SHAPE = (8, 128)  # optimal batch 8, optimal seq len 128
MAX_SHAPE = (
    16,
    128,
)  # max batch 16, max seq len 128 (must match MAX_SEQ_LENGTH from export/calib)

INPUT_NAMES = ["input_ids", "attention_mask"]


def build_engine(precision="fp16"):
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)

    engine_path = ENGINE_PATH_FP16 if precision == "fp16" else ENGINE_PATH_INT8

    logging.info(f"Parsing ONNX model: {ONNX_PATH}")
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
    min_shape_tuple = MIN_SHAPE
    opt_shape_tuple = OPT_SHAPE
    max_shape_tuple = MAX_SHAPE
    # Set shapes for both inputs
    profile.set_shape(
        INPUT_NAMES[0], min=min_shape_tuple, opt=opt_shape_tuple, max=max_shape_tuple
    )
    profile.set_shape(
        INPUT_NAMES[1], min=min_shape_tuple, opt=opt_shape_tuple, max=max_shape_tuple
    )
    config.add_optimization_profile(profile)
    logging.info(
        f"Added optimization profile: MIN={min_shape_tuple}, OPT={opt_shape_tuple}, MAX={max_shape_tuple}"
    )

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
            logging.info("INT8 mode enabled. Setting up calibrator...")
            # Get calibrator instance from the calibration script
            calibrator = get_calibrator()
            if calibrator is None:
                raise ValueError("Failed to get INT8 calibrator.")
            config.int8_calibrator = calibrator
            logging.info("INT8 calibrator set.")
        else:
            logging.warning(
                "INT8 not supported on this platform. Cannot build INT8 engine."
            )
            return  # Exit if INT8 requested but not supported
    else:
        raise ValueError(f"Unsupported precision: {precision}")

    # --- Build Engine ---
    logging.info(f"Building TensorRT engine ({precision})... This may take time.")
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError(
            f"Failed to build TensorRT engine ({precision}). Check logs for errors."
        )
    logging.info(f"TensorRT engine ({precision}) built successfully.")

    # --- Save Engine ---
    logging.info(f"Saving engine to: {engine_path}")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    logging.info(f"Engine ({precision}) saved.")


if __name__ == "__main__":
    logging.info("--- Building FP16 Engine ---")
    build_engine(precision="fp16")

    logging.info("\n--- Building INT8 Engine ---")
    try:
        build_engine(precision="int8")
    except Exception as e:
        logging.error(f"Failed to build INT8 engine: {e}", exc_info=True)

    logging.info("\nEngine building process finished.")
