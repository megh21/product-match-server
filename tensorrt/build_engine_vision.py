import tensorrt as trt
import os
import logging
import time
import gc
import torch

# Import the factory function from the vision calibrator script
# Make sure calibrate_vision.py uses the corrected paths now
from calibrate_vision import (
    get_vision_calibrator,
    VISION_CACHE_FILE,
    VISION_INPUT_NAME,
    INPUT_SHAPE,
    CORRECT_BASE_DIR,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)

# --- Configuration ---
# ===> PATH CORRECTIONS <===
VISION_ONNX_PATH = os.path.join(
    CORRECT_BASE_DIR, "internvl3_vision.onnx"
)  # Correct ONNX path
ENGINE_DIR = CORRECT_BASE_DIR  # Save engine in the same place
ENGINE_PREFIX = "internvl3_vision"  # Prefix for engine file names

WORKSPACE_MB = 4096
BUILD_FP16 = False
BUILD_INT8 = True

# Vision Input Details
BATCH_SIZE, C, H, W = INPUT_SHAPE

# --- Optimization Profile for Vision Model ---
INPUT_NAMES_FOR_ENGINE = [VISION_INPUT_NAME]
MIN_BATCH_V = 1
OPT_BATCH_V = 8
MAX_BATCH_V = 16

PROFILE_SHAPES = {
    VISION_INPUT_NAME: {
        "min": (MIN_BATCH_V, C, H, W),
        "opt": (OPT_BATCH_V, C, H, W),
        "max": (MAX_BATCH_V, C, H, W),
    }
}


# --- Build Engine Function (Generic) ---
# ... (build_engine function remains the same as before) ...
def build_engine(
    onnx_path,
    engine_path_prefix,
    input_names,
    profile_shapes,
    precision="fp16",
    calibrator_factory=None,
    cache_file=None,
):
    """Builds and saves a TensorRT engine."""
    engine_final_path = f"{engine_path_prefix}_{precision}.engine"
    if os.path.exists(engine_final_path):
        logging.warning(
            f"Engine file {engine_final_path} already exists. Skipping build."
        )
        return True

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    logging.info(f"Parsing ONNX model from: {onnx_path}")
    if not os.path.exists(onnx_path):
        logging.error(f"ONNX file not found: {onnx_path}")
        return False
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            logging.error(f"Failed to parse ONNX: {onnx_path}")
            for error in range(parser.num_errors):
                logging.error(f"  {parser.get_error(error)}")
            return False
    logging.info("ONNX model parsed successfully.")

    network_input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    logging.info(f"Network inputs found: {network_input_names}")
    if set(network_input_names) != set(input_names):
        logging.error(
            f"Mismatch between network inputs {network_input_names} and expected engine inputs {input_names}"
        )
        return False

    config = builder.create_builder_config()
    logging.info(f"Setting workspace size: {WORKSPACE_MB} MB")
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * (1024**2))

    profile = builder.create_optimization_profile()
    logging.info("Setting optimization profile:")
    for name in input_names:
        if name not in profile_shapes:
            logging.error(f"'{name}' not in profile_shapes.")
            return False
        shapes = profile_shapes[name]
        try:
            if name not in network_input_names:
                logging.error(f"Profile input '{name}' not in network inputs.")
                return False
            profile.set_shape(
                name, min=shapes["min"], opt=shapes["opt"], max=shapes["max"]
            )
            logging.info(
                f"  - {name}: min={shapes['min']}, opt={shapes['opt']}, max={shapes['max']}"
            )
        except Exception as e:
            logging.error(f"Error setting profile for {name}: {e}")
            return False
    config.add_optimization_profile(profile)

    calibrator = None
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logging.info("FP16 mode enabled.")
        else:
            logging.warning("FP16 not supported/fast.")
            precision = "fp32"
    elif precision == "int8":
        if not builder.platform_has_fast_int8:
            logging.warning("INT8 not supported/fast. Falling back.")
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                precision = "fp16"
                logging.info("Falling back to FP16.")
            else:
                precision = "fp32"
                logging.warning("Falling back to FP32.")
        else:
            logging.info("Enabling INT8 mode.")
            config.set_flag(trt.BuilderFlag.INT8)
            if builder.platform_has_fast_fp16:
                config.set_flag(trt.BuilderFlag.FP16)
                logging.info("Enabling FP16 mode alongside INT8.")
            if calibrator_factory is None:
                logging.error("Calibrator factory needed for INT8.")
                return False
            logging.info("Creating INT8 calibrator...")
            calibrator = calibrator_factory()  # Call the factory function
            if calibrator is None:
                logging.error("Failed to create calibrator.")
                return False
            config.int8_calibrator = calibrator
            logging.info("INT8 calibrator set.")

    start_time = time.time()
    logging.info(f"Building TensorRT engine ({precision})...")
    serialized_engine = builder.build_serialized_network(network, config)
    end_time = time.time()

    if calibrator is not None:
        logging.info("Cleaning up calibrator resources...")
        del calibrator
        gc.collect()  # Cleanup calibrator

    if serialized_engine is None:
        logging.error(f"Failed to build {precision} engine.")
        return False
    logging.info(f"Engine built in {end_time - start_time:.2f} sec.")

    logging.info(f"Saving {precision} engine to: {engine_final_path}")
    try:
        # Ensure the correct ENGINE_DIR is used for saving
        os.makedirs(os.path.dirname(engine_final_path), exist_ok=True)
        with open(engine_final_path, "wb") as f:
            f.write(serialized_engine)
        logging.info("Engine saved.")
    except Exception as e:
        logging.error(f"Failed to save engine: {e}")
        return False

    del serialized_engine, config, parser, network, builder
    gc.collect()
    time.sleep(1)
    return True


# --- Main Execution ---
if __name__ == "__main__":
    logging.info("--- Starting Vision TensorRT Engine Build Process ---")
    # Use the corrected ENGINE_DIR and ENGINE_PREFIX based on CORRECT_BASE_DIR
    os.makedirs(ENGINE_DIR, exist_ok=True)
    engine_base_path = os.path.join(ENGINE_DIR, ENGINE_PREFIX)

    success_fp16 = False
    if BUILD_FP16:
        logging.info("\n--- Building Vision FP16 Engine ---")
        success_fp16 = build_engine(
            VISION_ONNX_PATH,
            engine_base_path,  # Use corrected paths
            INPUT_NAMES_FOR_ENGINE,
            PROFILE_SHAPES,
            precision="fp16",
        )
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

    success_int8 = False
    if BUILD_INT8:
        logging.info("\n--- Building Vision INT8 Engine ---")
        success_int8 = build_engine(
            VISION_ONNX_PATH,
            engine_base_path,  # Use corrected paths
            INPUT_NAMES_FOR_ENGINE,
            PROFILE_SHAPES,
            precision="int8",
            calibrator_factory=get_vision_calibrator,  # Pass the factory function
            cache_file=VISION_CACHE_FILE,  # Pass corrected cache file path
        )
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(2)

    logging.info("\n--- Vision Build Process Summary ---")
    if BUILD_FP16:
        logging.info(f"FP16 Engine Build: {'SUCCESS' if success_fp16 else 'FAILED'}")
    if BUILD_INT8:
        logging.info(f"INT8 Engine Build: {'SUCCESS' if success_int8 else 'FAILED'}")
    logging.info("Vision engine building process finished.")
