import tensorrt as trt
import os
import logging
import time
import gc
import torch
from calibrate.calibrate_vlm import (
    VLMCalibrator,
    CACHE_FILE,
    INPUT_NAMES as CALIBRATOR_INPUT_NAMES,
    NUM_IMAGE_TOKENS,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)

# --- Configuration ---
ONNX_PATH = "models/internvl3_vlm/internvl3_chat_2b_full.onnx"  # Input ONNX model
ENGINE_DIR = "models/internvl3_vlm"
ENGINE_PREFIX = "internvl3_chat_2b"  # Use specific prefix

WORKSPACE_MB = 4096 * 4  # 16GB workspace, adjust based on GPU VRAM
BUILD_FP16 = True
BUILD_INT8 = True

# --- Optimization Profile
INPUT_NAMES_FOR_ENGINE = ["pixel_values", "input_ids", "attention_mask", "image_flags"]

MIN_BATCH = 1
OPT_BATCH = 4
MAX_BATCH = 8


MIN_SEQ_LEN = 64  # Min length for inference context/generation step
OPT_SEQ_LEN = 512  # Typical target context length
MAX_SEQ_LEN = 1024  # Max supported context length (adjust based on VRAM)

# Image dims (C, H, W) - should be fixed from preprocessing
_, C, H, W = (3, 448, 448)  # Hardcode or get from image_processor config if needed

PROFILE_SHAPES = {
    # Names must match INPUT_NAMES_FOR_ENGINE
    "pixel_values": {
        "min": (MIN_BATCH, C, H, W),
        "opt": (OPT_BATCH, C, H, W),
        "max": (MAX_BATCH, C, H, W),
    },
    "input_ids": {
        "min": (MIN_BATCH, MIN_SEQ_LEN),
        "opt": (OPT_BATCH, OPT_SEQ_LEN),
        "max": (MAX_BATCH, MAX_SEQ_LEN),
    },
    "attention_mask": {
        "min": (MIN_BATCH, MIN_SEQ_LEN),
        "opt": (OPT_BATCH, OPT_SEQ_LEN),
        "max": (MAX_BATCH, MAX_SEQ_LEN),
    },
    "image_flags": {
        "min": (MIN_BATCH, NUM_IMAGE_TOKENS),  # Batch dynamic, token dim fixed
        "opt": (OPT_BATCH, NUM_IMAGE_TOKENS),
        "max": (MAX_BATCH, NUM_IMAGE_TOKENS),
    },
}


# --- Build Engine Function
def build_engine(onnx_path, engine_path_prefix, precision="fp16", cache_file=None):
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

    # Parse the ONNX file from disk
    with open(onnx_path, "rb") as model_file:
        if not parser.parse(model_file.read()):
            logging.error("Failed to parse the ONNX model.")
            for error in range(parser.num_errors):
                logging.error(f"  Parser error: {parser.get_error(error)}")
            return False
        logging.info("ONNX model parsed successfully.")

    # Verify network inputs match expected names
    network_input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    logging.info(f"Network inputs found: {network_input_names}")
    if set(network_input_names) != set(INPUT_NAMES_FOR_ENGINE):
        logging.error(
            f"Mismatch between network inputs {network_input_names} and expected engine inputs {INPUT_NAMES_FOR_ENGINE}"
        )
        return False

    config = builder.create_builder_config()
    logging.info(f"Setting workspace size: {WORKSPACE_MB} MB")
    # Use set_memory_pool_limit for newer TRT versions
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * (1024**2))

    # --- Add Optimization Profile ---
    profile = builder.create_optimization_profile()
    logging.info("Setting optimization profile:")
    for name in INPUT_NAMES_FOR_ENGINE:  # Use the list defined for the engine
        if name not in PROFILE_SHAPES:
            logging.error(f"Input '{name}' not found in PROFILE_SHAPES configuration.")
            return False
        shapes = PROFILE_SHAPES[name]
        try:
            # Make sure the name exists in the parsed network
            if name not in network_input_names:
                logging.error(
                    f"Profile input '{name}' not found in the actual network inputs: {network_input_names}"
                )
                return False
            profile.set_shape(
                name, min=shapes["min"], opt=shapes["opt"], max=shapes["max"]
            )
            logging.info(
                f"  - {name}: min={shapes['min']}, opt={shapes['opt']}, max={shapes['max']}"
            )
        except Exception as e:
            logging.error(
                f"Error setting profile shape for {name}: {e}. Check dimensions and network definition."
            )
            return False
    config.add_optimization_profile(profile)

    # --- Set Precision ---
    calibrator = None
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            logging.info("Enabling FP16 mode.")
            config.set_flag(trt.BuilderFlag.FP16)
        else:
            logging.warning("FP16 not supported or not fast. Engine will be FP32.")
            precision = "fp32"
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            logging.info("Enabling INT8 mode.")
            config.set_flag(trt.BuilderFlag.INT8)
            if builder.platform_has_fast_fp16:
                logging.info("Enabling FP16 mode alongside INT8.")
                config.set_flag(trt.BuilderFlag.FP16)

            if cache_file is None:
                logging.error("Cache file path is required for INT8 calibration.")
                return False

            logging.info("Instantiating INT8 calibrator (VLMCalibrator)...")
            try:
                calibrator = VLMCalibrator(cache_file)
                # Sanity check calibrator input names vs engine input names
                if set(CALIBRATOR_INPUT_NAMES) != set(INPUT_NAMES_FOR_ENGINE):
                    logging.warning(
                        f"Mismatch between calibrator inputs {CALIBRATOR_INPUT_NAMES} and engine inputs {INPUT_NAMES_FOR_ENGINE}. Ensure calibrator provides all engine inputs."
                    )

                config.int8_calibrator = calibrator
                logging.info("INT8 calibrator set.")
            except Exception as e:
                logging.error(f"Failed to initialize calibrator: {e}", exc_info=True)
                return False
        else:
            logging.warning("INT8 not supported or not fast.")
            # Fallback logic
            if builder.platform_has_fast_fp16:
                logging.info("Falling back to FP16 mode.")
                config.set_flag(trt.BuilderFlag.FP16)
                precision = "fp16"
            else:
                logging.warning("Falling back to FP32 mode.")
                precision = "fp32"

    # --- Build Engine ---
    start_time = time.time()
    logging.info(
        f"Building TensorRT engine ({precision})... This will take time and memory."
    )

    # build_serialized_network is preferred
    serialized_engine = builder.build_serialized_network(network, config)

    end_time = time.time()

    # Clean up calibrator immediately after build finishes
    if calibrator is not None:
        logging.info("Cleaning up calibrator resources...")
        del calibrator
        gc.collect()

    if serialized_engine is None:
        logging.error(f"Failed to build the {precision} engine. Check TRT logs above.")
        # Check for common errors like OOM, unsupported layers, profile issues.
        return False

    logging.info(f"Engine built successfully in {end_time - start_time:.2f} seconds.")

    # --- Save Engine ---
    logging.info(f"Saving {precision} engine to: {engine_final_path}")
    try:
        os.makedirs(os.path.dirname(engine_final_path), exist_ok=True)
        with open(engine_final_path, "wb") as f:
            f.write(serialized_engine)
        logging.info("Engine saved.")
    except Exception as e:
        logging.error(f"Failed to save engine: {e}", exc_info=True)
        return False

    # Explicitly delete large objects
    del serialized_engine, config, parser, network, builder
    gc.collect()
    time.sleep(2)  # Small delay to help with cleanup

    return True


if __name__ == "__main__":
    logging.info("--- Starting TensorRT Engine Build Process ---")
    os.makedirs(ENGINE_DIR, exist_ok=True)

    success_fp16 = False
    if BUILD_FP16:
        logging.info("\n--- Building FP16 Engine ---")
        success_fp16 = build_engine(
            ONNX_PATH, os.path.join(ENGINE_DIR, ENGINE_PREFIX), precision="fp16"
        )
        if success_fp16:
            logging.info("FP16 engine build completed.")
        else:
            logging.error("FP16 engine build failed.")
        torch.cuda.empty_cache()  # Clear cache between builds
        gc.collect()
        time.sleep(5)  # Allow time for memory release

    success_int8 = False
    if BUILD_INT8:
        logging.info("\n--- Building INT8 Engine ---")
        # INT8 build requires the cache file path
        success_int8 = build_engine(
            ONNX_PATH,
            os.path.join(ENGINE_DIR, ENGINE_PREFIX),
            precision="int8",
            cache_file=CACHE_FILE,
        )
        if success_int8:
            logging.info("INT8 engine build completed.")
        else:
            logging.error("INT8 engine build failed.")
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)

    logging.info("\n--- Build Process Summary ---")
    if BUILD_FP16:
        logging.info(f"FP16 Engine Build: {'SUCCESS' if success_fp16 else 'FAILED'}")
    if BUILD_INT8:
        logging.info(f"INT8 Engine Build: {'SUCCESS' if success_int8 else 'FAILED'}")

    logging.info("TensorRT engine build process finished.")
