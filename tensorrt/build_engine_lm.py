import tensorrt as trt
import os
import logging
import time
import gc
import torch
import onnx

# Import the factory function from the LM calibrator script
from calibrate_lm import get_lm_calibrator, LM_CACHE_FILE, LM_INPUT_NAMES, HIDDEN_SIZE

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
# empty cacheand gpu memory
torch.cuda.empty_cache()
gc.collect()
# --- Configuration ---
LM_ONNX_PATH = "models/internvl3_lm/internvl3_language.onnx"
ENGINE_DIR = "models/internvl3_lm"
ENGINE_PREFIX = "internvl3_language"
WORKSPACE_MB = 4096 * 2
BUILD_FP16 = False
BUILD_INT8 = True

# --- Optimization Profile ---
# ... (Profile shapes remain the same) ...
INPUT_NAMES_FOR_ENGINE = LM_INPUT_NAMES
MIN_BATCH_L = 1
OPT_BATCH_L = 4
MAX_BATCH_L = 4
MIN_SEQ_LEN_L = 64
OPT_SEQ_LEN_L = 512
MAX_SEQ_LEN_L = 512

PROFILE_SHAPES = {
    "inputs_embeds": {
        "min": (MIN_BATCH_L, MIN_SEQ_LEN_L, HIDDEN_SIZE),
        "opt": (OPT_BATCH_L, OPT_SEQ_LEN_L, HIDDEN_SIZE),
        "max": (MAX_BATCH_L, MAX_SEQ_LEN_L, HIDDEN_SIZE),
    },
    "attention_mask": {
        "min": (MIN_BATCH_L, MIN_SEQ_LEN_L),
        "opt": (OPT_BATCH_L, OPT_SEQ_LEN_L),
        "max": (MAX_BATCH_L, MAX_SEQ_LEN_L),
    },
}


# --- Build Engine Function (Generic) ---
def build_engine(
    onnx_path,
    engine_path_prefix,
    input_names,
    profile_shapes,
    precision="fp16",
    calibrator_factory=None,
    cache_file=None,
):
    engine_final_path = f"{engine_path_prefix}_{precision}.engine"
    if os.path.exists(engine_final_path):
        logging.warning(
            f"Engine file {engine_final_path} already exists. Skipping build."
        )
        return True

    # ONNX Pre-check (using file path)
    logging.info(f"Attempting ONNX pre-check on file: {onnx_path}")
    try:
        onnx.checker.check_model(
            onnx_path
        )  # Removed check_external_data=True as it's invalid
        logging.info("ONNX library check successful.")
    except Exception as onnx_err:
        logging.error(f"ONNX pre-check failed: {onnx_err}", exc_info=True)
        return False

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(network_flags)
    parser = trt.OnnxParser(network, logger)

    logging.info(f"Parsing ONNX model with TensorRT directly from file: {onnx_path}")
    if not os.path.exists(onnx_path):
        logging.error(f"ONNX file not found: {onnx_path}")
        return False

    # ===> CHANGE: Use parse_from_file <===
    success = parser.parse_from_file(onnx_path)
    if not success:
        logging.error(f"TensorRT failed to parse ONNX from file: {onnx_path}")
        for error in range(parser.num_errors):
            logging.error(f"  Parser error: {parser.get_error(error)}")
        # Check specifically for external data errors
        if "Failed to open file" in str(
            [parser.get_error(i) for i in range(parser.num_errors)]
        ):
            logging.error(
                "This might indicate TRT cannot find/access the external data file(s) even when parsing from file path. Ensure they are in the same directory and have correct permissions."
            )
        return False
    # ===> END CHANGE <===
    logging.info("TensorRT ONNX parser successful.")

    # ... (Rest of the build_engine function remains the same) ...
    network_input_names = [network.get_input(i).name for i in range(network.num_inputs)]
    logging.info(f"Network inputs found: {network_input_names}")
    if set(network_input_names) != set(input_names):
        logging.error(
            f"Mismatch between network inputs {network_input_names} and expected engine inputs {input_names}"
        )
        # Clean up before returning
        del parser, network, builder
        gc.collect()
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
            calibrator = calibrator_factory()
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
        gc.collect()

    if serialized_engine is None:
        logging.error(f"Failed to build {precision} engine.")
        return False
    logging.info(f"Engine built in {end_time - start_time:.2f} sec.")

    # engine_final_path defined earlier in function
    logging.info(f"Saving {precision} engine to: {engine_final_path}")
    try:
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
# ... (Main execution block remains the same) ...
if __name__ == "__main__":
    logging.info("--- Starting Language Model TensorRT Engine Build Process ---")
    os.makedirs(ENGINE_DIR, exist_ok=True)
    engine_base_path = os.path.join(ENGINE_DIR, ENGINE_PREFIX)

    success_fp16 = False
    if BUILD_FP16:
        logging.info("\n--- Building LM FP16 Engine ---")
        success_fp16 = build_engine(
            LM_ONNX_PATH,
            engine_base_path,
            INPUT_NAMES_FOR_ENGINE,
            PROFILE_SHAPES,
            precision="fp16",
        )
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)

    success_int8 = False
    if BUILD_INT8:
        logging.info("\n--- Building LM INT8 Engine ---")
        success_int8 = build_engine(
            LM_ONNX_PATH,
            engine_base_path,
            INPUT_NAMES_FOR_ENGINE,
            PROFILE_SHAPES,
            precision="int8",
            calibrator_factory=get_lm_calibrator,
            cache_file=LM_CACHE_FILE,
        )
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(5)

    logging.info("\n--- Language Model Build Process Summary ---")
    if BUILD_FP16:
        logging.info(f"FP16 Engine Build: {'SUCCESS' if success_fp16 else 'FAILED'}")
    if BUILD_INT8:
        logging.info(f"INT8 Engine Build: {'SUCCESS' if success_int8 else 'FAILED'}")
    logging.info("Language Model engine building process finished.")
