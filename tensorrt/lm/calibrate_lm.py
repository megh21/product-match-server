import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import os
import logging
import gc
from transformers import AutoModelForCausalLM

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)

# --- Configuration ---
MODEL_ID = "OpenGVLab/InternVL3-2B"
LM_CACHE_FILE = "models/internvl3_lm/internvl3_language_calibration.cache"
OUTPUT_DIR_LM = "models/internvl3_lm"

# Calibration Settings
NUM_CALIBRATION_BATCHES = 50
CALIBRATION_BATCH_SIZE = 1

CALIBRATION_SEQ_LENGTH = 256

# LM Input Details - Critical: Match the exported LM ONNX
LM_INPUT_NAMES = ["inputs_embeds", "attention_mask"]
# Determine hidden size (Need to load config or model briefly)
try:
    temp_config = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, low_cpu_mem_usage=True, trust_remote_code=True
    ).config
    if hasattr(temp_config, "hidden_size"):
        HIDDEN_SIZE = temp_config.hidden_size
    elif hasattr(temp_config, "text_config") and hasattr(
        temp_config.text_config, "hidden_size"
    ):
        HIDDEN_SIZE = temp_config.text_config.hidden_size
    else:
        raise ValueError("Cannot find hidden_size in config")
    logging.info(f"LM Calibrator: Using hidden size: {HIDDEN_SIZE}")
    del temp_config
    gc.collect()
except Exception as e:
    logging.error(
        f"Could not determine hidden size for LM calibrator: {e}. Setting fallback."
    )
    HIDDEN_SIZE = 1536  # Fallback based on previous logs

# Define shapes for allocation (Batch Size, SeqLen, HiddenSize) or (Batch Size, SeqLen)
INPUT_SHAPES = {
    "inputs_embeds": (CALIBRATION_BATCH_SIZE, CALIBRATION_SEQ_LENGTH, HIDDEN_SIZE),
    "attention_mask": (CALIBRATION_BATCH_SIZE, CALIBRATION_SEQ_LENGTH),
}
# Define dtypes - MUST match the expected input types of the LM ONNX graph
INPUT_DTYPES = {
    "inputs_embeds": np.float16,  # LM likely expects float16 embeddings
    "attention_mask": np.int64,  # Attention mask is usually int64
}


# --- LM Calibrator Class ---
class LanguageModelCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, cache_file):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.batch_idx = 0
        self.max_batches = NUM_CALIBRATION_BATCHES
        self.device_inputs = {}  # Dictionary to hold device pointers

        self.allocate_buffers()
        logging.info(
            f"LanguageModelCalibrator: Generating {self.max_batches} random batches."
        )

    def allocate_buffers(self):
        """Allocates GPU memory for a calibration batch."""
        for name in LM_INPUT_NAMES:
            if name in self.device_inputs:
                continue  # Avoid re-allocation
            shape = INPUT_SHAPES[name]
            dtype = INPUT_DTYPES[name]
            element_size = trt.volume(shape) * np.dtype(dtype).itemsize
            self.device_inputs[name] = cuda.mem_alloc(element_size)
            logging.info(
                f"LM Calibrator: Allocated {element_size / (1024**2):.2f} MB for '{name}'."
            )

    def get_batch_size(self):
        return CALIBRATION_BATCH_SIZE

    def get_batch(
        self, names
    ):  # names is list, e.g., ['inputs_embeds', 'attention_mask']
        if self.batch_idx >= self.max_batches:
            return None  # Signal end

        batch_data_cpu = {}
        # Generate random inputs_embeds
        embed_shape = INPUT_SHAPES["inputs_embeds"]
        embed_dtype = INPUT_DTYPES["inputs_embeds"]
        batch_data_cpu["inputs_embeds"] = np.random.rand(*embed_shape).astype(
            embed_dtype
        )

        # Generate attention_mask (can be all ones for calibration)
        mask_shape = INPUT_SHAPES["attention_mask"]
        mask_dtype = INPUT_DTYPES["attention_mask"]
        batch_data_cpu["attention_mask"] = np.ones(mask_shape, dtype=mask_dtype)

        # Copy data to device buffers
        pointers = []
        for name in names:  # Use the order TRT requests
            if name not in self.device_inputs:
                self.allocate_buffers()  # Allocate if missing
            if name not in batch_data_cpu:
                logging.error(f"LM Calibrator: Cannot find generated data for '{name}'")
                return None
            cuda.memcpy_htod(
                self.device_inputs[name], np.ascontiguousarray(batch_data_cpu[name])
            )
            pointers.append(int(self.device_inputs[name]))

        self.batch_idx += 1
        if self.batch_idx % 10 == 0:
            logging.info(
                f"LM Calibration batch {self.batch_idx}/{self.max_batches} generated."
            )
        return pointers

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            try:
                logging.info(f"Reading LM calibration cache: {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    return f.read()
            except Exception as e:
                logging.error(f"Failed to read LM cache: {e}")
        return None

    def write_calibration_cache(self, cache):
        try:
            logging.info(f"Writing LM calibration cache: {self.cache_file}")
            os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(cache)
        except Exception as e:
            logging.error(f"Failed to write LM cache: {e}")


# --- Factory Function ---
def get_lm_calibrator():
    """Creates the LanguageModelCalibrator."""
    os.makedirs(OUTPUT_DIR_LM, exist_ok=True)
    return LanguageModelCalibrator(LM_CACHE_FILE)


# --- Main block for testing ---
if __name__ == "__main__":
    logging.info("--- Testing Language Model Calibrator Setup ---")
    calibrator = get_lm_calibrator()
    if calibrator:
        logging.info("LMCalibrator created. Testing one batch fetch...")
        batch_ptr = calibrator.get_batch(LM_INPUT_NAMES)  # Pass expected names
        if batch_ptr and len(batch_ptr) == len(LM_INPUT_NAMES):
            logging.info("First LM batch fetched successfully.")
        else:
            logging.warning("Failed to fetch first LM batch.")
    else:
        logging.error("Failed to create LMCalibrator.")
    logging.info("calibrate_lm.py finished initialization test.")
