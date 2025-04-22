import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import os
import logging
from transformers import AutoTokenizer
from dotenv import load_dotenv
import pandas as pd  # To load style metadata for calibration data

logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Configuration ---
ONNX_PATH = "models/text_encoder/text_encoder.onnx"
CACHE_FILE = "models/text_encoder/calibration.cache"
MODEL_ID = os.getenv("TEXT_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
MAX_SEQ_LENGTH = 128  # Must match export script

# Calibration Data Config

STYLES_CSV = os.getenv("STYLES_CSV", "data/styles.csv")  # Make sure path is correct
CALIBRATION_DATA_SOURCE = "product_names"  # or "huggingface_dataset"


NUM_CALIBRATION_SAMPLES = 500  # Number of text samples for calibration
BATCH_SIZE = 16  # Calibration batch size

INPUT_NAMES = ["input_ids", "attention_mask"]
# Shape for allocating GPU buffer (Batch, SeqLen)
INPUT_SHAPE_IDS = (BATCH_SIZE, MAX_SEQ_LENGTH)
INPUT_SHAPE_MASK = (BATCH_SIZE, MAX_SEQ_LENGTH)

# --- Helper function to load and preprocess data ---
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)


def load_calibration_data():
    if CALIBRATION_DATA_SOURCE == "product_names":
        logging.info(f"Loading product names from {STYLES_CSV}")
        if not os.path.exists(STYLES_CSV):
            raise FileNotFoundError(f"Styles CSV not found at {STYLES_CSV}")
        df = pd.read_csv(STYLES_CSV)  # Handle potential errors
        # Use 'productDisplayName' or similar relevant text column
        if "productDisplayName" not in df.columns:
            raise ValueError("Column 'productDisplayName' not found in styles.csv")
        # Drop missing names and ensure diversity (optional: sample)
        texts = df["productDisplayName"].dropna().unique().tolist()
        if len(texts) < NUM_CALIBRATION_SAMPLES:
            logging.warning(f"Only {len(texts)} unique product names found, using all.")
            return texts
        else:
            # Sample randomly if many names exist
            indices = np.random.choice(
                len(texts), NUM_CALIBRATION_SAMPLES, replace=False
            )
            return [texts[i] for i in indices]
    else:
        raise ValueError(f"Unknown CALIBRATION_DATA_SOURCE: {CALIBRATION_DATA_SOURCE}")


def preprocess_batch(text_batch):
    inputs = tokenizer(
        text_batch,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="np",
    )
    input_ids = inputs["input_ids"].astype(
        np.int64
    )  # Ensure correct dtype (often int64 for transformers)
    attention_mask = inputs["attention_mask"].astype(np.int64)

    # Pad batch to fixed BATCH_SIZE if needed (last batch might be smaller)
    num_in_batch = input_ids.shape[0]
    if num_in_batch < BATCH_SIZE:
        pad_size = BATCH_SIZE - num_in_batch
        pad_ids = np.zeros((pad_size, MAX_SEQ_LENGTH), dtype=input_ids.dtype)
        pad_mask = np.zeros((pad_size, MAX_SEQ_LENGTH), dtype=attention_mask.dtype)
        input_ids = np.concatenate((input_ids, pad_ids), axis=0)
        attention_mask = np.concatenate((attention_mask, pad_mask), axis=0)

    return input_ids, attention_mask


# --- Calibrator Class ---
class TextCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, dataset_texts, cache_file):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.texts = dataset_texts
        self.batch_idx = 0
        self.max_batches = (len(self.texts) + BATCH_SIZE - 1) // BATCH_SIZE
        self.device_inputs = []  # Pointers to GPU buffers for inputs

        self.allocate_buffers()
        logging.info(
            f"Using {len(self.texts)} text samples in {self.max_batches} batches for calibration."
        )

    def allocate_buffers(self):
        # Allocate GPU memory for input_ids and attention_mask
        ids_size = trt.volume(INPUT_SHAPE_IDS) * np.dtype(np.int64).itemsize
        mask_size = trt.volume(INPUT_SHAPE_MASK) * np.dtype(np.int64).itemsize
        self.d_input_ids = cuda.mem_alloc(ids_size)
        self.d_attention_mask = cuda.mem_alloc(mask_size)
        self.device_inputs = [
            int(self.d_input_ids),
            int(self.d_attention_mask),
        ]  # Store pointers as int
        logging.info(
            f"Allocated {(ids_size + mask_size) / (1024**2):.2f} MB on GPU for calibration inputs."
        )

    def get_batch_size(self):
        return BATCH_SIZE

    def get_batch(self, names):  # names correspond to INPUT_NAMES
        if self.batch_idx >= self.max_batches:
            logging.info("Calibration complete.")
            return None  # Signal completion

        start_idx = self.batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, len(self.texts))
        batch_texts = self.texts[start_idx:end_idx]

        logging.info(f"Calibration batch {self.batch_idx + 1}/{self.max_batches}")
        input_ids, attention_mask = preprocess_batch(batch_texts)

        # Copy data to GPU
        cuda.memcpy_htod(self.d_input_ids, np.ascontiguousarray(input_ids))
        cuda.memcpy_htod(self.d_attention_mask, np.ascontiguousarray(attention_mask))

        self.batch_idx += 1
        return self.device_inputs  # Return list of device pointers

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            logging.info(f"Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        logging.info("Calibration cache not found.")
        return None

    def write_calibration_cache(self, cache):
        logging.info(f"Writing calibration cache: {self.cache_file}")
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def free_buffers(self):
        # pycuda.autoinit usually handles freeing
        logging.info("Buffers conceptually freed (likely handled by pycuda.autoinit).")


# --- Main function (can be called by build script) ---
def get_calibrator():
    logging.info("Loading calibration data for text model...")
    calibration_texts = load_calibration_data()
    return TextCalibrator(calibration_texts, CACHE_FILE)


if __name__ == "__main__":
    # This part is mainly for testing the calibrator setup
    logging.info("Initializing text calibrator...")
    calibrator = get_calibrator()
    logging.info("Testing one batch fetch...")
    batch_data_ptrs = calibrator.get_batch(INPUT_NAMES)
    if batch_data_ptrs:
        logging.info(
            f"Calibrator fetched batch successfully. Pointers: {batch_data_ptrs}"
        )
    else:
        logging.warning("Calibrator get_batch returned None (check data/batches).")
    # The actual cache generation happens when this calibrator is used during engine build.
    logging.info(
        "Text Calibrator script initialized. Ready for use during engine build."
    )
