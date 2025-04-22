import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import os
import logging

# Use AutoImageProcessor as identified in the export logs
from transformers import AutoImageProcessor
from datasets import load_dataset  # To get calibration data
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Configuration ---
# Define paths specific to this model component
MODEL_ID = "OpenGVLab/InternVL3-2B"  # Model ID used for processor
ONNX_PATH = "models/internvl3/internvl3_vision.onnx"
CACHE_FILE = "models/internvl3/internvl3_vision_calibration.cache"

# Calibration Data Config
DATASET_NAME = "restufiqih/fashion-product"  # Or your preferred image dataset
DATASET_SPLIT = "train"
NUM_CALIBRATION_IMAGES = 200  # Number of images for calibration
BATCH_SIZE = 8  # Calibration batch size

# Input details - Must match the ONNX model and processor used
INPUT_NAME = "pixel_values"
IMAGE_SIZE = 448  # For InternVL3 Vision Model
INPUT_SHAPE = (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)  # Batch, Channel, Height, Width
INPUT_DTYPE = np.float32  # AutoImageProcessor usually outputs float32

# --- Helper function to load and preprocess data ---
try:
    # Use the processor that worked in the export script's fallback
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    logging.info(f"Loaded AutoImageProcessor for {MODEL_ID}")
except Exception as e:
    logging.error(f"Failed to load AutoImageProcessor {MODEL_ID}: {e}")
    raise


def load_calibration_batch(dataset, indices):
    """Loads and preprocesses a batch of images for calibration."""
    try:
        # Adjust key based on dataset structure ('image', 'img', etc.)
        image_key = "image"  # Default, check your dataset features if error
        if image_key not in dataset.features:
            keys = list(dataset.features.keys())
            logging.error(
                f"Dataset key '{image_key}' not found. Available keys: {keys}"
            )
            raise KeyError(
                f"Cannot find suitable image key in dataset features: {keys}"
            )

        # Load PIL images, ensure RGB
        batch_pil_images = [dataset[i][image_key].convert("RGB") for i in indices]
    except Exception as e:
        logging.error(f"Error loading images from dataset: {e}", exc_info=True)
        raise

    # Preprocess using the loaded AutoImageProcessor
    try:
        inputs = image_processor(
            images=batch_pil_images,
            return_tensors="np",
            do_rescale=True,
            do_normalize=True,
        )
        pixel_values = inputs["pixel_values"].astype(INPUT_DTYPE)
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}", exc_info=True)
        raise

    # Ensure the batch matches the expected INPUT_SHAPE for calibration
    # Handle cases where the last batch might be smaller
    current_batch_size = pixel_values.shape[0]
    if current_batch_size < BATCH_SIZE:
        padding_size = BATCH_SIZE - current_batch_size
        padding_shape = (padding_size, *pixel_values.shape[1:])
        # Pad with zeros
        padding = np.zeros(padding_shape, dtype=pixel_values.dtype)
        pixel_values = np.concatenate((pixel_values, padding), axis=0)

    # Verify final shape
    if pixel_values.shape != INPUT_SHAPE:
        logging.error(
            f"FATAL: Batch shape mismatch! Got {pixel_values.shape}, expected {INPUT_SHAPE}. Check processor/IMAGE_SIZE."
        )
        raise ValueError("Calibration batch shape mismatch")

    return pixel_values


# --- Calibrator Class ---
class InternVL3VisionCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, dataset, cache_file):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.dataset = dataset  # Expecting HuggingFace dataset object
        self.batch_idx = 0
        self.max_batches = (NUM_CALIBRATION_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE
        self.device_input = None  # Device memory allocation

        # Pre-allocate GPU memory
        self.allocate_buffers()
        logging.info(
            f"Using {NUM_CALIBRATION_IMAGES} images in {self.max_batches} batches for calibration."
        )

    def allocate_buffers(self):
        """Allocates GPU memory for a calibration batch."""
        if self.device_input is not None:  # Avoid re-allocation if already done
            return
        element_size = trt.volume(INPUT_SHAPE) * np.dtype(INPUT_DTYPE).itemsize
        self.device_input = cuda.mem_alloc(element_size)
        logging.info(
            f"Allocated {element_size / (1024**2):.2f} MB on GPU for calibration input."
        )

    def get_batch_size(self):
        """Returns the calibration batch size."""
        return BATCH_SIZE

    def get_batch(self, names):
        """Gets the next calibration batch."""
        if self.batch_idx >= self.max_batches:
            logging.info("Calibration complete. No more batches.")
            return None  # Signal completion

        start_idx = self.batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, NUM_CALIBRATION_IMAGES)
        indices = list(range(start_idx, end_idx))  # Indices for HF dataset

        logging.info(f"Calibration batch {self.batch_idx + 1}/{self.max_batches}")
        try:
            current_batch_np = load_calibration_batch(self.dataset, indices)
        except Exception as e:
            logging.error(
                f"Failed to load/preprocess calibration batch {self.batch_idx + 1}: {e}"
            )
            # Returning None signals error to TensorRT build process
            return None

        # Ensure buffers are allocated (might be needed if __init__ failed)
        if self.device_input is None:
            logging.error("GPU buffer not allocated.")
            return None

        # Copy data from host (CPU) to device (GPU)
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(current_batch_np))

        self.batch_idx += 1
        # Return list of device pointers (cast to int)
        return [int(self.device_input)]

    def read_calibration_cache(self):
        """Reads the calibration cache."""
        if os.path.exists(self.cache_file):
            try:
                logging.info(f"Reading calibration cache: {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    return f.read()
            except Exception as e:
                logging.error(
                    f"Failed to read calibration cache {self.cache_file}: {e}"
                )
        logging.info("Calibration cache not found or failed to read.")
        return None

    def write_calibration_cache(self, cache):
        """Writes the calibration cache."""
        try:
            logging.info(f"Writing calibration cache: {self.cache_file}")
            os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(cache)
        except Exception as e:
            logging.error(f"Failed to write calibration cache {self.cache_file}: {e}")

    # No explicit free_buffers needed when using pycuda.autoinit usually


# --- Main function (for use by build script) ---
def get_internvl3_vision_calibrator():
    """Factory function to create the calibrator."""
    logging.info(
        f"Loading dataset '{DATASET_NAME}' for InternVL3 Vision calibration..."
    )
    try:
        calib_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=False)
        # Select a subset for calibration if the dataset is large
        if len(calib_dataset) > NUM_CALIBRATION_IMAGES * 2:  # Heuristic check
            calib_dataset = calib_dataset.select(range(NUM_CALIBRATION_IMAGES))
            logging.info(
                f"Selected first {NUM_CALIBRATION_IMAGES} samples for calibration."
            )
    except Exception as e:
        logging.error(f"Failed to load dataset '{DATASET_NAME}': {e}", exc_info=True)
        return None

    logging.info(
        f"Using dataset split: {calib_dataset.info.dataset_name} - {calib_dataset.info.split}"
    )
    logging.info(f"Dataset features: {calib_dataset.features}")

    return InternVL3VisionCalibrator(calib_dataset, CACHE_FILE)


if __name__ == "__main__":
    # Test the calibrator setup
    logging.info("Initializing InternVL3 Vision calibrator...")
    calibrator = get_internvl3_vision_calibrator()
    if calibrator:
        logging.info("Testing one batch fetch...")
        # Need input name from ONNX model
        batch_data_ptr = calibrator.get_batch([INPUT_NAME])
        if batch_data_ptr:
            logging.info(
                f"Calibrator fetched batch successfully. Pointer: {batch_data_ptr[0]}"
            )
        else:
            logging.warning(
                "Calibrator get_batch returned None (check data/batches/logs)."
            )
    else:
        logging.error("Failed to create calibrator instance.")
    logging.info("InternVL3 Vision Calibrator script initialized.")
