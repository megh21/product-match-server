import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
from PIL import Image
import os
import glob  # To find local files
import logging
from transformers import AutoImageProcessor  # Use the correct processor
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)
load_dotenv()

# --- Configuration ---
MODEL_ID = "OpenGVLab/InternVL3-2B"  # For loading processor

#
CORRECT_BASE_DIR = "models/internvl3"  # Directory where ONNX exists
VISION_ONNX_PATH = os.path.join(CORRECT_BASE_DIR, "internvl3_vision.onnx")
VISION_CACHE_FILE = os.path.join(CORRECT_BASE_DIR, "internvl3_vision_calibration.cache")
OUTPUT_DIR_VISION = CORRECT_BASE_DIR  # Use the same base directory

# Calibration Settings
CALIBRATION_IMAGE_DIR = "data/images"  # Your local image directory
NUM_CALIBRATION_IMAGES = 100
CALIBRATION_BATCH_SIZE = 8

# Vision Input Details
VISION_INPUT_NAME = "pixel_values"
IMAGE_SIZE = 448
VISION_INPUT_DTYPE = np.float16
INPUT_SHAPE = (CALIBRATION_BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)

# --- Image Processor Loading ---
try:
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    logging.info(f"Loaded AutoImageProcessor for {MODEL_ID}")
except Exception as e:
    logging.error(f"Failed to load AutoImageProcessor {MODEL_ID}: {e}", exc_info=True)
    raise


def load_local_vision_calibration_batch(image_files, indices):
    """Loads and preprocesses a batch of local image files."""
    batch_pil_images = []
    for idx in indices:
        try:
            img_path = image_files[idx]
            img = Image.open(img_path).convert("RGB")
            batch_pil_images.append(img)
        except Exception as e:
            logging.warning(f"Error loading calibration image {image_files[idx]}: {e}")
            continue

    if not batch_pil_images:
        logging.error("No images successfully loaded in the current calibration batch.")
        raise ValueError("Failed to load any images for the batch")

    try:
        inputs = image_processor(images=batch_pil_images, return_tensors="np")
        pixel_values = inputs[VISION_INPUT_NAME].astype(VISION_INPUT_DTYPE)
    except Exception as e:
        logging.error(f"Error during image preprocessing for batch: {e}", exc_info=True)
        raise

    current_batch_size = pixel_values.shape[0]
    if current_batch_size < CALIBRATION_BATCH_SIZE:
        padding_size = CALIBRATION_BATCH_SIZE - current_batch_size
        padding_shape = (padding_size,) + pixel_values.shape[1:]
        padding = np.zeros(padding_shape, dtype=pixel_values.dtype)
        pixel_values = np.concatenate((pixel_values, padding), axis=0)

    if pixel_values.shape != INPUT_SHAPE:
        logging.error(
            f"FATAL: Vision batch shape mismatch! Got {pixel_values.shape}, expected {INPUT_SHAPE}."
        )
        raise ValueError("Vision calibration batch shape mismatch")

    return pixel_values


class VisionCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, image_file_list, cache_file):  # Takes list of file paths
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.image_files = image_file_list  # Store the list of files
        self.num_images = len(self.image_files)
        self.batch_idx = 0
        self.max_batches = (
            self.num_images + CALIBRATION_BATCH_SIZE - 1
        ) // CALIBRATION_BATCH_SIZE
        self.device_input = None

        if self.num_images == 0:
            raise ValueError(
                "No image files found for calibration in the provided list."
            )

        self.allocate_buffers()
        logging.info(
            f"VisionCalibrator: Using {self.num_images} images in {self.max_batches} batches."
        )

    def allocate_buffers(self):
        if self.device_input is not None:
            return
        element_size = trt.volume(INPUT_SHAPE) * np.dtype(VISION_INPUT_DTYPE).itemsize
        self.device_input = cuda.mem_alloc(element_size)
        logging.info(
            f"VisionCalibrator: Allocated {element_size / (1024**2):.2f} MB GPU memory."
        )

    def get_batch_size(self):
        return CALIBRATION_BATCH_SIZE

    def get_batch(self, names):  # names is list, e.g., ['pixel_values']
        if self.batch_idx >= self.max_batches:
            return None  # Signal end

        start_idx = self.batch_idx * CALIBRATION_BATCH_SIZE
        end_idx = min(start_idx + CALIBRATION_BATCH_SIZE, self.num_images)
        indices = list(range(start_idx, end_idx))

        try:
            current_batch_np = load_local_vision_calibration_batch(
                self.image_files, indices
            )
        except Exception as e:
            logging.error(f"VisionCalibrator: Failed batch {self.batch_idx + 1}: {e}")
            return None  # Signal error

        if self.device_input is None:
            self.allocate_buffers()  # Re-alloc if needed

        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(current_batch_np))
        self.batch_idx += 1
        return [int(self.device_input)]  # Return pointer as int

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            try:
                logging.info(f"Reading vision calibration cache: {self.cache_file}")
                with open(self.cache_file, "rb") as f:
                    return f.read()
            except Exception as e:
                logging.error(f"Failed to read cache: {e}")
        return None

    def write_calibration_cache(self, cache):
        try:
            logging.info(f"Writing vision calibration cache: {self.cache_file}")
            os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(cache)
        except Exception as e:
            logging.error(f"Failed to write cache: {e}")


def get_vision_calibrator():
    """Finds local images and creates the VisionCalibrator."""
    logging.info(f"Searching for calibration images in: {CALIBRATION_IMAGE_DIR}")
    try:
        image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
        all_files = []
        for ext in image_extensions:
            all_files.extend(glob.glob(os.path.join(CALIBRATION_IMAGE_DIR, ext)))

        if not all_files:
            logging.error(f"No image files found in {CALIBRATION_IMAGE_DIR}")
            return None

        num_available = len(all_files)
        num_to_use = min(NUM_CALIBRATION_IMAGES, num_available)
        if num_to_use < NUM_CALIBRATION_IMAGES:
            logging.warning(
                f"Requested {NUM_CALIBRATION_IMAGES} calibration images, found {num_available}. Using {num_to_use}."
            )

        # random.shuffle(all_files)
        calibration_files = all_files[:num_to_use]
        logging.info(f"Using {len(calibration_files)} images for vision calibration.")

    except Exception as e:
        logging.error(f"Failed to find/select local image files: {e}", exc_info=True)
        return None

    os.makedirs(OUTPUT_DIR_VISION, exist_ok=True)
    # Pass the corrected cache file path
    return VisionCalibrator(calibration_files, VISION_CACHE_FILE)


# --- Main block for testing ---
if __name__ == "__main__":
    logging.info("--- Testing Vision Calibrator Setup (Local Data) ---")
    calibrator = get_vision_calibrator()
    if calibrator:
        logging.info("VisionCalibrator created. Testing one batch fetch...")
        batch_ptr = calibrator.get_batch([VISION_INPUT_NAME])
        if batch_ptr:
            logging.info("First batch fetched successfully.")
        else:
            logging.warning("Failed to fetch first batch.")
    else:
        logging.error("Failed to create VisionCalibrator.")
    logging.info("calibrate_vision.py finished initialization test.")
