import tensorrt as trt
import pycuda.driver as cuda
import numpy as np
import os
import logging
from transformers import AutoModelForCausalLM  # To get the transform
from datasets import load_dataset
from dotenv import load_dotenv
import torch

logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Configuration ---
# Must match the exported ONNX model's source VLM
VLM_MODEL_ID = os.getenv("VLM_MODEL_ID", "Qwen/Qwen-VL-Chat")
ONNX_PATH = f"models/{VLM_MODEL_ID.split('/')[-1]}_vision/qwen_vision_encoder.onnx"
CACHE_FILE = (
    f"models/{VLM_MODEL_ID.split('/')[-1]}_vision/qwen_vision_calibration.cache"
)

# Calibration Data Config (Use an image dataset)
DATASET_NAME = "restufiqih/fashion-product"  # Or another suitable image dataset
DATASET_SPLIT = "train"
NUM_CALIBRATION_IMAGES = 200
BATCH_SIZE = 8  # Calibration batch size

INPUT_NAME = "pixel_values"  # Must match ONNX input name
# Determine input shape from Qwen model config or export script (e.g., 448x448)
# Important: Must be consistent!
IMAGE_SIZE = 448
INPUT_SHAPE = (BATCH_SIZE, 3, IMAGE_SIZE, IMAGE_SIZE)  # (Batch, Channel, Height, Width)
INPUT_DTYPE = np.float16  # Qwen often uses float16 internally


# --- Helper function to load model's image transform ---
# We need the specific image preprocessing used by the model
@torch.no_grad()
def get_image_transform(model_id):
    try:
        # Load only the vision tower's transform if possible, or load full model briefly
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.float16,
        )
        transform = model.transformer.visual.image_transform
        del model  # Free memory
        torch.cuda.empty_cache()
        logging.info(f"Loaded image transform for {model_id}")
        return transform
    except Exception as e:
        logging.error(
            f"Failed to load image transform: {e}. Calibration might be inaccurate.",
            exc_info=True,
        )
        # Fallback to generic transform? Risky. Best to fix loading.
        return None


image_transform = get_image_transform(VLM_MODEL_ID)
if image_transform is None:
    exit(1)  # Cannot proceed without correct transform


# --- Helper function to load and preprocess data ---
def load_calibration_batch(dataset, indices):
    try:
        # Adapt based on the dataset's image key ('image', 'img', etc.)
        batch_pil_images = [dataset[i]["image"].convert("RGB") for i in indices]
    except KeyError:
        logging.error(
            f"Dataset does not contain 'image' key. Available keys: {dataset.features.keys()}"
        )
        raise
    except Exception as e:
        logging.error(f"Error loading images from dataset: {e}")
        raise

    try:
        batch_tensor = image_transform(batch_pil_images).to(
            INPUT_DTYPE
        )  # Apply transform
    except Exception as e:
        logging.error(f"Error applying image transform: {e}")

        raise

    # Ensure correct shape and handle potential padding for the last batch
    if batch_tensor.shape[0] < BATCH_SIZE:
        pad_size = BATCH_SIZE - batch_tensor.shape[0]
        padding_shape = (pad_size, *batch_tensor.shape[1:])
        # Pad with zeros, ensure correct type
        padding = torch.zeros(padding_shape, dtype=batch_tensor.dtype)
        batch_tensor = torch.cat((batch_tensor, padding), dim=0)

    # Final check on shape
    if batch_tensor.shape != INPUT_SHAPE:
        logging.warning(
            f"Batch tensor shape mismatch: Got {batch_tensor.shape}, expected {INPUT_SHAPE}. Check transform/config."
        )

    return batch_tensor.cpu().numpy()  # Return as numpy array


# --- Calibrator Class ---
class QwenVisionCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, dataset, cache_file):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.dataset = dataset  # This should be the loaded Hugging Face dataset object
        self.batch_idx = 0
        self.max_batches = (NUM_CALIBRATION_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE
        self.device_input = None

        self.allocate_buffers()
        logging.info(
            f"Using {NUM_CALIBRATION_IMAGES} images in {self.max_batches} batches for calibration."
        )

    def allocate_buffers(self):
        element_size = trt.volume(INPUT_SHAPE) * np.dtype(INPUT_DTYPE).itemsize
        self.device_input = cuda.mem_alloc(element_size)
        logging.info(
            f"Allocated {element_size / (1024**2):.2f} MB on GPU for calibration input."
        )

    def get_batch_size(self):
        return BATCH_SIZE

    def get_batch(self, names):  # names = [INPUT_NAME]
        if self.batch_idx >= self.max_batches:
            logging.info("Calibration complete.")
            return None

        start_idx = self.batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, NUM_CALIBRATION_IMAGES)
        # Get indices for Hugging Face dataset (needs integer list)
        indices = list(range(start_idx, end_idx))

        logging.info(f"Calibration batch {self.batch_idx + 1}/{self.max_batches}")
        current_batch_np = load_calibration_batch(self.dataset, indices)

        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(current_batch_np))

        self.batch_idx += 1
        return [int(self.device_input)]  # Return list of device pointers

    def read_calibration_cache(self):
        # Implementation same as other calibrators
        if os.path.exists(self.cache_file):
            logging.info(f"Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        return None

    def write_calibration_cache(self, cache):
        logging.info(f"Writing calibration cache: {self.cache_file}")
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        with open(self.cache_file, "wb") as f:
            f.write(cache)

    def free_buffers(self):
        # pycuda.autoinit
        logging.info("Buffers conceptually freed.")


# --- Main function (for use by build script) ---
def get_qwen_vision_calibrator():
    logging.info(f"Loading dataset {DATASET_NAME} for Qwen Vision calibration...")
    # Load the actual dataset object
    calib_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=False)
    # If  streaming=True
    # calib_dataset = list(calib_dataset.take(NUM_CALIBRATION_IMAGES))
    return QwenVisionCalibrator(calib_dataset, CACHE_FILE)


if __name__ == "__main__":
    logging.info("Initializing Qwen Vision calibrator...")
    calibrator = get_qwen_vision_calibrator()
    logging.info("Testing one batch fetch...")
    batch_ptr = calibrator.get_batch([INPUT_NAME])
    if batch_ptr:
        logging.info("Calibrator fetched batch successfully.")
    else:
        logging.warning("Calibrator get_batch returned None.")
    logging.info("Qwen Vision Calibrator script ready.")
