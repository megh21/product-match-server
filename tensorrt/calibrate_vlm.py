import tensorrt as trt
import torch
from PIL import Image
import os
import glob
import logging

# Use explicit imports now that export uses them
from transformers import AutoTokenizer, AutoImageProcessor


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)

# --- Configuration ---
MODEL_ID = "OpenGVLab/InternVL3-2B"  # Match export script
# --- Calibration Data Source --- (Keep your chosen method)
CALIBRATION_IMAGE_DIR = "calibration_data/images/"
USE_LOCAL_DIR = True
DATASET_NAME = "lambdalabs/pokemon-blip-captions"  # Example dataset
DATASET_SPLIT = "train"
IMAGE_COLUMN = "image"
# --- Calibration Settings ---
NUM_CALIBRATION_IMAGES = 100
CALIBRATION_BATCH_SIZE = 4  # Adjust based on GPU memory
CACHE_FILE = "models/internvl3_vlm/calibration.cache"  # Match output dir

# --- Expected Input Info (CRITICAL: Must match ONNX export) ---
# Order matters if implementation relies on it, names matter for clarity
INPUT_NAMES = ["pixel_values", "input_ids", "attention_mask", "image_flags"]
# Define shapes for allocation (Using CALIBRATION_BATCH_SIZE)
NUM_IMAGE_TOKENS = 256  # From export script logs/model config
MAX_SEQ_LEN_CALIB = 256  # Match EXPORT_MAX_LENGTH used in export for consistency

INPUT_SHAPES = {
    "pixel_values": (
        CALIBRATION_BATCH_SIZE,
        3,
        448,
        448,
    ),  # Check export log if different
    "input_ids": (CALIBRATION_BATCH_SIZE, MAX_SEQ_LEN_CALIB),
    "attention_mask": (CALIBRATION_BATCH_SIZE, MAX_SEQ_LEN_CALIB),
    "image_flags": (
        CALIBRATION_BATCH_SIZE,
        NUM_IMAGE_TOKENS,
    ),  # Added image_flags shape
}
# Define dtypes for allocation
INPUT_DTYPES = {
    "pixel_values": torch.float16,
    "input_ids": torch.int64,
    "attention_mask": torch.int64,
    "image_flags": torch.int64,  # Added image_flags dtype
}
# --- Special Token ---
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"  # Match export script

# --- Helper function to load and preprocess a batch ---
try:
    # Load tokenizer and processor separately
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    logging.info("Loaded tokenizer and image processor for calibration.")
    # Get token ID needed for prompt
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    if img_context_token_id == tokenizer.unk_token_id:
        logging.warning(f"Calibration: Token '{IMG_CONTEXT_TOKEN}' not found!")
except Exception as e:
    logging.error(f"Failed to load processor/tokenizer {MODEL_ID}: {e}")
    exit(1)


def get_calibration_files(image_dir):
    # ... (function remains the same) ...
    files = (
        glob.glob(os.path.join(image_dir, "*.jpg"))
        + glob.glob(os.path.join(image_dir, "*.png"))
        + glob.glob(os.path.join(image_dir, "*.jpeg"))
    )
    if not files:
        raise FileNotFoundError(f"No calibration images found in {image_dir}")
    logging.info(f"Found {len(files)} potential calibration images in {image_dir}.")
    return files


def preprocess_batch_for_calib(image_paths_or_items, is_local_dir):
    """Loads images, creates prompts, processes image/text, creates image_flags."""
    batch_images = []
    # Use a fixed, simple prompt format for calibration batches
    calib_prompt_template = (
        f"User: {IMG_CONTEXT_TOKEN}\nDescribe the image.\nAssistant:"
    )

    for item in image_paths_or_items:
        try:
            if is_local_dir:
                image = Image.open(item).convert("RGB")
            else:
                image = item[IMAGE_COLUMN].convert("RGB")
            batch_images.append(image)
        except Exception as e:
            logging.warning(
                f"Skipping calibration item due to error: {e} (Item: {item})"
            )
            continue

    if not batch_images:
        return None

    num_images_in_batch = len(batch_images)
    prompts = [calib_prompt_template] * num_images_in_batch

    try:
        # 1. Process Images
        image_inputs = image_processor(images=batch_images, return_tensors="pt")
        pixel_values = image_inputs["pixel_values"]  # Keep as torch tensor

        # 2. Process Text
        text_inputs = tokenizer(
            text=prompts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LEN_CALIB,
        )
        input_ids = text_inputs["input_ids"]
        attention_mask = text_inputs["attention_mask"]

        # 3. Create image_flags for the batch
        image_flags = torch.ones(
            num_images_in_batch, NUM_IMAGE_TOKENS, dtype=torch.long
        )

        # Return dictionary matching INPUT_NAMES
        return {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "image_flags": image_flags,
        }

    except Exception as e:
        logging.error(
            f"Error during calibration batch preprocessing: {e}", exc_info=True
        )
        return None


# --- Calibrator Class ---
class VLMCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file):
        trt.IInt8EntropyCalibrator2.__init__(self)
        self.cache_file = cache_file
        self.batch_idx = 0
        self.device_inputs = {}  # Dictionary to hold device buffers {'input_name': tensor}

        # --- Load Calibration Data Source ---
        # ... (loading logic remains the same) ...
        if USE_LOCAL_DIR:
            self.image_files = get_calibration_files(CALIBRATION_IMAGE_DIR)
            self.data_source = self.image_files[
                : min(NUM_CALIBRATION_IMAGES, len(self.image_files))
            ]
            self.is_local_dir = True
            logging.info(
                f"Using {len(self.data_source)} images from local directory for calibration."
            )
        else:
            # ... (HF dataset loading logic) ...
            pass  # Add HF dataset loading if needed

        self.num_items = len(self.data_source)
        if self.num_items == 0:
            raise ValueError("No calibration data loaded.")
        self.max_batches = (
            self.num_items + CALIBRATION_BATCH_SIZE - 1
        ) // CALIBRATION_BATCH_SIZE
        logging.info(
            f"Total calibration items: {self.num_items}, Batches: {self.max_batches}"
        )

        # --- Allocate GPU Buffers ---
        logging.info("Allocating GPU buffers for calibration...")
        for name in INPUT_NAMES:  # Use the updated list including image_flags
            shape = INPUT_SHAPES[name]
            dtype = INPUT_DTYPES[name]
            self.device_inputs[name] = torch.zeros(
                shape, dtype=dtype, device="cuda"
            ).contiguous()
            logging.info(
                f"  Allocated buffer for '{name}' with shape {shape} and dtype {dtype}"
            )

    def get_batch_size(self):
        return CALIBRATION_BATCH_SIZE

    def get_batch(self, names):  # 'names' is the list of input names TRT expects
        if self.batch_idx >= self.max_batches:
            # logging.info("Calibration dataset exhausted.") # Reduce log noise
            return None

        start_idx = self.batch_idx * CALIBRATION_BATCH_SIZE
        end_idx = min(start_idx + CALIBRATION_BATCH_SIZE, self.num_items)
        current_batch_items = self.data_source[start_idx:end_idx]
        actual_batch_size = len(current_batch_items)

        if actual_batch_size == 0:
            self.batch_idx += 1
            return self.get_batch(names)

        # logging.info(f"Processing calib batch {self.batch_idx + 1}/{self.max_batches}") # Reduce log noise
        processed_batch_dict = preprocess_batch_for_calib(
            current_batch_items, self.is_local_dir
        )

        if processed_batch_dict is None:
            logging.warning(
                f"Skipping calibration batch {self.batch_idx + 1} due to preprocessing errors."
            )
            self.batch_idx += 1
            return self.get_batch(names)

        # --- Copy data to allocated GPU buffers ---
        pointers = []
        # Use the names TRT requests to ensure correct order, but access data from our dict
        for name in names:
            if name not in processed_batch_dict:
                logging.error(
                    f"Input '{name}' expected by TensorRT not found in processed batch data!"
                )
                return None  # Critical error

            tensor = processed_batch_dict[
                name
            ]  # Get the tensor from our processed dict
            device_buffer = self.device_inputs[name]
            target_dtype = INPUT_DTYPES[name]
            target_shape = INPUT_SHAPES[name]  # Full batch shape

            # Ensure tensor is on CPU before checking shape/dtype/copying
            tensor = tensor.cpu()

            # Convert to target dtype (e.g., pixel_values to float16)
            # This assumes preprocess_batch_for_calib returns tensors in exportable dtypes
            # but explicit conversion here adds safety.
            tensor = tensor.to(target_dtype)

            # --- Batch Padding Handling ---
            current_shape = tensor.shape
            padded_tensor = tensor
            if current_shape[0] < CALIBRATION_BATCH_SIZE:
                pad_size = CALIBRATION_BATCH_SIZE - current_shape[0]
                padding_shape = (pad_size,) + current_shape[1:]
                # Use zeros for padding
                padding = torch.zeros(padding_shape, dtype=target_dtype)
                padded_tensor = torch.cat((tensor, padding), dim=0)
            elif current_shape[0] > CALIBRATION_BATCH_SIZE:
                logging.warning(
                    f"Tensor batch size {current_shape[0]} > expected {CALIBRATION_BATCH_SIZE} for {name}"
                )
                padded_tensor = tensor[:CALIBRATION_BATCH_SIZE]  # Truncate

            # --- Shape Validation against Buffer ---
            if padded_tensor.shape != target_shape:
                logging.error(
                    f"Shape mismatch for '{name}': Tensor shape {padded_tensor.shape} vs Buffer shape {target_shape}. Check MAX_SEQ_LEN_CALIB ({MAX_SEQ_LEN_CALIB}) or NUM_IMAGE_TOKENS ({NUM_IMAGE_TOKENS})."
                )
                # Add specific padding/truncation for non-batch dims if feasible,
                # but usually indicates config error.
                # Example: Pad sequence dimension for input_ids/attention_mask
                if (
                    name in ["input_ids", "attention_mask"]
                    and len(padded_tensor.shape) > 1
                    and padded_tensor.shape[1] < target_shape[1]
                ):
                    pad_width = target_shape[1] - padded_tensor.shape[1]
                    padded_tensor = torch.nn.functional.pad(
                        padded_tensor, (0, pad_width)
                    )  # Pad last dim
                # Truncate if needed
                if (
                    name in ["input_ids", "attention_mask"]
                    and len(padded_tensor.shape) > 1
                    and padded_tensor.shape[1] > target_shape[1]
                ):
                    padded_tensor = padded_tensor[:, : target_shape[1]]

                # Re-check after potential fix
                if padded_tensor.shape != target_shape:
                    logging.error(
                        f"Could not fix shape mismatch for '{name}'. Stopping calibration."
                    )
                    return None

            # Copy data to the pre-allocated GPU buffer
            device_buffer.copy_(padded_tensor)
            pointers.append(device_buffer.data_ptr())

        self.batch_idx += 1
        if self.batch_idx % 10 == 0:  # Log progress occasionally
            logging.info(
                f"Calibration batch {self.batch_idx}/{self.max_batches} processed."
            )

        return pointers

    def read_calibration_cache(self):
        # ... (function remains the same) ...
        if os.path.exists(self.cache_file):
            logging.info(f"Reading calibration cache: {self.cache_file}")
            try:
                with open(self.cache_file, "rb") as f:
                    return f.read()
            except Exception as e:
                logging.error(f"Failed to read cache file {self.cache_file}: {e}")
                return None
        else:
            logging.info("Calibration cache not found.")
            return None

    def write_calibration_cache(self, cache):
        # ... (function remains the same) ...
        logging.info(
            f"Writing calibration cache: {self.cache_file} ({len(cache)} bytes)"
        )
        try:
            os.makedirs(os.path.dirname(self.cache_file) or ".", exist_ok=True)
            with open(self.cache_file, "wb") as f:
                f.write(cache)
            logging.info("Cache written successfully.")
        except Exception as e:
            logging.error(f"Failed to write cache file {self.cache_file}: {e}")

    def __del__(self):
        # ... (function remains the same) ...
        logging.info("Calibrator finished.")
        if hasattr(self, "device_inputs"):  # Check if allocation happened
            del self.device_inputs
        torch.cuda.empty_cache()


# --- Main execution (for testing calibrator standalone) ---
if __name__ == "__main__":
    logging.info("--- Testing Calibrator Initialization ---")
    try:
        calibrator_test = VLMCalibrator(CACHE_FILE)
        logging.info("Calibrator initialized successfully.")
        logging.info(f"Expecting input names: {INPUT_NAMES}")

        # Optionally, fetch one batch
        # logging.info("--- Testing get_batch ---")
        # test_pointers = calibrator_test.get_batch(INPUT_NAMES) # Pass expected names
        # if test_pointers and len(test_pointers) == len(INPUT_NAMES):
        #      logging.info(f"get_batch returned {len(test_pointers)} pointers successfully.")
        # else:
        #      logging.warning(f"get_batch test failed or returned unexpected number of pointers.")

        logging.info(
            "Calibrator script setup appears OK. Ready for use by build_engine_vlm.py."
        )

    except Exception as e:
        logging.error(f"Error during calibrator test: {e}", exc_info=True)
