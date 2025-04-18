import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit # Handles CUDA context initialization/cleanup
import numpy as np
from PIL import Image
import requests
import os
import logging
from transformers import CLIPProcessor
from datasets import load_dataset # To get calibration data

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
ONNX_PATH = "models/clip_vision/clip_vision.onnx"
CACHE_FILE = "models/clip_vision/calibration.cache"
MODEL_ID = "openai/clip-vit-base-patch32" # Match the exported model
WORKSPACE_MB = 4096
# Calibration Data Config
DATASET_NAME = "restufiqih/fashion-product" # Example: Use a relevant image dataset
DATASET_SPLIT = "train"
NUM_CALIBRATION_IMAGES = 200 # Number of images to use for calibration
BATCH_SIZE = 8 # Batch size for calibration

INPUT_NAME = "pixel_values" # Must match the name in the ONNX file
# Input shape expected by the ONNX model (check export script/netron)
# Typically (batch, channel, height, width) for CLIP vision
INPUT_SHAPE = (BATCH_SIZE, 3, 224, 224) # Batch size here is calibration batch size

# --- Helper function to load and preprocess data ---
processor = CLIPProcessor.from_pretrained(MODEL_ID)

def load_calibration_batch(dataset, indices):
    # Get the correct image key from the dataset
    # Most datasets use 'image' or 'pixel_values' instead of 'img'
    try:
        # Try different common image keys
        if 'image' in dataset[0]:
            batch_images = [dataset[i]['image'] for i in indices]
        elif 'pixel_values' in dataset[0]:
            batch_images = [dataset[i]['pixel_values'] for i in indices]
        else:
            # List all available keys for debugging
            available_keys = dataset[0].keys()
            raise KeyError(f"No image key found. Available keys: {available_keys}")
            
    except Exception as e:
        logging.error(f"Error accessing dataset: {str(e)}")
        raise

    # Rest of the function remains the same
    inputs = processor(images=batch_images, return_tensors="np", padding=True)
    pixel_values = inputs['pixel_values'].astype(np.float32)

    # Ensure the batch matches the expected INPUT_SHAPE for calibration
    # Handle cases where the last batch might be smaller
    if pixel_values.shape[0] < BATCH_SIZE:
        # Pad the batch if necessary (e.g., repeat last image or use zeros)
        padding_size = BATCH_SIZE - pixel_values.shape[0]
        padding = np.zeros((padding_size, *pixel_values.shape[1:]), dtype=pixel_values.dtype)
        # Or replicate last item: padding = np.repeat(pixel_values[-1:], padding_size, axis=0)
        pixel_values = np.concatenate((pixel_values, padding), axis=0)

    # Ensure correct shape (it should be already)
    if pixel_values.shape != INPUT_SHAPE:
         # This might happen if processor output changes or config is wrong
         logging.warning(f"Batch shape mismatch: Got {pixel_values.shape}, expected {INPUT_SHAPE}. Reshaping/Padding might be needed.")
         # Add reshaping/error handling logic if necessary
         # For now, we assume processor output matches
         pass # Assuming processor gives correct shape

    return pixel_values

# Add this after loading the dataset to inspect its structure
def print_dataset_info(dataset):
    # Get first item to inspect structure
    first_item = dataset[0]
    logging.info(f"Dataset features: {dataset.features}")
    logging.info(f"First item keys: {first_item.keys()}")

# --- Calibrator Class ---
class ClipVisionCalibrator(trt.IInt8MinMaxCalibrator):
    def __init__(self, dataset, cache_file):
        trt.IInt8MinMaxCalibrator.__init__(self)
        self.cache_file = cache_file
        self.dataset = dataset
        self.batch_idx = 0
        self.max_batches = (NUM_CALIBRATION_IMAGES + BATCH_SIZE - 1) // BATCH_SIZE
        self.device_input = None # Device memory allocation

        # Pre-allocate GPU memory
        self.allocate_buffers()
        logging.info(f"Using {NUM_CALIBRATION_IMAGES} images in {self.max_batches} batches for calibration.")

    def allocate_buffers(self):
        # Calculate size needed for one batch
        element_size = trt.volume(INPUT_SHAPE) * trt.float32.itemsize
        self.device_input = cuda.mem_alloc(element_size)
        logging.info(f"Allocated {element_size / (1024**2):.2f} MB on GPU for calibration input.")


    def get_batch_size(self):
        return BATCH_SIZE

    def get_batch(self, names): # 'names' argument is list of input names
        if self.batch_idx >= self.max_batches:
            logging.info("Calibration complete. No more batches.")
            return None # Signal completion

        start_idx = self.batch_idx * BATCH_SIZE
        end_idx = min(start_idx + BATCH_SIZE, NUM_CALIBRATION_IMAGES)
        indices = range(start_idx, end_idx)

        logging.info(f"Calibration batch {self.batch_idx + 1}/{self.max_batches}")
        current_batch = load_calibration_batch(self.dataset, indices)

        # Copy data from host (CPU) to device (GPU)
        cuda.memcpy_htod(self.device_input, np.ascontiguousarray(current_batch))

        self.batch_idx += 1
        return [int(self.device_input)] # Return list of device pointers

    def read_calibration_cache(self):
        if os.path.exists(self.cache_file):
            logging.info(f"Reading calibration cache: {self.cache_file}")
            with open(self.cache_file, "rb") as f:
                return f.read()
        logging.info("Calibration cache not found.")
        return None

    def write_calibration_cache(self, cache):
         logging.info(f"Writing calibration cache: {self.cache_file}")
         with open(self.cache_file, "wb") as f:
             f.write(cache)

    def free_buffers(self):
         if self.device_input:
              # Note: pycuda.autoinit usually handles freeing, but explicit is safer in scripts
              try:
                  # self.device_input.free() # autoinit should handle this
                  pass
              except Exception as e:
                  logging.warning(f"Error freeing buffer (might be managed by autoinit): {e}")
         logging.info("Buffers conceptually freed (likely handled by pycuda.autoinit).")


# --- Main Calibration Execution ---
def generate_calibration_cache():
    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    logging.info(f"Parsing ONNX model: {ONNX_PATH}")
    if not parser.parse_from_file(ONNX_PATH):
        for error in range(parser.num_errors):
            logging.error(f"ONNX Parser Error: {parser.get_error(error)}")
        raise ValueError("Failed to parse the ONNX file.")

    # Ensure network input shape matches calibration shape if not dynamic
    # profile = builder.create_optimization_profile()
    # profile.set_shape(INPUT_NAME, min=INPUT_SHAPE, opt=INPUT_SHAPE, max=INPUT_SHAPE) # If fixed batch calib
    # config.add_optimization_profile(profile)

    logging.info("Loading calibration dataset...")
    # Use streaming=True for large datasets if memory is a concern
    calib_dataset = load_dataset(DATASET_NAME, split=DATASET_SPLIT, streaming=False)
    # If streaming=True, you'd need to adapt the Calibrator to iterate differently
    # calib_dataset_list = list(calib_dataset.take(NUM_CALIBRATION_IMAGES)) # Convert streamed to list


    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 4 * (1024**3)) # 4GB workspace limit, adjust as needed

    # --- Configure INT8 Calibration ---
    config.set_flag(trt.BuilderFlag.INT8)
    calibrator = ClipVisionCalibrator(calib_dataset, CACHE_FILE)
    config.int8_calibrator = calibrator
    logging.info("INT8 Calibration configured.")

    # --- Build Engine JUST to generate cache (optional, cache generated during calibration) ---
    # We don't actually need to build the engine here if only cache is needed,
    # the calibrator's get_batch will be called by trtexec later.
    # However, running this ensures the calibrator works.
    logging.info("Running calibration process (engine build not saved here)...")
    # serialized_engine = builder.build_serialized_network(network, config)
    # if serialized_engine is None:
    #     logging.error("Engine build failed during calibration test run.")
    # else:
    #     logging.info("Calibration process simulation successful (cache should be written).")
    # del serialized_engine # Free memory

    # The cache is written by the calibrator instance when TRT uses it.
    # We just need to ensure the calibrator runs correctly. Let's simulate one batch fetch.
    # names = [network.get_input(i).name for i in range(network.num_inputs)]
    # batch_data_ptr = calibrator.get_batch(names)
    # if batch_data_ptr:
    #     logging.info("Calibrator get_batch test successful.")
    # else:
    #      logging.warning("Calibrator get_batch test returned None (might be expected if max_batches=0).")


    # --- Cleanup ---
    # Calibrator buffers usually freed by pycuda.autoinit when script exits
    # calibrator.free_buffers() # Call explicitly if not using autoinit or for clarity

    # Check if cache file was created by the (potential) build process above or previous runs
    if os.path.exists(CACHE_FILE):
         logging.info(f"Calibration cache process seems complete. Cache file exists: {CACHE_FILE}")
    else:
         logging.warning(f"Calibration cache file {CACHE_FILE} not found. Ensure the calibrator runs successfully (e.g., via trtexec).")

def build_engine(onnx_path, engine_path, precision="fp16", calib_dataset=None, cache_file=None):
    logger = trt.Logger(trt.Logger.VERBOSE if os.environ.get("TRT_VERBOSE") else trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    logging.info(f"Parsing ONNX model: {onnx_path}")
    with open(onnx_path, 'rb') as model_file:
        if not parser.parse(model_file.read()):
            for error in range(parser.num_errors):
                logging.error(f"ONNX Parser Error: {parser.get_error(error)}")
            raise ValueError("Failed to parse the ONNX file.")
    logging.info("ONNX model parsed successfully.")

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, WORKSPACE_MB * (1024**2))
    logging.info(f"Workspace size set to {WORKSPACE_MB} MB")

    # Add optimization profile for dynamic shapes
    profile = builder.create_optimization_profile()
    min_shape = (1, 3, 224, 224)      # Minimum batch size
    opt_shape = INPUT_SHAPE           # Optimal/default batch size
    max_shape = (32, 3, 224, 224)     # Maximum batch size
    
    profile.set_shape(INPUT_NAME, min_shape, opt_shape, max_shape)
    config.add_optimization_profile(profile)
    logging.info(f"Added optimization profile with shapes - min: {min_shape}, opt: {INPUT_SHAPE}, max: {max_shape}")

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
            logging.info("INT8 mode enabled.")
            if calib_dataset is None or cache_file is None:
                raise ValueError("Calibration dataset and cache file needed for INT8.")
            logging.info("Setting up INT8 calibrator...")
            config.int8_calibrator = ClipVisionCalibrator(calib_dataset, cache_file) # Use the class from calibrate script
            # Optional: Set calibration profile if needed (usually automatic)
            # profile = builder.create_optimization_profile()
            # ... set shapes for profile ...
            # config.add_optimization_profile(profile)

        else:
            logging.warning("INT8 not supported on this platform, using FP32.")
            precision = "fp32" # Fallback

    # --- Build Engine ---
    logging.info(f"Building TensorRT engine ({precision})... This may take a while.")
    # Note: build_serialized_network is preferred for saving/loading
    serialized_engine = builder.build_serialized_network(network, config)
    if serialized_engine is None:
        raise RuntimeError("Failed to build TensorRT engine.")
    logging.info("TensorRT engine built successfully.")

    # --- Save Engine ---
    logging.info(f"Saving engine to: {engine_path}")
    os.makedirs(os.path.dirname(engine_path), exist_ok=True)
    with open(engine_path, "wb") as f:
        f.write(serialized_engine)
    logging.info("Engine saved.")


if __name__ == "__main__":
    # This script primarily defines the calibrator.
    # We call a dummy function here just to trigger the imports and class definition.
    # The actual calibration cache generation happens when trtexec USES this calibrator.
    # If you want this script ITSELF to generate the cache, you need to fully
    # build the engine with the config pointing to this calibrator instance.
    # For now, let's just confirm the setup looks okay.
    logging.info("Calibrator script initialized. Ready for use with trtexec.")
    if __name__ == "__main__":
        logging.info("Loading calibration dataset for INT8 build...")
        DATASET_NAME_BUILD = "restufiqih/fashion-product"
        DATASET_SPLIT_BUILD = "train"
        NUM_IMAGES_BUILD = 200 # Match NUM_CALIBRATION_IMAGES
        calib_dataset_main = load_dataset(DATASET_NAME_BUILD, split=DATASET_SPLIT_BUILD, streaming=False)
        
        # Add dataset inspection
        print_dataset_info(calib_dataset_main)
        
        # --- Ensure these match the settings at the top of the file ---
        # --- --- --- --- --- --- --- --- --- --- --- --- --- --- --- ---

        # Load the dataset instance that will be passed to the calibrator during build
        calib_dataset_main = load_dataset(DATASET_NAME_BUILD, split=DATASET_SPLIT_BUILD, streaming=False)
        # If using streaming=True, you might need this instead (but ensure calibrator handles list):
        # calib_dataset_main = list(load_dataset(DATASET_NAME_BUILD, split=DATASET_SPLIT_BUILD, streaming=True).take(NUM_IMAGES_BUILD))

        # Print dataset information
        print_dataset_info(calib_dataset_main)

        # Define paths for the output engine and potentially the cache
        ENGINE_PATH_INT8 = "models/clip_vision/clip_vision_int8_py.engine"
        # CACHE_FILE is already defined at the top of your script

        # Call the build function for INT8 precision
        build_engine(
            onnx_path=ONNX_PATH,
            engine_path=ENGINE_PATH_INT8,
            precision="int8",
            calib_dataset=calib_dataset_main, # Pass the loaded dataset
            cache_file=CACHE_FILE           # Pass the cache file path
        )

        logging.info("INT8 Engine building process finished.")