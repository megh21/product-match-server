import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoImageProcessor
from PIL import Image
import os
import logging
import onnx
import onnxruntime as ort
import inspect  # To check signature
import gc

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)

# --- Configuration ---
MODEL_ID = "OpenGVLab/InternVL3-2B"
OUTPUT_DIR = "models/internvl3_vlm"
ONNX_FILENAME = "internvl3_chat_2b_full.onnx"
IMAGE_PATH = "data/images/10000.jpg"
EXPORT_MAX_LENGTH = 256

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
ONNX_PATH = os.path.join(OUTPUT_DIR, ONNX_FILENAME)

# --- Load Model, Tokenizer, and Image Processor ---
logging.info(f"Loading model: {MODEL_ID}")
try:
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cuda",
    ).eval()
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load model: {e}", exc_info=True)
    exit(1)

# --- Log Model Forward Signature ---
try:
    signature = inspect.signature(model.forward)
    logging.info(f"Model forward signature: {signature}")
    model_forward_params = list(signature.parameters.keys())
    logging.info(f"Model forward parameter names: {model_forward_params}")
except Exception as e:
    logging.warning(f"Could not inspect model signature: {e}")
    model_forward_params = [
        "pixel_values",
        "input_ids",
        "attention_mask",
        "image_flags",
    ]  # Fallback
    logging.warning(f"Using fallback parameter order: {model_forward_params}")

logging.info(f"Loading tokenizer: {MODEL_ID}")
try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    logging.info("Tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load tokenizer: {e}", exc_info=True)
    exit(1)

logging.info(f"Loading image processor: {MODEL_ID}")
try:
    image_processor = AutoImageProcessor.from_pretrained(
        MODEL_ID, trust_remote_code=True
    )
    logging.info("Image processor loaded successfully.")
except Exception as e:
    logging.error(f"Failed to load image processor: {e}", exc_info=True)
    exit(1)

# --- Get Image Context Token ID ---
logging.info("Setting up image context token ID...")
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
if img_context_token_id == tokenizer.unk_token_id:
    logging.warning(f"Token '{IMG_CONTEXT_TOKEN}' not found in tokenizer vocab!")
logging.info(
    f"Using Image context token: '{IMG_CONTEXT_TOKEN}', ID: {img_context_token_id}"
)

# --- Prepare Dummy Input ---
logging.info("Preparing dummy input...")
try:
    image = Image.open(IMAGE_PATH).convert("RGB")
    logging.info(f"Loaded image from {IMAGE_PATH}")
except FileNotFoundError:
    logging.error(f"Image not found at {IMAGE_PATH}. Creating dummy image.")
    img_size = 448
    image = Image.new("RGB", (img_size, img_size))
    logging.info(f"Created dummy image with size {img_size}x{img_size}")
except Exception as e:
    logging.error(f"Error loading image: {e}. Creating dummy image.")
    image = Image.new("RGB", (448, 448))

prompt = f"User: {IMG_CONTEXT_TOKEN}\nDescribe the image.\nAssistant:"
NUM_IMAGE_TOKENS = 256

try:
    # 1. Process Image
    logging.info("Processing image...")
    image_inputs = image_processor(images=image, return_tensors="pt")
    pixel_values = image_inputs["pixel_values"].to(torch.float16)
    logging.info(
        f"Processed pixel_values shape: {pixel_values.shape}, dtype: {pixel_values.dtype}"
    )

    # 2. Process Text
    logging.info("Processing text...")
    logging.info(f"Using max_length: {EXPORT_MAX_LENGTH} for tokenizer during export")
    text_inputs = tokenizer(
        text=prompt,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=EXPORT_MAX_LENGTH,
    )
    input_ids = text_inputs["input_ids"]
    attention_mask = text_inputs["attention_mask"]
    logging.info(f"Processed input_ids shape: {input_ids.shape}")
    logging.info(f"Processed attention_mask shape: {attention_mask.shape}")

    # 3. Create image_flags
    image_flags = torch.ones(1, NUM_IMAGE_TOKENS, dtype=torch.long)
    logging.info(f"Created image_flags with shape: {image_flags.shape}")

    # 4. Define the inputs we will actually pass to the model FORWARD and ONNX EXPORT
    # These MUST match names in the forward signature that we want to provide.
    input_names_for_export = [
        "pixel_values",
        "input_ids",
        "attention_mask",
        "image_flags",
    ]
    logging.info(f"Input names being used for export: {input_names_for_export}")

    # 5. Create the dictionary of inputs to pass and move to device
    inputs_on_device = {
        "pixel_values": pixel_values.to(model.device),
        "input_ids": input_ids.to(model.device),
        "attention_mask": attention_mask.to(model.device),
        "image_flags": image_flags.to(model.device),
    }
    # Log shapes on device
    for name, tensor in inputs_on_device.items():
        logging.info(
            f"  Input '{name}': shape={tensor.shape}, dtype={tensor.dtype}, device={tensor.device}"
        )

    # Create a CPU copy for potential verification later
    inputs_on_cpu = {name: tensor.cpu() for name, tensor in inputs_on_device.items()}

    # 6. Test forward pass using KEYWORD ARGUMENTS
    logging.info("Performing test forward pass using keyword arguments...")
    try:
        with torch.no_grad():
            # Pass arguments by name using **kwargs
            test_output = model(**inputs_on_device)
            logging.info("Test forward pass successful.")
            if hasattr(test_output, "logits"):
                logging.info(f"Test logits shape: {test_output.logits.shape}")
            else:
                logging.warning("Test output structure doesn't have .logits attribute.")
    except Exception as e:
        logging.error(f"Test forward pass failed: {e}", exc_info=True)
        logging.error("Cannot proceed to export.")
        exit(1)


except Exception as e:
    logging.error(f"Failed to prepare inputs: {e}", exc_info=True)
    exit(1)

# --- Define Output Names and Dynamic Axes ---
output_names = ["logits"]

# Ensure dynamic_axes uses the same keys as input_names_for_export
dynamic_axes = {
    "pixel_values": {0: "batch_size"},
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "image_flags": {0: "batch_size"},  # Batch dynamic, image token dim fixed
    "logits": {0: "batch_size", 1: "sequence_length"},
}

# --- Export to ONNX ---
logging.info(f"Exporting model to ONNX: {ONNX_PATH}")
torch.cuda.empty_cache()
gc.collect()

try:
    with torch.no_grad():
        # Pass the dictionary of tensors directly to 'args'
        # Ensure input_names list matches the order of keys if ONNX needs it explicitly
        # (Though usually mapping by name from the dict works for opset 11+)
        torch.onnx.export(
            model,
            args=inputs_on_device,  # Pass dictionary of inputs
            f=ONNX_PATH,
            input_names=input_names_for_export,  # Names corresponding to dict keys
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=14,
            export_params=True,
            do_constant_folding=True,
            # verbose=True # Enable for detailed tracing if needed
        )
    logging.info("ONNX export successful.")
except Exception as e:
    logging.error(f"ONNX export failed: {e}", exc_info=True)
    exit(1)

# --- Verify ONNX Model ---
logging.info("Verifying ONNX model...")
try:
    onnx.checker.check_model(ONNX_PATH)
    logging.info("ONNX model basic check successful.")

    logging.info("Running inference with ONNX Runtime for verification...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # ... (provider check) ...
    ort_session = ort.InferenceSession(ONNX_PATH, providers=providers)

    # --- Get input names FROM the ORT session ---
    ort_input_names = [inp.name for inp in ort_session.get_inputs()]
    ort_output_names = [
        out.name for out in ort_session.get_outputs()
    ]  # Also good to check output names
    logging.info(f"ONNX Runtime session expects inputs: {ort_input_names}")
    logging.info(f"ONNX Runtime session expects outputs: {ort_output_names}")

    # Prepare inputs for ONNX Runtime (use CPU numpy arrays)
    inputs_on_cpu = {name: tensor.cpu() for name, tensor in inputs_on_device.items()}

    # --- Create ort_inputs dict using names from the session ---
    ort_inputs = {}
    # Map the names from our original list to the names ORT expects
    # This assumes the order is the same, which might be fragile.
    # A safer way is to map based on the original intended names if they differ.
    if set(ort_input_names) == set(input_names_for_export):
        logging.info("ORT input names match exported names.")
        # Use the names consistently
        for name in ort_input_names:
            # Get the corresponding numpy array from our prepared dict
            if name in inputs_on_cpu:
                ort_inputs[name] = inputs_on_cpu[name].numpy()
            else:
                logging.error(
                    f"Logic error: Name '{name}' from ORT session not found in prepared CPU inputs."
                )
                raise KeyError(f"Missing input {name} for ORT")
    else:
        logging.warning(
            f"Mismatch between exported names {input_names_for_export} and ORT session names {ort_input_names}."
        )
        logging.warning(
            "Attempting to map based on order (might be unreliable). Inspect with Netron if verification fails."
        )
        if len(ort_input_names) == len(input_names_for_export):
            for i, ort_name in enumerate(ort_input_names):
                original_name = input_names_for_export[i]
                logging.info(
                    f"Mapping exported '{original_name}' (index {i}) to ORT '{ort_name}'"
                )
                ort_inputs[ort_name] = inputs_on_cpu[original_name].numpy()
        else:
            logging.error(
                "Cannot map inputs due to length mismatch. Inspect ONNX graph."
            )
            raise ValueError("Input name mapping failed.")

    # Run session (use the output names reported by ORT session too)
    ort_outputs_list = ort_session.run(ort_output_names, ort_inputs)

    # Map outputs back to expected names if needed (assuming single output 'logits')
    if "logits" in ort_output_names:
        logits_index = ort_output_names.index("logits")
        ort_logits_output = ort_outputs_list[logits_index]
        logging.info(f"ONNX Runtime output 'logits' shape: {ort_logits_output.shape}")
    else:
        logging.warning(
            f"Output 'logits' not found in ORT output names: {ort_output_names}. Using first output."
        )
        ort_logits_output = ort_outputs_list[0]  # Assume first output is logits
        logging.info(f"ONNX Runtime first output shape: {ort_logits_output.shape}")

    # Compare with PyTorch output
    with torch.no_grad():
        torch_outputs = model(**inputs_on_device)
    # ... (rest of comparison logic using ort_logits_output) ...
    if hasattr(torch_outputs, "logits"):
        torch_logits = torch_outputs.logits
    # ...
    torch_logits_np = torch_logits.cpu().numpy()
    # ... compare torch_logits_np and ort_logits_output ...

except ImportError:
    logging.warning("ONNX or ONNX Runtime not installed. Skipping verification.")
except Exception as e:
    logging.warning(f"ONNX verification failed: {e}", exc_info=True)


# --- Cleanup ---
logging.info("Cleaning up GPU memory...")
del model, tokenizer, image_processor, inputs_on_device, inputs_on_cpu
torch.cuda.empty_cache()
gc.collect()

logging.info(f"VLM ONNX export process completed. Model saved to {ONNX_PATH}")
