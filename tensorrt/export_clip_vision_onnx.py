import torch
from transformers import CLIPVisionModelWithProjection, CLIPProcessor
from PIL import Image
import requests
import os
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
MODEL_ID = "openai/clip-vit-base-patch32"
OUTPUT_DIR = "models/clip_vision"
ONNX_FILENAME = "clip_vision.onnx"
# Example input for tracing - use a real image URL or path
# IMAGE_URL = "http://images.cocodataset.org/val2017/000000039769.jpg"
# Or use a local file:
IMAGE_PATH = "data/images/10000.jpg"

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
onnx_path = os.path.join(OUTPUT_DIR, ONNX_FILENAME)

# --- Load Model and Processor ---
logging.info(f"Loading model: {MODEL_ID}")
model = CLIPVisionModelWithProjection.from_pretrained(MODEL_ID)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval() # Set to evaluation mode

# --- Prepare Dummy Input ---
logging.info("Preparing dummy input...")
try:
    image = Image.open(IMAGE_PATH)
    # image = Image.open(requests.get(IMAGE_URL, stream=True).raw)
except Exception as e:
    logging.error(f"Failed to load image: {e}")
    # Create a purely dummy tensor if image loading fails
    logging.warning("Using random tensor as dummy input.")
    dummy_image = torch.randn(1, 3, 224, 224) # Batch, Channels, H, W
    inputs = {'pixel_values': dummy_image}
else:
    inputs = processor(images=image, return_tensors="pt")

dummy_input = inputs['pixel_values'] # Shape: (batch_size, 3, 224, 224)
logging.info(f"Dummy input shape: {dummy_input.shape}")

# --- Define Input/Output Names and Dynamic Axes ---
input_names = ["pixel_values"]
output_names = ["image_embeds"] # Output name from CLIPVisionModelWithProjection
dynamic_axes = {
    "pixel_values": {0: "batch_size"}, # Make batch size dynamic
    "image_embeds": {0: "batch_size"}
}

# --- Export to ONNX ---
logging.info(f"Exporting model to ONNX: {onnx_path}")
try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14, # Use 14 or higher for better compatibility
        export_params=True,
        do_constant_folding=True,
    )
    logging.info("ONNX export successful!")
except Exception as e:
    logging.error(f"ONNX export failed: {e}", exc_info=True)
    exit(1)

# --- Verify ONNX Model (Optional but Recommended) ---
logging.info("Verifying ONNX model...")
try:
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logging.info("ONNX model check successful.")

    # Further verification with ONNX Runtime
    import onnxruntime as ort
    logging.info("Running inference with ONNX Runtime...")
    ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    ort_inputs = {input_names[0]: dummy_input.cpu().numpy()}
    ort_outputs = ort_session.run(output_names, ort_inputs)
    logging.info(f"ONNX Runtime output shape: {ort_outputs[0].shape}")

    # Compare with PyTorch output (optional, needs careful tolerance check)
    with torch.no_grad():
        torch_outputs = model(dummy_input)
        torch_embeds = torch_outputs.image_embeds
    # np.testing.assert_allclose(torch_embeds.cpu().numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05)
    # logging.info("PyTorch and ONNX Runtime outputs match (within tolerance).")

except Exception as e:
    logging.warning(f"ONNX verification failed: {e}", exc_info=True)

logging.info(f"CLIP Vision model exported to {onnx_path}")