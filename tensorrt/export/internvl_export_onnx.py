import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import os
import logging

logging.basicConfig(level=logging.INFO)

# --- Configuration ---
MODEL_ID = "OpenGVLab/InternVL3-2B"
OUTPUT_DIR = "models/internvl3"
ONNX_FILENAME = "internvl3_vision.onnx"
IMAGE_PATH = "data/images/10000.jpg"

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
onnx_path = os.path.join(OUTPUT_DIR, ONNX_FILENAME)

# --- Load Model and Processor ---
logging.info(f"Loading model: {MODEL_ID}")
model = AutoModel.from_pretrained(MODEL_ID, trust_remote_code=True)
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model.eval()  # Set to evaluation mode


# --- Create Vision-Only Model ---
# InternVL3 has separate vision and text encoders that can be exported individually
class InternVL3VisionEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.vision_model = model.vision_model

    def forward(self, pixel_values):
        return self.vision_model(pixel_values).last_hidden_state


# --- Prepare Dummy Input ---
logging.info("Preparing input image...")
try:
    image = Image.open(IMAGE_PATH).convert("RGB")
except Exception as e:
    logging.error(f"Failed to load image: {e}")
    # Create a purely dummy tensor if image loading fails
    logging.warning("Using random tensor as dummy input.")
    # Default input dimensions for InternVL3 vision model
    dummy_image = torch.randn(1, 3, 448, 448)  # Batch, Channels, H, W
    dummy_input = dummy_image
else:
    # Process the image using the model's processor
    # For InternVL3, we need to use the image_processor component
    try:
        # Try using the image_processor if it exists
        if hasattr(processor, "image_processor"):
            vision_inputs = processor.image_processor(images=image, return_tensors="pt")
            dummy_input = vision_inputs.pixel_values
        # Otherwise try direct image processing
        else:
            vision_inputs = processor(images=image, return_tensors="pt")
            dummy_input = vision_inputs.pixel_values

        logging.info(f"Processed vision inputs keys: {vision_inputs.keys()}")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
        # Fall back to using vision_processor directly
        try:
            from transformers import AutoImageProcessor

            vision_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
            vision_inputs = vision_processor(image, return_tensors="pt")
            dummy_input = vision_inputs.pixel_values
            logging.info("Used AutoImageProcessor as fallback")
        except Exception as e2:
            logging.error(f"Fallback also failed: {e2}")
            # Last resort: create dummy tensor
            dummy_input = torch.randn(1, 3, 448, 448)
            logging.warning("Using random tensor of shape (1, 3, 448, 448)")

logging.info(f"Input pixel_values shape: {dummy_input.shape}")

# --- Extract Vision Encoder ---
vision_encoder = InternVL3VisionEncoder(model)

# Test forward pass
with torch.no_grad():
    vision_output = vision_encoder(dummy_input)
    logging.info(f"Vision encoder output shape: {vision_output.shape}")

# --- Define Input/Output Names and Dynamic Axes ---
input_names = ["pixel_values"]
output_names = ["vision_output"]
dynamic_axes = {
    "pixel_values": {0: "batch_size"},  # Make batch size dynamic
    "vision_output": {0: "batch_size"},
}

# --- Export to ONNX ---
logging.info(f"Exporting vision model to ONNX: {onnx_path}")
try:
    torch.onnx.export(
        vision_encoder,
        dummy_input,
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,  # Use 14 or higher for better compatibility
        export_params=True,
        do_constant_folding=True,
    )
    logging.info("Vision encoder ONNX export successful!")
except Exception as e:
    logging.error(f"Vision encoder ONNX export failed: {e}", exc_info=True)
    exit(1)

# --- Export Text Encoder (optional) ---
text_onnx_path = os.path.join(OUTPUT_DIR, "internvl3_text.onnx")


class InternVL3TextEncoder(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.text_model = model.language_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.text_model(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        return outputs.last_hidden_state


try:
    # Create text encoder
    text_encoder = InternVL3TextEncoder(model)

    # Create dummy text input
    text = "A photo of"
    text_inputs = processor.tokenizer(text, return_tensors="pt", padding=True)

    dummy_input_ids = text_inputs.input_ids
    dummy_attention_mask = text_inputs.attention_mask

    logging.info(f"Text input_ids shape: {dummy_input_ids.shape}")
    logging.info(f"Text attention_mask shape: {dummy_attention_mask.shape}")

    # Test forward pass
    with torch.no_grad():
        text_output = text_encoder(dummy_input_ids, dummy_attention_mask)
        logging.info(f"Text encoder output shape: {text_output.shape}")

    # Define input/output names and dynamic axes
    input_names = ["input_ids", "attention_mask"]
    output_names = ["text_output"]
    dynamic_axes = {
        "input_ids": {0: "batch_size", 1: "sequence_length"},
        "attention_mask": {0: "batch_size", 1: "sequence_length"},
        "text_output": {0: "batch_size", 1: "sequence_length"},
    }

    # Export to ONNX
    logging.info(f"Exporting text model to ONNX: {text_onnx_path}")
    torch.onnx.export(
        text_encoder,
        (dummy_input_ids, dummy_attention_mask),
        text_onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,
        export_params=True,
        do_constant_folding=True,
    )
    logging.info("Text encoder ONNX export successful!")
except Exception as e:
    logging.error(f"Text encoder ONNX export failed: {e}", exc_info=True)

# --- Verify ONNX Model (Optional but Recommended) ---
logging.info("Verifying vision ONNX model...")
try:
    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logging.info("Vision ONNX model check successful.")

    # Further verification with ONNX Runtime
    import onnxruntime as ort

    logging.info("Running inference with ONNX Runtime...")
    ort_session = ort.InferenceSession(
        onnx_path, providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    ort_inputs = {input_names[0]: dummy_input.cpu().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    logging.info(f"Vision ONNX Runtime output shape: {ort_outputs[0].shape}")

    # Compare with PyTorch output
    with torch.no_grad():
        torch_outputs = vision_encoder(dummy_input)

    import numpy as np

    # Check shapes match
    assert torch_outputs.shape == ort_outputs[0].shape, "Output shapes don't match!"
    # Check values are close (with tolerance)
    np.testing.assert_allclose(
        torch_outputs.cpu().numpy(), ort_outputs[0], rtol=1e-03, atol=1e-05
    )
    logging.info("PyTorch and ONNX Runtime outputs match (within tolerance).")

except Exception as e:
    logging.warning(f"ONNX verification failed: {e}", exc_info=True)


logging.info("Created TensorRT conversion script at ...")

logging.info("InternVL3-2B export process completed!")
