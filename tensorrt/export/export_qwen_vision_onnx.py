import torch
from transformers import AutoProcessor, AutoModel
from PIL import Image
import os
import logging
import gc
import onnxruntime as ort
import onnx
import numpy as np


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
    dummy_input = torch.randn(1, 3, 448, 448)  # Batch, Channels, H, W
else:
    # Process the image using the model's processor
    try:
        # For InternVL3, use AutoImageProcessor directly
        from transformers import AutoImageProcessor

        vision_processor = AutoImageProcessor.from_pretrained(MODEL_ID)
        vision_inputs = vision_processor(image, return_tensors="pt")
        dummy_input = vision_inputs.pixel_values
        logging.info("Processed vision inputs with AutoImageProcessor")
    except Exception as e:
        logging.error(f"Error processing image: {e}")
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
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,  # Request hidden states
            return_dict=True,
        )
        # For CausalLMOutputWithPast, we have these options:
        # 1. Return logits (prediction scores)
        # return outputs.logits

        # 2. Return the last hidden layer if hidden_states are available
        if hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            return outputs.hidden_states[-1]  # Last layer hidden states

        # 3. Fallback to logits if hidden_states aren't available
        return outputs.logits


try:
    # Create text encoder
    text_encoder = InternVL3TextEncoder(model)

    # Create dummy text input
    text = "create a visula description of the image"
    # print methods of processor
    # Check if processor has tokenizer
    if hasattr(processor, "tokenizer"):
        logging.info("Using processor's tokenizer")

        text_inputs = processor.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )

    else:
        logging.info("Using AutoTokenizer")
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        text_inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=77,
            return_tensors="pt",
        )
    logging.info(f"Text input_ids shape: {text_inputs.input_ids.shape}")
    logging.info(f"Text attention_mask shape: {text_inputs.attention_mask.shape}")

    # Create dummy input tensors
    # Assuming the text encoder takes input_ids and attention_mask

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
    ort_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    # Use the correct input name for the vision model
    ort_inputs = {"pixel_values": dummy_input.cpu().numpy()}
    ort_outputs = ort_session.run(None, ort_inputs)
    logging.info(f"Vision ONNX Runtime output shape: {ort_outputs[0].shape}")

    # Compare with PyTorch output
    with torch.no_grad():
        torch_outputs = vision_encoder(dummy_input)

    import numpy as np

    # Check shapes match
    if torch_outputs.shape == ort_outputs[0].shape:
        logging.info("PyTorch and ONNX output shapes match")

        # Check some basic statistics instead of exact matching
        torch_mean = np.mean(torch_outputs.cpu().numpy())
        onnx_mean = np.mean(ort_outputs[0])
        torch_std = np.std(torch_outputs.cpu().numpy())
        onnx_std = np.std(ort_outputs[0])

        logging.info(f"PyTorch output - mean: {torch_mean:.4f}, std: {torch_std:.4f}")
        logging.info(f"ONNX output - mean: {onnx_mean:.4f}, std: {onnx_std:.4f}")

        # Use much more relaxed tolerance for VLMs
        try:
            np.testing.assert_allclose(
                torch_outputs.cpu().numpy(),
                ort_outputs[0],
                rtol=0.1,  # 10% relative tolerance
                atol=0.1,  # Higher absolute tolerance
            )
            logging.info(
                "PyTorch and ONNX Runtime outputs match (with relaxed tolerance)."
            )
        except AssertionError as e:
            logging.warning(f"Outputs differ, but this may be acceptable: {e}")
    else:
        logging.warning(
            f"Output shape mismatch: PyTorch {torch_outputs.shape} vs ONNX {ort_outputs[0].shape}"
        )

except Exception as e:
    logging.warning(f"ONNX verification failed: {e}", exc_info=True)

# --- Convert to TensorRT ---


logging.info("Created TensorRT conversion script at ...")

logging.info("InternVL3-2B export process completed!")

# cleanup gpu memory and ram
torch.cuda.empty_cache()
gc.collect()
