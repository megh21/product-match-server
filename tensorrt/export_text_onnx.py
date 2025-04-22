import torch
from transformers import AutoTokenizer, AutoModel
import os
import logging
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Configuration ---
MODEL_ID = os.getenv("TEXT_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
OUTPUT_DIR = "models/text_encoder"
ONNX_FILENAME = "text_encoder.onnx"
# Max sequence length the model supports or you want to enforce
MAX_SEQ_LENGTH = 128

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
onnx_path = os.path.join(OUTPUT_DIR, ONNX_FILENAME)

# --- Load Model and Tokenizer ---
logging.info(f"Loading model and tokenizer: {MODEL_ID}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModel.from_pretrained(MODEL_ID)
model.eval()  # Set to evaluation mode

# --- Prepare Dummy Input ---
logging.info("Preparing dummy input...")
dummy_text = ["This is a sample text for ONNX export."]
inputs = tokenizer(
    dummy_text,
    padding="max_length",  # Pad to max length for consistent shape during export
    truncation=True,
    max_length=MAX_SEQ_LENGTH,
    return_tensors="pt",
)
dummy_input_ids = inputs["input_ids"]
dummy_attention_mask = inputs["attention_mask"]
logging.info(f"Dummy input_ids shape: {dummy_input_ids.shape}")
logging.info(f"Dummy attention_mask shape: {dummy_attention_mask.shape}")

# --- Define Input/Output Names and Dynamic Axes ---
input_names = ["input_ids", "attention_mask"]
# Check model output format - often 'last_hidden_state' and 'pooler_output'
# For sentence-transformers, we often want the pooled output or mean of last_hidden_state
# Let's assume we'll take the pooler_output if available, or handle pooling later
output_names = [
    "last_hidden_state",
    "pooler_output",
]  # Adjust if model outputs differently

dynamic_axes = {
    "input_ids": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "last_hidden_state": {0: "batch_size", 1: "sequence_length"},
    "pooler_output": {0: "batch_size"},
}

# --- Export to ONNX ---
logging.info(f"Exporting model to ONNX: {onnx_path}")
try:
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),  # Pass inputs as a tuple
        onnx_path,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        opset_version=14,  # Use 14 or higher
        export_params=True,
        do_constant_folding=True,
    )
    logging.info("ONNX export successful!")
except Exception as e:
    logging.error(f"ONNX export failed: {e}", exc_info=True)
    exit(1)

# --- Verify ONNX Model (Optional) ---
logging.info("Verifying ONNX model (optional)...")
try:
    import onnx

    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    logging.info("ONNX model check successful.")

    # import onnxruntime as ort
    # logging.info("Running inference with ONNX Runtime...")
    # ort_session = ort.InferenceSession(onnx_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # ort_inputs = {
    #     "input_ids": dummy_input_ids.cpu().numpy(),
    #     "attention_mask": dummy_attention_mask.cpu().numpy()
    # }
    # ort_outputs = ort_session.run(output_names, ort_inputs)
    # logging.info(f"ONNX Runtime output shapes: {[o.shape for o in ort_outputs]}")

except Exception as e:
    logging.warning(f"ONNX verification failed: {e}", exc_info=True)

logging.info(f"Text model exported to {onnx_path}")
