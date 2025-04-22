import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import logging
import onnx
import onnxruntime as ort
import numpy as np
import inspect
import gc

# --- Basic Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True
)

# --- Configuration ---
MODEL_ID = "OpenGVLab/InternVL3-2B"
OUTPUT_DIR = "models/internvl3_lm"  # Separate directory for LM ONNX
ONNX_FILENAME = "internvl3_language.onnx"
# Dummy input parameters for LM export
DUMMY_BATCH_SIZE = 1
DUMMY_SEQ_LENGTH = 256  # Representative sequence length
ONNX_OPSET = 14

# --- Ensure Output Directory Exists ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
ONNX_PATH = os.path.join(OUTPUT_DIR, ONNX_FILENAME)

# --- Load Full Model (to extract LM part) and Tokenizer ---
logging.info(f"Loading full model to extract LM: {MODEL_ID}")
try:
    # Load the CausalLM model which contains the language model head
    full_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
        device_map="cuda",  # Load to GPU to get parameters easily
    ).eval()
    logging.info("Full model loaded successfully.")

    # Extract the language model component
    # The exact attribute name might vary, common names are 'language_model', 'model', 'transformer'
    # Inspect the full_model object if needed (`print(full_model)`)
    if hasattr(full_model, "language_model"):
        language_model = full_model.language_model
        logging.info("Extracted 'language_model' attribute.")
    elif hasattr(full_model, "model") and not isinstance(
        full_model.model, torch.nn.ModuleList
    ):  # Avoid grabbing just layers list
        language_model = full_model.model  # Often the main transformer block
        logging.info("Extracted 'model' attribute.")
    elif hasattr(full_model, "transformer"):
        language_model = full_model.transformer
        logging.info("Extracted 'transformer' attribute.")
    else:
        logging.error(
            "Could not automatically identify the language model component within the loaded CausalLM model."
        )
        exit(1)

    # We might also need the final LM head if it's separate from the main block
    if hasattr(full_model, "lm_head") and not any(
        p is full_model.lm_head for p in language_model.parameters()
    ):
        # Combine the main block and the head if they are separate modules
        class LMWithHead(torch.nn.Module):
            def __init__(self, lang_model, lm_head):
                super().__init__()
                self.model = lang_model
                self.lm_head = lm_head

            def forward(self, inputs_embeds, attention_mask=None, **kwargs):
                # Pass through main model
                outputs = self.model(
                    inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs
                )

                # For CausalLMOutputWithPast, logits are directly available
                if hasattr(outputs, "logits"):
                    # Model already applied the LM head, return logits directly
                    return outputs.logits

                # For models that return hidden states without applying LM head
                elif hasattr(outputs, "last_hidden_state"):
                    hidden_states = outputs.last_hidden_state
                    # Apply LM head manually
                    logits = self.lm_head(hidden_states)
                    return logits

                # For tuple outputs (older models)
                elif isinstance(outputs, tuple):
                    hidden_states = outputs[0]
                    # Apply LM head manually
                    logits = self.lm_head(hidden_states)
                    return logits

                # Fallback - return whatever we got (debug case)
                else:
                    logging.warning(
                        f"Unexpected output type: {type(outputs)}, returning as is"
                    )
                    return outputs

        language_model = LMWithHead(language_model, full_model.lm_head)
        logging.info("Combined language model block with lm_head.")

    # Get hidden size for dummy input_embeds
    try:
        # Try different ways to get hidden size config
        if hasattr(full_model.config, "hidden_size"):
            hidden_size = full_model.config.hidden_size
        elif hasattr(full_model.config, "text_config") and hasattr(
            full_model.config.text_config, "hidden_size"
        ):
            hidden_size = full_model.config.text_config.hidden_size
        elif hasattr(language_model.config, "hidden_size"):
            hidden_size = language_model.config.hidden_size
        else:
            # Infer from embedding layer if possible
            if hasattr(language_model, "embed_tokens"):
                hidden_size = language_model.embed_tokens.embedding_dim
            else:
                raise ValueError("Cannot determine hidden_size")
        logging.info(f"Determined hidden size: {hidden_size}")
    except Exception as e:
        logging.error(f"Failed to determine hidden size: {e}", exc_info=True)
        exit(1)

    # Load tokenizer separately (not needed for export itself, but for verification)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
    logging.info("Tokenizer loaded.")


except Exception as e:
    logging.error(f"Failed to load model or components: {e}", exc_info=True)
    exit(1)


# --- Prepare Dummy Input for LM Export ---
logging.info("Preparing dummy input for Language Model export...")

# 1. Dummy input_embeds (simulating combined text+vision embeddings)
# Needs shape: (batch_size, sequence_length, hidden_size)
dummy_input_embeds = torch.randn(
    DUMMY_BATCH_SIZE,
    DUMMY_SEQ_LENGTH,
    hidden_size,
    dtype=torch.float16,  # Match model dtype
    device=full_model.device,  # Put on same device as model
)
logging.info(f"Created dummy_input_embeds with shape: {dummy_input_embeds.shape}")

# 2. Dummy attention_mask
# Needs shape: (batch_size, sequence_length)
dummy_attention_mask = torch.ones(
    DUMMY_BATCH_SIZE,
    DUMMY_SEQ_LENGTH,
    dtype=torch.int64,  # Usually int64 or long
    device=full_model.device,
)
logging.info(f"Created dummy_attention_mask with shape: {dummy_attention_mask.shape}")


# 3. Define input names and tuple for export
# The LM component likely expects 'inputs_embeds' and 'attention_mask'
# Check language_model.forward signature if unsure
input_names = ["inputs_embeds", "attention_mask"]
dummy_input_tuple = (dummy_input_embeds, dummy_attention_mask)
logging.info(f"Input names for LM export: {input_names}")

# --- Define Output Names and Dynamic Axes ---
output_names = ["logits"]  # LM head outputs logits

dynamic_axes = {
    "inputs_embeds": {0: "batch_size", 1: "sequence_length"},
    "attention_mask": {0: "batch_size", 1: "sequence_length"},
    "logits": {0: "batch_size", 1: "sequence_length"},
}

# --- Test LM Forward Pass (Optional but Recommended) ---
logging.info("Performing test forward pass on isolated LM...")
try:
    with torch.no_grad():
        # We might need to pass kwargs if the combined LMWithHead expects them
        # Use inspect on the final `language_model` object if needed
        # test_output = language_model(inputs_embeds=dummy_input_embeds, attention_mask=dummy_attention_mask)
        test_output = language_model(*dummy_input_tuple)  # Try with tuple first

        # Check if output is logits directly or needs extraction
        if isinstance(test_output, tuple):
            test_logits = test_output[0]
        elif hasattr(test_output, "logits"):
            test_logits = test_output.logits
        else:  # Assume direct output if LMWithHead is used
            test_logits = test_output

        logging.info(
            f"Test LM forward pass successful. Logits shape: {test_logits.shape}"
        )

except Exception as e:
    logging.error(f"Test LM forward pass failed: {e}", exc_info=True)
    # Try inspecting the forward signature of the final `language_model` object
    try:
        logging.info(
            f"Final LM forward signature: {inspect.signature(language_model.forward)}"
        )
    except Exception as e:
        logging.error(
            f"Failed to inspect final LM forward signature: {e}", exc_info=True
        )
    exit(1)

# --- Export LM to ONNX ---
logging.info(f"Exporting Language Model to ONNX: {ONNX_PATH}")
torch.cuda.empty_cache()
gc.collect()

try:
    with torch.no_grad():
        torch.onnx.export(
            language_model,
            args=dummy_input_tuple,
            f=ONNX_PATH,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=ONNX_OPSET,
            export_params=True,
            do_constant_folding=True,
            # ===> ADD THIS ARGUMENT <===
            all_tensors_to_one_file=False,  # Try forcing separate files
            # verbose=True # Keep verbose off unless needed again
        )
    logging.info("LM ONNX export successful.")

except Exception as e:
    logging.error(f"LM ONNX export failed: {e}", exc_info=True)
    exit(1)

# --- Verify LM ONNX Model (Optional but Recommended) ---
logging.info("Verifying LM ONNX model...")
try:
    # Check the path first for large models
    onnx.checker.check_model(ONNX_PATH)
    logging.info("LM ONNX model basic check successful.")

    logging.info("Running LM inference with ONNX Runtime...")
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    # ... (provider check) ...
    ort_session = ort.InferenceSession(ONNX_PATH, providers=providers)

    ort_input_names = [inp.name for inp in ort_session.get_inputs()]
    ort_output_names = [out.name for out in ort_session.get_outputs()]
    logging.info(f"LM ORT session expects inputs: {ort_input_names}")
    logging.info(f"LM ORT session expects outputs: {ort_output_names}")

    if set(ort_input_names) != set(input_names):
        raise ValueError(
            f"Mismatch between exported LM inputs {input_names} and ORT inputs {ort_input_names}"
        )

    # Prepare ORT inputs (use CPU numpy arrays)
    ort_inputs = {
        "inputs_embeds": dummy_input_embeds.cpu().numpy(),
        "attention_mask": dummy_attention_mask.cpu().numpy(),
    }

    ort_outputs_list = ort_session.run(ort_output_names, ort_inputs)
    ort_logits = ort_outputs_list[
        ort_output_names.index("logits")
    ]  # Get logits by name
    logging.info(f"LM ONNX Runtime output 'logits' shape: {ort_logits.shape}")

    # Compare with PyTorch output (use the same inputs)
    with torch.no_grad():
        torch_output = language_model(*dummy_input_tuple)
        # Extract logits consistently
        if isinstance(torch_output, tuple):
            torch_logits = torch_output[0]
        elif hasattr(torch_output, "logits"):
            torch_logits = torch_output.logits
        else:
            torch_logits = torch_output
    torch_logits_np = torch_logits.cpu().numpy()
    logging.info(f"PyTorch LM output 'logits' shape: {torch_logits_np.shape}")

    # Compare shapes and statistics
    if torch_logits_np.shape == ort_logits.shape:
        logging.info("PyTorch LM and ONNX output shapes match.")
        try:
            np.testing.assert_allclose(
                torch_logits_np, ort_logits.astype(np.float32), rtol=5e-2, atol=5e-2
            )
            logging.info("PyTorch LM and ONNX Runtime outputs match within tolerance.")
        except AssertionError as e:
            logging.warning(f"LM Outputs differ significantly: {e}")
            # ... (print stats if needed) ...
    else:
        logging.warning(
            f"LM Output shape mismatch: PyTorch {torch_logits_np.shape} vs ONNX {ort_logits.shape}"
        )


except ImportError:
    logging.warning("ONNX or ONNX Runtime not installed. Skipping verification.")
except Exception as e:
    logging.warning(f"LM ONNX verification failed: {e}", exc_info=True)

# --- Cleanup ---
logging.info("Cleaning up GPU memory...")
del full_model, language_model, tokenizer
del dummy_input_embeds, dummy_attention_mask, dummy_input_tuple
# del test_output, test_logits # Delete intermediate vars if they exist
torch.cuda.empty_cache()
gc.collect()

logging.info(f"LM ONNX export process completed. Model saved to {ONNX_PATH}")
