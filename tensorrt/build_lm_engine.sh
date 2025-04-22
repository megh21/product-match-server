#!/bin/bash
# build_lm_engine.sh (v3 - using conversion script + checkpoint_dir)

set -e # Exit on error

# --- Configuration ---
HF_MODEL_ID="OpenGVLab/InternVL3-2B"
CONVERSION_SCRIPT_PATH="./tensorrt/convert_hf_to_trtllm_ckpt.py" # Path to the new Python script
CONVERTED_CHECKPOINT_DIR="./trt_engines/InternVL3-2B/lm_fp16_ckpt" # Intermediate checkpoint
OUTPUT_ENGINE_DIR="./trt_engines/InternVL3-2B/lm_fp16_engine"     # Final engine output

TP_SIZE=1
PP_SIZE=1
MAX_BATCH_SIZE=4
MAX_INPUT_LEN=1024
MAX_OUTPUT_LEN=150
BUILD_PRECISION="float16" # Match the --dtype used in the python script

# --- Step 1: Convert HF Model to TRT-LLM Checkpoint Format ---
echo "------------------------------------------------------------------"
echo "Step 1: Converting HF Model to TensorRT-LLM Checkpoint Format..."
echo "Source HF Model: ${HF_MODEL_ID}"
echo "Output Checkpoint Dir: ${CONVERTED_CHECKPOINT_DIR}"
echo "------------------------------------------------------------------"

# Create the checkpoint directory if it doesn't exist
mkdir -p "$CONVERTED_CHECKPOINT_DIR"

# Run the conversion script (ensure python points to your venv python if needed)
python "$CONVERSION_SCRIPT_PATH" \
    --hf_model_id "$HF_MODEL_ID" \
    --output_dir "$CONVERTED_CHECKPOINT_DIR" \
    --dtype "$BUILD_PRECISION" \
    --qformat full_prec \
    --tp_size "$TP_SIZE" \
    --pp_size "$PP_SIZE" \
    --trust_remote_code # Add this flag for InternVL

# Check if conversion script produced expected output (basic check)
# A config.json is usually created by quantize_and_export
if [ ! -f "${CONVERTED_CHECKPOINT_DIR}/config.json" ]; then
    echo "ERROR: Conversion script did not produce expected config.json in ${CONVERTED_CHECKPOINT_DIR}"
    echo "       Please check the output of the conversion script."
    exit 1
fi

echo "Conversion script finished."


# --- Step 2: Build TensorRT-LLM Engine from Checkpoint ---
echo "------------------------------------------------------------------"
echo "Step 2: Building TensorRT-LLM Engine from Checkpoint..."
echo "Checkpoint Dir: ${CONVERTED_CHECKPOINT_DIR}"
echo "Output Engine Dir: ${OUTPUT_ENGINE_DIR}"
echo "------------------------------------------------------------------"

# Create the final engine directory
mkdir -p "$OUTPUT_ENGINE_DIR"

# Run trtllm-build using the checkpoint directory
venv/bin/trtllm-build \
    --checkpoint_dir "$CONVERTED_CHECKPOINT_DIR" \
    --output_dir "$OUTPUT_ENGINE_DIR" \
    --gemm_plugin "$BUILD_PRECISION" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_input_len "$MAX_INPUT_LEN" \
    --max_output_len "$MAX_OUTPUT_LEN" \
    --max_beam_width 1 \
    --tp_size "$TP_SIZE" \
    --pp_size "$PP_SIZE" \
    --use_inflight_batching \
    --paged_kv_cache enable \
    --remove_input_padding enable \
    --use_gpt_attention_plugin "$BUILD_PRECISION" \
    # Removed --dtype (inferred from checkpoint)
    # Removed --trust_remote_code (handled by conversion script)

echo "-----------------------------------------------------"
echo "TensorRT-LLM Engine build process for LM finished."
echo "Engine files should be in: $OUTPUT_ENGINE_DIR"
echo "Log file: ${OUTPUT_ENGINE_DIR}/build.log"
echo "-----------------------------------------------------"