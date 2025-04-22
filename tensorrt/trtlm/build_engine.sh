#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
MODEL_ID="OpenGVLab/InternVL3-2B"
CALIBRATION_DATA_DIR="data/images" # Your image directory
OUTPUT_ENGINE_DIR="./trt_engines/internvl-chat-1.5/int8_calib"
CALIBRATION_CACHE_FILE="internvl_calibration.cache" # Must match output of calibrate script
CALIBRATION_SCRIPT="generate_internvl_calibration_cache.py"

# Model loading precision for calibration script (match if possible)
MODEL_PRECISION="bf16" # Use 'bf16' or 'fp16'

# GPU settings (likely fine for a 2B model on single GPU)
WORLD_SIZE=1
TP_SIZE=1 # Tensor Parallelism
PP_SIZE=1 # Pipeline Parallelism

# Inference constraints (adjust as needed)
MAX_BATCH_SIZE=4 # Can potentially increase for smaller models
MAX_INPUT_LEN=1024 # Max tokens (prompt + image features placeholder) - Check model's context length
MAX_OUTPUT_LEN=150 # Max generated caption length

# Quantization settings
QUANT_MODE="int8_sq"      # SmoothQuant INT8 for activations/weights
KV_CACHE_QUANT="int8"     # INT8 for KV cache

# Base precision for engine build & plugins (match MODEL_PRECISION if possible)
BUILD_PRECISION="bfloat16" # 'bfloat16' or 'float16'

# --- Check if BF16 is viable for build ---
# Simple check using python, assumes torch is installed
SUPPORTS_BF16=$(python -c "import torch; print(torch.cuda.is_bf16_supported())")

if [ "$BUILD_PRECISION" = "bfloat16" ] && [ "$SUPPORTS_BF16" = "False" ]; then
  echo "Warning: Requested build precision BF16 not supported by GPU, falling back to FP16."
  BUILD_PRECISION="float16"
  MODEL_PRECISION="fp16" # Also use fp16 for calibration loading
fi
echo "Using build precision: $BUILD_PRECISION"


# --- Step 1: Generate Calibration Cache ---
echo "-----------------------------------------------------"
echo "Step 1: Generating INT8 Calibration Cache for ${MODEL_ID}..."
echo "-----------------------------------------------------"
python "$CALIBRATION_SCRIPT" \
    --model_id "$MODEL_ID" \
    --calibration_data_dir "$CALIBRATION_DATA_DIR" \
    --output_cache_file "$CALIBRATION_CACHE_FILE" \
    --calib_batch_size 1 \
    --precision "$MODEL_PRECISION" # Pass chosen precision

# Check if calibration cache was created
if [ ! -f "$CALIBRATION_CACHE_FILE" ]; then
    echo "ERROR: Calibration cache file '$CALIBRATION_CACHE_FILE' not found after running calibration script."
    echo "       Build process cannot continue without a cache file (even a dummy one)."
    exit 1
fi
# Check if it's empty (potential sign of dummy file or failed calibration)
if [ ! -s "$CALIBRATION_CACHE_FILE" ]; then
     echo "WARNING: Calibration cache file '$CALIBRATION_CACHE_FILE' is empty."
     echo "         This likely means calibration failed or used the dummy function."
     echo "         The resulting INT8 engine may have very poor accuracy."
fi
echo "Calibration cache generation script finished."


# --- Step 2: Build TensorRT-LLM Engine ---
echo "-----------------------------------------------------"
echo "Step 2: Building TensorRT Engine (INT8 Quantized)..."
echo "-----------------------------------------------------"

# Ensure output directory exists
mkdir -p "$OUTPUT_ENGINE_DIR"

# Construct the build command
# Flags for INT8 SmoothQuant + INT8 KV cache:
# --quant_mode=int8_sq
# --int8_kv_cache
# --quant_ckpt_path=<path_to_calibration.cache>

echo "Running trtllm build..."
trtllm build \
    --model_dir "$MODEL_ID" \
    --output_dir "$OUTPUT_ENGINE_DIR" \
    --dtype "$BUILD_PRECISION" \
    --quant_mode "$QUANT_MODE" \
    --int8_kv_cache \
    --quant_ckpt_path "$CALIBRATION_CACHE_FILE" \
    --gemm_plugin "$BUILD_PRECISION" \
    --max_batch_size "$MAX_BATCH_SIZE" \
    --max_input_len "$MAX_INPUT_LEN" \
    --max_output_len "$MAX_OUTPUT_LEN" \
    --max_beam_width 1 \
    --tp_size "$TP_SIZE" \
    --pp_size "$PP_SIZE" \
    --use_inflight_batching \
    --use_gpt_attention_plugin "$BUILD_PRECISION" \
    --remove_input_padding \
    --paged_kv_cache \
    --trust_remote_code \
    --workers 1 # Adjust based on CPU cores if needed

    # VLM Specific Flags (Check TensorRT-LLM docs/examples for InternVL specifically):
    # The standard build might handle it if the model structure is recognized
    # via transformers integration and trust_remote_code.
    # If errors occur related to vision components, you might need flags like:
    # --with_multi_modal (or similar)
    # --vision_model_type (e.g., internvit)
    # ...etc., IF your version of TensorRT-LLM supports them for this model.

echo "-----------------------------------------------------"
echo "TensorRT Engine build process finished."
echo "Engine files should be in: $OUTPUT_ENGINE_DIR"
echo "Log file: ${OUTPUT_ENGINE_DIR}/build.log" # TensorRT-LLM usually creates a log
echo "-----------------------------------------------------"

# Optional: Clean up calibration cache
# echo "Removing calibration cache file: $CALIBRATION_CACHE_FILE"
# rm "$CALIBRATION_CACHE_FILE"