python quantize.py \
    --model_dir OpenGVLab/InternVL-Chat-V1.5 \
    --output_dir /path/to/converted_internvl_checkpoint \
    --qformat full_prec `# Start with no quantization (FP16/BF16)` \
    # --qformat int8_sq --kv_cache_dtype int8 # If trying INT8 later
    --dtype float16 `# Or bfloat16` \
    --calib_dataset /path/to/your/calibration_data.jsonl `# Needed even for full_prec? Check script logic` \
    --calib_size 100 \
    --trust_remote_code \
    --tp_size 1 \
    --pp_size 1 \