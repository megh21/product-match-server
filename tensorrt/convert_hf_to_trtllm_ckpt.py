import argparse
import os
import json
import tempfile
import logging

# Attempt to import the key function from the installed library
try:
    from tensorrt_llm.quantization import quantize_and_export
except ImportError:
    logging.error(
        "Could not import quantize_and_export from tensorrt_llm.quantization."
    )
    logging.error("Ensure tensorrt_llm is correctly installed.")
    exit(1)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def create_dummy_calibration_file(tmp_dir, num_samples=8):
    """Creates a dummy jsonl file for calibration placeholder."""
    calib_file_path = os.path.join(tmp_dir, "dummy_calibration.jsonl")
    dummy_samples = [
        {"text": "This is a dummy sentence for calibration purposes."},
        {"text": "TensorRT-LLM requires a dataset path."},
        {"text": "Generating a small temporary file."},
        {"text": "The quick brown fox jumps over the lazy dog."},
        {"text": "Lorem ipsum dolor sit amet."},
        {"text": "Placeholder data for model conversion."},
        {"text": "Testing the data loading pipeline."},
        {"text": "One more dummy line."},
    ]
    actual_samples = dummy_samples * (num_samples // len(dummy_samples) + 1)
    actual_samples = actual_samples[:num_samples]

    try:
        with open(calib_file_path, "w") as f:
            for sample in actual_samples:
                f.write(json.dumps(sample) + "\n")
        logging.info(
            f"Created dummy calibration file at: {calib_file_path} with {len(actual_samples)} samples."
        )
        return calib_file_path
    except Exception as e:
        logging.error(f"Failed to create dummy calibration file: {e}", exc_info=True)
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert HuggingFace model to TensorRT-LLM checkpoint format."
    )
    parser.add_argument(
        "--hf_model_id",
        required=True,
        help="HuggingFace model ID (e.g., 'OpenGVLab/InternVL-Chat-V1.5')",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory to save the TensorRT-LLM checkpoint",
    )
    parser.add_argument(
        "--dtype",
        default="float16",
        choices=["float16", "bfloat16"],
        help="Data type for weights (ensure GPU support for bfloat16)",
    )
    parser.add_argument(
        "--qformat",
        default="full_prec",
        const="full_prec",
        nargs="?",
        choices=["full_prec"],
        help="Only 'full_prec' is supported by this script for conversion.",
    )  # Keep this simple for now
    parser.add_argument("--tp_size", type=int, default=1, help="Tensor parallel size")
    parser.add_argument("--pp_size", type=int, default=1, help="Pipeline parallel size")
    parser.add_argument(
        "--calib_size_dummy",
        type=int,
        default=16,
        help="Number of dummy samples for placeholder calibration file",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device for loading HF model",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Allow loading models with custom code (Necessary for InternVL)",
    )
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--tokenizer_max_seq_length", type=int, default=2048)
    parser.add_argument(
        "--device_map", default="auto", choices=["auto", "sequential", "cpu", "gpu"]
    )
    parser.add_argument(
        "--kv_cache_dtype",
        default=None,
        choices=["int8", "fp8", None],
        help="KV Cache dtype (usually None for full_prec).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size (used internally, even without calibration).",
    )
    parser.add_argument(
        "--calib_max_seq_length",
        type=int,
        default=512,
        help="Max sequence length (used internally).",
    )
    parser.add_argument(
        "--awq_block_size",
        type=int,
        default=128,
        help="AWQ block size (placeholder, not used for full_prec).",
    )
    parser.add_argument(
        "--cp_size", type=int, default=1, help="Context Parallel size (usually 1)."
    )
    # ====> REMOVED quantize_lm_head from parser <====
    # parser.add_argument("--quantize_lm_head", action='store_true', default=False)
    parser.add_argument("--num_medusa_heads", type=int, default=0)
    parser.add_argument("--num_medusa_layers", type=int, default=0)
    parser.add_argument("--max_draft_len", type=int, default=0)
    parser.add_argument("--medusa_hidden_act", type=str, default="silu")
    parser.add_argument("--medusa_model_dir", type=str, default=None)
    parser.add_argument("--quant_medusa_head", action="store_true", default=False)
    parser.add_argument("--auto_quantize_bits", type=float, default=None)

    args = parser.parse_args()

    if not args.trust_remote_code:
        logging.warning(
            "The specified HF model might require '--trust_remote_code' to load correctly."
        )
        if "InternVL" in args.hf_model_id:
            args.trust_remote_code = True
            logging.info(
                "Enabled --trust_remote_code automatically for InternVL model."
            )

    with tempfile.TemporaryDirectory() as tmpdir:
        dummy_calib_path = create_dummy_calibration_file(tmpdir, args.calib_size_dummy)
        if not dummy_calib_path:
            logging.error("Failed to proceed without dummy calibration file.")
            return

        logging.info(f"Starting conversion for {args.hf_model_id}...")
        logging.info(f"Outputting TRT-LLM checkpoint to: {args.output_dir}")
        logging.info(
            f"Using dtype: {args.dtype}, TP: {args.tp_size}, PP: {args.pp_size}"
        )

        quantize_kwargs = {
            "model_dir": args.hf_model_id,
            "output_dir": args.output_dir,
            "dtype": args.dtype,
            "qformat": args.qformat,
            "calib_dataset": dummy_calib_path,
            "calib_size": args.calib_size_dummy,
            "tp_size": args.tp_size,
            "pp_size": args.pp_size,
            "seed": args.seed,
            "tokenizer_max_seq_length": args.tokenizer_max_seq_length,
            "device": args.device,
            "device_map": args.device_map,
            "kv_cache_dtype": args.kv_cache_dtype,
            "batch_size": args.batch_size,
            "calib_max_seq_length": args.calib_max_seq_length,
            "awq_block_size": args.awq_block_size,
            "cp_size": args.cp_size,
            # ====> REMOVED quantize_lm_head from kwargs <====
            # "quantize_lm_head": args.quantize_lm_head,
            "num_medusa_heads": args.num_medusa_heads,
            "num_medusa_layers": args.num_medusa_layers,
            "max_draft_len": args.max_draft_len,
            "medusa_hidden_act": args.medusa_hidden_act,
            "medusa_model_dir": args.medusa_model_dir,
            "quant_medusa_head": args.quant_medusa_head,
            "auto_quantize_bits": args.auto_quantize_bits,
        }
        if args.trust_remote_code:
            os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"] = "True"
            logging.info(
                "Set TRANSFORMERS_TRUST_REMOTE_CODE=True environment variable."
            )

        try:
            quantize_and_export(**quantize_kwargs)
            logging.info("Conversion process completed successfully.")
            logging.info(
                f"TensorRT-LLM checkpoint should be available in: {args.output_dir}"
            )

        except TypeError as te:
            if "trust_remote_code" in str(te):
                logging.error(
                    "The installed `quantize_and_export` function does not accept `trust_remote_code` directly."
                )
                logging.error("Attempted using TRANSFORMERS_TRUST_REMOTE_CODE env var.")
                logging.error(
                    "If loading still fails, manual patching of tensorrt_llm library might be needed."
                )
            else:
                logging.error(f"TypeError during conversion: {te}", exc_info=True)
        except Exception as e:
            logging.error(
                f"An error occurred during the conversion process: {e}", exc_info=True
            )
        finally:
            if (
                args.trust_remote_code
                and "TRANSFORMERS_TRUST_REMOTE_CODE" in os.environ
            ):
                del os.environ["TRANSFORMERS_TRUST_REMOTE_CODE"]


if __name__ == "__main__":
    main()
