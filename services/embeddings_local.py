import tritonclient.http as httpclient
from tritonclient.utils import np_to_triton_dtype
import numpy as np
from PIL import Image
import os
from dotenv import load_dotenv
import logging
from transformers import AutoTokenizer, CLIPProcessor  # For preprocessing

logging.basicConfig(level=logging.INFO)
load_dotenv()

# --- Configuration ---
TRITON_URL = os.getenv("TRITON_URL", "localhost:8000")
# Text model details (for tokenizer)
TEXT_MODEL_ID = os.getenv("TEXT_MODEL_ID", "sentence-transformers/all-MiniLM-L6-v2")
TEXT_MODEL_TRITON_NAME = os.getenv("TRITON_TEXT_MODEL_NAME", "text_encoder")
MAX_SEQ_LENGTH = 128  # Should match tokenizer/export config

# Vision model details (for processor) - USE THE ONE YOU DEPLOYED
# If using CLIP:
VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "openai/clip-vit-base-patch32")
VISION_MODEL_TRITON_NAME = os.getenv("TRITON_VISION_MODEL_NAME", "vision_encoder")
# If using Qwen Vision (update .env TRITON_VISION_MODEL_NAME and VISION_MODEL_ID if switching)
# VISION_MODEL_ID = os.getenv("VLM_MODEL_ID", "Qwen/Qwen-VL-Chat") # Need Qwen Tokenizer/Processor if using this
# VISION_MODEL_TRITON_NAME = os.getenv("TRITON_VISION_MODEL_NAME", "qwen_vision_encoder") # Match deployed name

# Expected output names from Triton config.pbtxt
TEXT_OUTPUT_NAME = "pooler_output"  # Or "last_hidden_state" - check your model/config
VISION_OUTPUT_NAME = "image_embeds"  # Or "vision_outputs" for Qwen - check config

# Add lazy loading of processors
_text_tokenizer = None
_vision_processor = None


def get_processors():
    global _text_tokenizer, _vision_processor
    try:
        from transformers import CLIPProcessor, AutoTokenizer

        VISION_MODEL_ID = os.getenv("VLM_MODEL")
        TEXT_MODEL_ID = os.getenv("TEXT_MODEL")

        _text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        _vision_processor = CLIPProcessor.from_pretrained(VISION_MODEL_ID)

    except Exception as e:
        logging.error(f"Failed to load tokenizers/processors: {e}")
        raise
    return _text_tokenizer, _vision_processor


# lazy load preprocessors
def load_processors():
    """Lazy load the tokenizer and processor for text and vision models."""
    try:
        text_tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL_ID)
        # Load the correct processor based on the vision model being used
        if "clip" in VISION_MODEL_ID.lower():
            vision_processor = CLIPProcessor.from_pretrained(VISION_MODEL_ID)
            logging.info(f"Loaded CLIPProcessor for {VISION_MODEL_ID}")
        elif "qwen" in VISION_MODEL_ID.lower():
            # Qwen might need its own tokenizer for image processing, or a specific processor
            # Re-use the VLM tokenizer which often handles image inputs
            vision_processor = AutoTokenizer.from_pretrained(
                VISION_MODEL_ID, trust_remote_code=True
            )
            logging.info(f"Loaded Qwen Tokenizer/Processor for {VISION_MODEL_ID}")

        else:
            raise ValueError(
                f"Unsupported vision model type for processor: {VISION_MODEL_ID}"
            )

        return text_tokenizer, vision_processor

    except Exception as e:
        logging.error(f"Failed to load tokenizers/processors: {e}", exc_info=True)
        raise


# Define a helper for Qwen image processing if needed
def qwen_process_images(processor, images):
    # formatted_input = [{"image": img} for img in images]  # Assuming PIL list
    logging.warning(
        "Qwen image preprocessing requires careful implementation matching its internal logic."
    )
    # Simplified placeholder: use CLIP-style processing for now
    # Replace with actual Qwen processing logic.
    logging.warning("Using CLIP-style processing as placeholder for Qwen.")
    inputs = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")(
        images=images, return_tensors="pt"
    )
    return inputs["pixel_values"]


# --- Triton Client Wrapper ---
class TritonClient:
    def __init__(self, url: str):
        self.url = url
        self.text_tokenizer, self.vision_processor = load_processors()
        self.client = None
        self._initialize_client()
        try:
            self.client = httpclient.InferenceServerClient(url=self.url, verbose=False)
            if not self.client.is_server_live():
                raise ConnectionError(f"Triton server at {url} is not live.")
            logging.info(f"Triton client connected to {url}")
        except Exception as e:
            logging.error(f"Failed to initialize Triton client: {e}", exc_info=True)
            raise

    def _prepare_text_input(self, texts: list[str]):
        inputs = self.text_tokenizer(
            texts,
            padding="max_length",  # Use max_length padding
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np",
        )
        input_ids = inputs["input_ids"].astype(np.int64)
        attention_mask = inputs["attention_mask"].astype(np.int64)

        # Create Triton input objects
        input_tensors = [
            httpclient.InferInput(
                "input_ids", input_ids.shape, np_to_triton_dtype(input_ids.dtype)
            ),
            httpclient.InferInput(
                "attention_mask",
                attention_mask.shape,
                np_to_triton_dtype(attention_mask.dtype),
            ),
        ]
        input_tensors[0].set_data_from_numpy(input_ids)
        input_tensors[1].set_data_from_numpy(attention_mask)
        return input_tensors

    def _prepare_vision_input(self, images: list[Image.Image]):
        """Prepares image batch for Triton vision model"""
        if not images:
            return []

        # Use the loaded processor/tokenizer
        if "clip" in VISION_MODEL_ID.lower():
            inputs = self.vision_processor(
                images=images, return_tensors="pt", padding=True
            )
            pixel_values = (
                inputs["pixel_values"].cpu().numpy().astype(np.float32)
            )  # Match TRT engine input type
        elif "qwen" in VISION_MODEL_ID.lower():
            # Use the Qwen processing logic defined earlier
            # This part needs refinement!
            vision_processor = self.vision_processor
            pixel_values_tensor = qwen_process_images(vision_processor, images)
            pixel_values = (
                pixel_values_tensor.cpu().numpy().astype(np.float16)
            )  # Match TRT engine input type (often FP16)
        else:
            raise ValueError(f"Preprocessing not defined for {VISION_MODEL_ID}")

        # Create Triton input object
        input_tensor = httpclient.InferInput(
            "pixel_values",  # Must match name in Triton config.pbtxt
            pixel_values.shape,
            np_to_triton_dtype(pixel_values.dtype),
        )
        input_tensor.set_data_from_numpy(pixel_values)
        return [input_tensor]

    def get_text_embedding(self, texts: list[str]) -> np.ndarray:
        """Gets text embeddings from Triton"""
        if not texts:
            return np.array([])
        try:
            input_tensors = self._prepare_text_input(texts)
            # Define expected output
            outputs = [httpclient.InferRequestedOutput(TEXT_OUTPUT_NAME)]

            response = self.client.infer(
                model_name=TEXT_MODEL_TRITON_NAME, inputs=input_tensors, outputs=outputs
            )
            embedding = response.as_numpy(TEXT_OUTPUT_NAME)
            # if TEXT_OUTPUT_NAME == "last_hidden_state":
            #     # Mean pooling - requires attention mask
            #     pass
            return embedding

        except Exception as e:
            logging.error(f"Triton text inference failed: {e}", exc_info=True)
            # Return empty or raise? Returning empty for resilience.
            return np.array([])

    def get_vision_embedding(self, images: list[Image.Image]) -> np.ndarray:
        """Gets vision embeddings from Triton"""
        if not images:
            return np.array([])
        try:
            input_tensors = self._prepare_vision_input(images)
            outputs = [httpclient.InferRequestedOutput(VISION_OUTPUT_NAME)]

            response = self.client.infer(
                model_name=VISION_MODEL_TRITON_NAME,
                inputs=input_tensors,
                outputs=outputs,
            )
            embedding = response.as_numpy(VISION_OUTPUT_NAME)
            return embedding

        except Exception as e:
            logging.error(f"Triton vision inference failed: {e}", exc_info=True)
            return np.array([])


# --- Factory/Instance ---
# Create a single client instance for the module to reuse but laod it lazily
def get_triton_client():
    """Lazy load the Triton client instance"""
    global triton_client_instance
    if triton_client_instance is None:
        try:
            triton_client_instance = TritonClient(url=TRITON_URL)
        except Exception as e:
            logging.error(f"Failed to create Triton client: {e}")
            raise
    return triton_client_instance


# --- Public Functions ---
def get_text_embedding(texts: list[str]) -> np.ndarray:
    global triton_client_instance
    if triton_client_instance is None:  # Retry connection if failed initially
        try:
            triton_client_instance = TritonClient(url=TRITON_URL)
        except Exception as e:
            logging.error(f"Failed to connect to Triton: {e}")
            return np.array([])  # Return empty on failure

    return triton_client_instance.get_text_embedding(texts)


def get_vision_embedding(images: list[Image.Image]) -> np.ndarray:
    global triton_client_instance
    if triton_client_instance is None:  # Retry connection
        try:
            triton_client_instance = TritonClient(url=TRITON_URL)
        except Exception as e:
            logging.error(f"Failed to connect to Triton: {e}")
            return np.array([])

    return triton_client_instance.get_vision_embedding(images)


def get_combined_embedding(text: str, image: Image.Image) -> np.ndarray:
    """Generates combined text and vision embedding using Triton"""
    # Get embeddings in parallel? (Requires async client or threading)
    # For simplicity, do sequentially
    text_emb = get_text_embedding([text])
    image_emb = get_vision_embedding([image])

    if text_emb.size == 0 or image_emb.size == 0:
        logging.error("Failed to get one or both embeddings for combination.")
        return np.array([])  # Return empty if failure

    # Ensure 2D array for concatenation (batch size 1)
    if text_emb.ndim == 1:
        text_emb = text_emb.reshape(1, -1)
    if image_emb.ndim == 1:
        image_emb = image_emb.reshape(1, -1)

    # Concatenate along the feature dimension (axis=1)
    combined = np.concatenate([text_emb, image_emb], axis=1)
    return combined  # Shape (1, text_dim + vision_dim)


if __name__ == "__main__":
    # Example Usage (requires Triton server running with models)
    logging.info("Testing local Triton embeddings...")

    # Test Text Embedding
    sample_texts = ["A red dress for summer.", "Blue sneakers."]
    try:
        text_embeds = get_text_embedding(sample_texts)
        if text_embeds.size > 0:
            logging.info(
                f"Text embeddings shape: {text_embeds.shape}"
            )  # Expected: (2, 384)
        else:
            logging.warning("Text embedding test failed.")
    except Exception as e:
        logging.error(f"Text embedding test error: {e}")

    # Test Vision Embedding
    try:
        # Use a real image path accessible here
        img_path = "data/images/10000.jpg"
        if os.path.exists(img_path):
            sample_image = Image.open(img_path)
            image_embeds = get_vision_embedding(
                [sample_image, sample_image]
            )  # Test batching
            if image_embeds.size > 0:
                logging.info(
                    f"Vision embeddings shape: {image_embeds.shape}"
                )  # Expected: (2, 512) or (2, 4096) etc.
            else:
                logging.warning("Vision embedding test failed.")
        else:
            logging.warning(
                f"Sample image not found at {img_path}, skipping vision test."
            )

    except Exception as e:
        logging.error(f"Vision embedding test error: {e}")

    # Test Combined Embedding
    try:
        img_path = "data/images/10000.jpg"
        if os.path.exists(img_path):
            sample_image = Image.open(img_path)
            combined_emb = get_combined_embedding("Stylish jacket", sample_image)
            if combined_emb.size > 0:
                logging.info(
                    f"Combined embedding shape: {combined_emb.shape}"
                )  # E.g., (1, 384 + 512)
            else:
                logging.warning("Combined embedding test failed.")
        else:
            logging.warning(
                f"Sample image not found at {img_path}, skipping combined test."
            )
    except Exception as e:
        logging.error(f"Combined embedding test error: {e}")
