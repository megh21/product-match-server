import os
from dotenv import load_dotenv
import logging
import numpy as np
from PIL import Image

load_dotenv()
logging.basicConfig(level=logging.INFO)

EMBEDDING_MODE = os.getenv("EMBEDDING_MODE", "cloud")
logging.info(f"Using {EMBEDDING_MODE.upper()} embedding mode")

# Initialize variables
_get_text_embedding = None
_get_vision_embedding = None
_get_combined_embedding = None
TEXT_DIM = VISION_DIM = COMBINED_DIM = 0

if EMBEDDING_MODE == "local":
    logging.info("Initializing LOCAL mode with Triton")
    try:
        from .embeddings_local import (
            get_text_embedding as _get_text_embedding,
            get_vision_embedding as _get_vision_embedding,
            get_combined_embedding as _get_combined_embedding,
        )

        TEXT_DIM = int(os.getenv("LOCAL_TEXT_EMBED_DIM", 384))
        VISION_DIM = int(os.getenv("LOCAL_VISION_EMBED_DIM", 512))
        COMBINED_DIM = TEXT_DIM + VISION_DIM

    except ImportError as e:
        logging.error(f"Failed to import local embedding module: {e}")
        raise RuntimeError("Local embedding module failed to load")

elif EMBEDDING_MODE == "cloud":
    logging.info("Initializing CLOUD mode with Cohere")
    try:
        from .embedding_v_zero import (
            get_text_embedding as _get_text_embedding_cloud,
            get_image_embedding as _get_vision_embedding_cloud,
        )

        def _get_vision_embedding(images: list[Image.Image]) -> np.ndarray:
            if not images:
                return np.array([])
            img = images[0]
            import io
            import base64

            buffer = io.BytesIO()
            img.save(buffer, format="JPEG")
            base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
            data_uri = f"data:image/jpeg;base64,{base64_image}"

            import cohere

            co = cohere.ClientV2(os.getenv("COHERE_API_KEY", ""))
            response = co.embed(
                model="embed-english-v3.0",
                input_type="image",
                embedding_types=["float"],
                images=[data_uri],
            )
            return np.array(response.embeddings.float)

        def _get_text_embedding(text: str) -> np.ndarray:
            return _get_text_embedding_cloud(text)

        def _get_combined_embedding(text: str, image: Image.Image) -> np.ndarray:
            # Get text embedding (1024-dim)
            text_emb = _get_text_embedding(text)
            if text_emb.ndim == 2:
                text_emb = text_emb[0]  # Take first vector if batch returned
            
            # Get image embedding (1024-dim)
            image_emb = _get_vision_embedding([image])
            if image_emb.ndim == 2:
                image_emb = image_emb[0]  # Take first vector if batch returned
            
            # Verify dimensions before concatenation
            if text_emb.shape[0] != VISION_DIM or image_emb.shape[0] != VISION_DIM:
                logging.error(f"Dimension mismatch - Text: {text_emb.shape}, Image: {image_emb.shape}")
                raise ValueError("Embedding dimensions don't match expected size")
            
            # Concatenate them to create combined embedding (2048-dim)
            result = np.concatenate([text_emb, image_emb])
            
            # Verify final dimension
            if result.shape[0] != COMBINED_DIM:
                logging.error(f"Combined embedding dimension mismatch. Expected {COMBINED_DIM}, got {result.shape[0]}")
                raise ValueError("Combined embedding dimension incorrect")
                
            return result

        TEXT_DIM = VISION_DIM = 1024
        COMBINED_DIM = TEXT_DIM + VISION_DIM

    except ImportError as e:
        logging.error(f"Failed to import cloud embedding module: {e}")
        raise RuntimeError("Cloud embedding module failed to load")

else:
    raise ValueError(f"Invalid EMBEDDING_MODE: {EMBEDDING_MODE}")

# Expose the functions and dimensions
get_text_embedding = _get_text_embedding
get_vision_embedding = _get_vision_embedding
get_combined_embedding = _get_combined_embedding

DIM_TEXT = TEXT_DIM
DIM_VISION = VISION_DIM
DIM_COMBINED = COMBINED_DIM

CURRENT_MODE = EMBEDDING_MODE
