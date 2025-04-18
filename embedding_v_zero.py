import cohere
import numpy as np
from PIL import Image
import base64
import io
import os
from dotenv import load_dotenv

load_dotenv()
# Initialize Cohere client - replace with your API key
co = cohere.ClientV2(os.getenv("COHERE_API_KEY", ""))


def image_to_base64(image_path: str) -> str:
    """Convert image to base64 string with proper data URI format"""
    with Image.open(image_path) as image:
        if image.mode != "RGB":
            image = image.convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG")
        base64_image = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"


def get_image_embedding(image_path: str):
    """Get embedding for an image using Cohere's multimodal model"""
    image_b64 = image_to_base64(image_path)
    response = co.embed(
         model="embed-english-v3.0",
    input_type="image",
    embedding_types=["float"],
    images=[image_b64],
)
    return np.array(response.embeddings.float)


def get_text_embedding(text: str):
    """Get embedding for text using Cohere's model"""
    response = co.embed(
        texts=[text],
        model="embed-english-v3.0",
        input_type="search_document",
        embedding_types=["float"],
    )
    return np.array(response.embeddings.float)


def get_combined_embedding(text: str, image_path: str):
    """Get combined embedding for both text and image"""
    text_emb = get_text_embedding(text)
    image_emb = get_image_embedding(image_path)
    return np.concatenate([text_emb, image_emb])


if __name__ == "__main__":
    text_in = "Sample text"
    image_path = "data/images/10000.jpg"
    text_emb = get_text_embedding(text_in)
    print("Text Embedding:", text_emb)
    image_emb = get_image_embedding(image_path)
    print("Image Embedding:", image_emb)
    combined_emb = get_combined_embedding(text_in, image_path)
    print("Combined Embedding:", combined_emb)
    print("Combined Embedding Shape:", combined_emb.shape)
    print("Text Embedding Shape:", text_emb.shape)
    print("Image Embedding Shape:", image_emb.shape)
