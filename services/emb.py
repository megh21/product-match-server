import torch
import clip
from PIL import Image
from sentence_transformers import SentenceTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load Models
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)
text_model = SentenceTransformer("all-MiniLM-L6-v2")


def get_image_embedding(image_path: str):
    image = Image.open(image_path).convert("RGB")
    image_input = clip_preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        img_emb = clip_model.encode_image(image_input)
    return img_emb.squeeze().cpu().numpy()


def get_text_embedding(text: str):
    return text_model.encode(text)


def get_combined_embedding(text: str, image_path: str):
    text_emb = get_text_embedding(text)
    image_emb = get_image_embedding(image_path)
    return torch.tensor(list(text_emb) + list(image_emb)).numpy()
