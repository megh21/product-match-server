import os
import pandas as pd
from mongo import insert_sample_products
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
IMAGES_DIR = os.getenv("IMAGES_DIR")
STYLES_DIR = os.getenv("STYLES_DIR")
IMAGES_CSV = os.getenv("IMAGES_CSV")
STYLES_CSV = os.getenv("STYLES_CSV")


def load_image_links() -> Dict[str, str]:
    """Load image filename to link mapping from CSV"""
    df = pd.read_csv(IMAGES_CSV)
    return dict(zip(df.filename, df.link))


def load_style_metadata() -> pd.DataFrame:
    """Load style metadata from CSV"""
    return pd.read_csv(STYLES_CSV)


def create_product_documents() -> List[dict]:
    """Create product documents combining image and style data"""
    # Load mappings and metadata
    image_links = load_image_links()
    styles_df = load_style_metadata()

    products = []

    # Iterate through images directory
    for image_file in os.listdir(IMAGES_DIR):
        if not image_file.endswith(".jpg"):
            continue

        product_id = int(image_file.split(".")[0])

        # Get style metadata
        style = styles_df[styles_df.id == product_id].iloc[0]

        # Create product document
        product = {
            "product_id": product_id,
            "name": style.productDisplayName,
            "image_filename": image_file,
            "image_path": os.path.join(IMAGES_DIR, image_file),
            "image_url": image_links.get(image_file, ""),
            "metadata": {
                "gender": style.gender,
                "masterCategory": style.masterCategory,
                "subCategory": style.subCategory,
                "articleType": style.articleType,
                "baseColour": style.baseColour,
                "season": style.season,
                "year": style.year,
                "usage": style.usage,
            },
        }

        # Load additional style data if exists
        # style_json_path = os.path.join(STYLES_DIR, f"{product_id}.json")
        # if os.path.exists(style_json_path):
        #     with open(style_json_path) as f:
        #         style_data = json.load(f)
        #         product["style_data"] = style_data

        products.append(product)

    return products


def main():
    """Main function to run ingestion"""
    print("Starting data ingestion...")

    # Ensure required directories exist
    for dir_path in [IMAGES_DIR, STYLES_DIR]:
        if not os.path.exists(dir_path):
            print(f"Creating directory: {dir_path}")
            os.makedirs(dir_path)

    # Create and insert product documents
    try:
        products = create_product_documents()
        print(f"Created {len(products)} product documents")

        insert_sample_products(products)
        print("Ingestion completed successfully")

    except Exception as e:
        print(f"Error during ingestion: {str(e)}")


if __name__ == "__main__":
    main()
