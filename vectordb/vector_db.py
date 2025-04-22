import os
import faiss
import numpy as np
from dotenv import load_dotenv
from services.embeddings import (
    get_combined_embedding,
    DIM_COMBINED,  # Use dimension from embeddings module
    CURRENT_MODE,  # Get current mode
)
from ingest import create_product_documents
import pickle
import time
from typing import Dict, List, Tuple
from PIL import Image
import logging  # Use logging

load_dotenv()
logging.basicConfig(level=logging.INFO)

# Paths
DATA_DIR = "data"
IMAGE_DIR = os.path.join(
    DATA_DIR, "images"
)  # Make sure this aligns with ingest.py IMAGES_DIR
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
MAPPING_PATH = os.path.join(DATA_DIR, "id_mapping.pkl")
PARTIAL_INDEX_PATH = os.path.join(DATA_DIR, "partial_index.pkl")

# FAISS Setup - Use dimension from embeddings module
DIM = DIM_COMBINED
index = None
id_mapping = {}

# Constants for cloud mode rate limiting (only if needed)
CLOUD_BATCH_SIZE = 5
CLOUD_RATE_LIMIT_DELAY = 60
CLOUD_MAX_RETRIES = 3


def ensure_data_dir():
    os.makedirs(DATA_DIR, exist_ok=True)


def save_index():
    ensure_data_dir()
    if index is None or index.ntotal == 0:
        logging.warning("Index is None or empty, nothing to save.")
        return
    try:
        faiss.write_index(index, INDEX_PATH)
        with open(MAPPING_PATH, "wb") as f:
            pickle.dump(id_mapping, f)
        logging.info(f"Saved FAISS index ({index.ntotal} vectors) and mapping.")
    except Exception as e:
        logging.error(f"Error saving index: {e}", exc_info=True)


def load_index():
    global index, id_mapping
    if index is None:
        # Initialize index if it doesn't exist, ensure DIM is valid
        if DIM <= 0:
            logging.error(
                f"Invalid embedding dimension: {DIM}. Cannot initialize FAISS."
            )
            return False
        logging.info(f"Initializing FAISS IndexFlatL2 with dimension {DIM}")
        index = faiss.IndexFlatL2(DIM)  # Use L2 distance

    if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
        try:
            logging.info(f"Loading FAISS index from {INDEX_PATH}")
            index = faiss.read_index(INDEX_PATH)
            # Check loaded index dimension
            if index.d != DIM:
                logging.error(
                    f"Loaded index dimension ({index.d}) != expected dimension ({DIM}). Rebuilding required."
                )
                # Reset index and mapping to force rebuild
                index = faiss.IndexFlatL2(DIM)
                id_mapping = {}
                # Clean up potentially corrupted files
                if os.path.exists(INDEX_PATH):
                    os.remove(INDEX_PATH)
                if os.path.exists(MAPPING_PATH):
                    os.remove(MAPPING_PATH)
                if os.path.exists(PARTIAL_INDEX_PATH):
                    os.remove(PARTIAL_INDEX_PATH)
                return False  # Indicate failure / need rebuild

            with open(MAPPING_PATH, "rb") as f:
                id_mapping = pickle.load(f)
            logging.info(f"Loaded existing FAISS index with {index.ntotal} vectors.")
            return True
        except Exception as e:
            logging.error(
                f"Error loading index files: {e}. Will attempt rebuild.", exc_info=True
            )
            # Reset index and mapping on load failure
            index = faiss.IndexFlatL2(DIM)
            id_mapping = {}
            return False
    else:
        logging.info("Index files not found.")
        return False


def process_batch(
    products: List[dict], start_idx: int, MAX_RETRIES=3
) -> Tuple[List[np.ndarray], Dict[int, int]]:
    """Process a batch of products with rate limiting"""
    vectors = []
    batch_mapping = {}

    for i, prod in enumerate(products):
        idx = start_idx + i
        retries = 0
        while retries < MAX_RETRIES:
            try:
                image_path = prod["image_path"]
                emb = get_combined_embedding(prod["name"], image_path)
                vectors.append(emb)
                batch_mapping[idx] = prod["product_id"]
                break
            except Exception as e:
                retries += 1
                if retries == MAX_RETRIES:
                    print(f"error: {e}")
                    print(
                        f"Failed to process product {prod['product_id']} after {MAX_RETRIES} retries"
                    )
                else:
                    wait_time = (2**retries) * 5  # Exponential backoff
                    print(
                        f"Retrying product {prod['product_id']} in {wait_time} seconds..."
                    )
                    time.sleep(wait_time)

    return vectors, batch_mapping


def save_partial_progress(vectors: List[np.ndarray], current_mapping: Dict[int, int]):
    ensure_data_dir()
    data = {
        "vectors": np.array(vectors) if vectors else np.array([]),
        "mapping": current_mapping,
    }
    try:
        with open(PARTIAL_INDEX_PATH, "wb") as f:
            pickle.dump(data, f)
        logging.info(f"Saved partial progress: {len(vectors)} vectors")
    except Exception as e:
        logging.error(f"Error saving partial progress: {e}")


def load_partial_progress() -> Tuple[List[np.ndarray], Dict[int, int]]:
    if not os.path.exists(PARTIAL_INDEX_PATH):
        return [], {}
    try:
        with open(PARTIAL_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        vectors = data["vectors"].tolist() if len(data["vectors"]) > 0 else []
        mapping = data["mapping"]
        logging.info(f"Loaded partial progress: {len(vectors)} vectors")
        return vectors, mapping
    except Exception as e:
        logging.error(f"Error loading partial progress: {e}. Starting fresh.")
        return [], {}


def build_faiss_index(force_rebuild=False):
    global index, id_mapping

    if DIM <= 0:
        logging.error(
            f"Cannot build index with invalid dimension: {DIM}. Check .env and embeddings module."
        )
        return

    if not force_rebuild and load_index():
        logging.info("Using existing FAISS index.")
        return

    logging.info("Building new FAISS index...")
    # Ensure index is initialized (load_index attempts this, but double check)
    if index is None or index.d != DIM:
        index = faiss.IndexFlatL2(DIM)
        id_mapping = {}
        logging.info(f"Re-initialized FAISS index with dimension {DIM}")

    # Load partial progress or start fresh
    vectors, id_mapping = load_partial_progress() if not force_rebuild else ([], {})
    start_faiss_idx = len(vectors)  # FAISS index position
    processed_product_ids = set(id_mapping.values())  # Track which product_ids are done

    logging.info(f"Resuming from FAISS index {start_faiss_idx}...")

    products = create_product_documents()
    # Filter out products already processed based on product_id
    products_to_process = [
        p for p in products if p["product_id"] not in processed_product_ids
    ]

    logging.info(f"Total products in source: {len(products)}")
    logging.info(f"Products already indexed: {len(processed_product_ids)}")
    logging.info(f"Products remaining to process: {len(products_to_process)}")

    # Determine batch size based on mode
    batch_size = (
        16 if CURRENT_MODE == "local" else CLOUD_BATCH_SIZE
    )  # Larger batch for local
    total_batches = (len(products_to_process) + batch_size - 1) // batch_size

    for i in range(0, len(products_to_process), batch_size):
        batch_products = products_to_process[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        logging.info(
            f"Processing batch {batch_num}/{total_batches} (Size: {len(batch_products)})"
        )

        batch_vectors = []
        batch_mapping_update = {}
        current_faiss_idx = start_faiss_idx + len(
            vectors
        )  # Keep track of current FAISS index

        for prod in batch_products:
            product_id = prod["product_id"]
            image_path = prod.get("image_path")
            text = prod.get("name", "")  # Use product name

            if not image_path or not os.path.exists(image_path):
                logging.warning(
                    f"Image path missing or invalid for product {product_id}: {image_path}. Skipping."
                )
                continue
            if not text:
                logging.warning(
                    f"Text (name) missing for product {product_id}. Using empty text."
                )

            retries = 0
            success = False
            while retries < (
                CLOUD_MAX_RETRIES if CURRENT_MODE == "cloud" else 1
            ):  # Only retry for cloud
                try:
                    with Image.open(image_path) as img:
                        img = img.convert("RGB")  # Ensure RGB
                        # Get combined embedding using the dispatcher
                        emb = get_combined_embedding(text, img)

                        if emb is not None and emb.size > 0:
                            # Ensure correct shape (1, DIM)
                            if emb.ndim == 1:
                                emb = emb.reshape(1, -1)

                            # Verify dimension
                            if emb.shape[1] != DIM:
                                logging.error(
                                    f"Embedding dim mismatch for product {product_id}! Expected {DIM}, got {emb.shape[1]}. Skipping."
                                )
                                # This indicates a problem with the embedding model or config
                                break  # Stop retrying for this product

                            batch_vectors.append(emb.astype("float32"))
                            batch_mapping_update[current_faiss_idx] = product_id
                            current_faiss_idx += 1
                            success = True
                            break  # Success
                        else:
                            logging.warning(
                                f"Received empty embedding for product {product_id}. Text: '{text}', Image: {image_path}"
                            )
                            # Don't retry if embedding is empty unless it's a known transient error

                except ConnectionError as e:  # Specific error for Triton/network issues
                    logging.warning(
                        f"Connection error processing product {product_id}: {e}. Retry {retries + 1}..."
                    )
                    if CURRENT_MODE == "cloud":
                        time.sleep(5 * (retries + 1))  # Basic backoff
                except Exception as e:
                    logging.error(
                        f"Error processing product {product_id}: {e}", exc_info=True
                    )
                    if CURRENT_MODE == "cloud":
                        time.sleep(5 * (retries + 1))  # Basic backoff for cloud errors
                    else:
                        break  # Don't retry local errors unless specifically handled

                retries += 1
                if not success and retries >= (
                    CLOUD_MAX_RETRIES if CURRENT_MODE == "cloud" else 1
                ):
                    logging.error(
                        f"Failed to process product {product_id} after retries."
                    )

        # Add collected vectors for the batch to the main list and update mapping
        if batch_vectors:
            vectors.extend(batch_vectors)
            id_mapping.update(batch_mapping_update)
            logging.info(f"Collected {len(batch_vectors)} vectors in this batch.")
            # Save partial progress after each successful batch processing
            save_partial_progress(vectors, id_mapping)
        else:
            logging.info("No vectors generated in this batch.")

        # Apply rate limit only for cloud mode
        if CURRENT_MODE == "cloud" and i + batch_size < len(products_to_process):
            logging.info(
                f"Waiting {CLOUD_RATE_LIMIT_DELAY} seconds before next batch (Cloud mode)..."
            )
            time.sleep(CLOUD_RATE_LIMIT_DELAY)

    logging.info("Batch processing complete.")
    if vectors:
        logging.info(f"Finalizing index with {len(vectors)} total vectors...")
        # Reshape the final list of vectors [(1, DIM), (1, DIM), ...] into (N, DIM)
        final_embeddings_np = np.concatenate(vectors, axis=0).astype("float32")

        if final_embeddings_np.shape[1] != DIM:
            raise ValueError(
                f"FATAL: Final vector dimension mismatch! Expected {DIM}, got {final_embeddings_np.shape[1]}."
            )

        # Add vectors to the index (potentially replacing if rebuilding)
        if force_rebuild or index.ntotal == 0:  # Check if index needs reset
            if index.ntotal > 0:
                index.reset()  # Clear existing vectors if rebuilding
            index.add(final_embeddings_np)
            logging.info(f"Added {index.ntotal} vectors to the new index.")
        else:
            # If resuming, we assume vectors only contains NEW vectors
            # Note: This assumes the partial progress logic correctly identified unprocessed items.
            index.add(final_embeddings_np)
            logging.info(
                f"Added {len(final_embeddings_np)} new vectors. Index total: {index.ntotal}"
            )

        save_index()  # Save the final index and mapping
        logging.info("Saved final index to disk.")

        # Clean up partial progress file on successful completion
        if os.path.exists(PARTIAL_INDEX_PATH):
            try:
                os.remove(PARTIAL_INDEX_PATH)
            except OSError as e:
                logging.warning(f"Could not remove partial progress file: {e}")
    else:
        logging.info("No vectors generated during the build process.")


def search_faiss(query_vector: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
    """Search the FAISS index for the nearest neighbors of a query vector
    Returns a list of tuples (product_id, similarity_score)"""
    global index, id_mapping
    if index is None:
        logging.warning("FAISS index is not loaded.")
        return []
    if not id_mapping:  # Check if mapping is loaded
        logging.warning("ID mapping is not loaded.")
        return []
    if index is None or index.ntotal == 0:
        logging.warning("Search called but FAISS index is not loaded or empty.")
        return []
    if DIM <= 0:
        logging.error("Search called with invalid dimension {DIM}")
        return []

    try:
        # Ensure query vector is 2D float32 and correct dimension
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        if query_vector.shape[1] != DIM:
            logging.error(
                f"Query vector dimension mismatch! Expected {DIM}, got {query_vector.shape[1]}."
            )
            return []
        query_vector = query_vector.astype("float32")

        distances, indices = index.search(query_vector, k)

        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # FAISS uses -1 for invalid index
                product_id = id_mapping.get(int(idx))
                if product_id is not None:
                    similarity = int(
                        normalize_distance_to_similarity(float(distances[0][i])) * 100
                    )
                    results.append((product_id, similarity))

        return results
    except Exception as e:
        logging.error(f"FAISS search error: {e}", exc_info=True)
        return []


def normalize_distance_to_similarity(
    distance: float, scale_factor: float = 5.0
) -> float:
    """Convert L2 distance to similarity score between 0 and 1"""
    return 1.0 / (1.0 + distance / scale_factor)


# sync_databases function remains the same conceptually
def sync_databases():
    """Synchronize MongoDB and FAISS index"""
    from mongo import products_col  # Import here to avoid circular imports

    # Check if index is loaded
    if index is None or not id_mapping:
        logging.warning("Cannot sync: FAISS index or mapping not loaded.")
        # Try loading it?
        if not load_index():
            logging.error("Failed to load index for sync check.")
            return False  # Indicate potential issue

    # Get all product IDs from MongoDB
    try:
        mongo_product_ids = set(
            doc["product_id"] for doc in products_col.find({}, {"product_id": 1})
        )
    except Exception as e:
        logging.error(f"Failed to get product IDs from MongoDB: {e}")
        return False  # Cannot compare

    # Get all product IDs from FAISS index mapping
    faiss_product_ids = set(id_mapping.values())

    # Find mismatches
    missing_in_faiss = mongo_product_ids - faiss_product_ids
    missing_in_mongo = faiss_product_ids - mongo_product_ids

    if missing_in_faiss or missing_in_mongo:
        logging.warning("Data mismatch detected:")
        if missing_in_faiss:
            logging.warning(
                f"Products in MongoDB but not in FAISS ({len(missing_in_faiss)}): {list(missing_in_faiss)[:10]}..."
            )  # Show sample
        if missing_in_mongo:
            logging.warning(
                f"Products in FAISS but not in MongoDB ({len(missing_in_mongo)}): {list(missing_in_mongo)[:10]}..."
            )  # Show sample
        return False

    logging.info(f"Databases are in sync ({len(mongo_product_ids)} products).")
    return True


if __name__ == "__main__":
    logging.info(f"Running vector_db main script in {CURRENT_MODE} mode.")
    # Ensure index is built
    build_faiss_index(force_rebuild=False)  # Set force_rebuild=True to test rebuilding

    # Example search only if index built successfully
    if index is not None and index.ntotal > 0:
        logging.info("Performing example search...")
        # Use a sample product for query
        try:
            sample_product_id_to_query = list(id_mapping.values())[
                0
            ]  # Get first product ID
            from mongo import get_product_by_id  # Local import for testing

            sample_product = get_product_by_id(sample_product_id_to_query)

            if (
                sample_product
                and "image_path" in sample_product
                and os.path.exists(sample_product["image_path"])
            ):
                query_text = sample_product.get("name", "sample query")
                query_image_path = sample_product["image_path"]
                with Image.open(query_image_path) as img:
                    img = img.convert("RGB")
                    query_emb = get_combined_embedding(query_text, img)

                    if query_emb is not None and query_emb.size > 0:
                        top_results = search_faiss(query_emb, k=5)
                        logging.info(
                            f"Search Results for product {sample_product_id_to_query} ('{query_text}'):"
                        )
                        for pid, score in top_results:
                            logging.info(f"  - Product ID: {pid}, Score: {score:.4f}")
                    else:
                        logging.warning(
                            "Failed to generate query embedding for example search."
                        )
            else:
                logging.warning(
                    "Could not find sample product or image path for example search."
                )
        except Exception as e:
            logging.error(f"Error during example search: {e}", exc_info=True)
    else:
        logging.warning("Index is empty or not loaded, skipping example search.")

    # Check sync
    logging.info("Running database sync check...")
    sync_databases()
