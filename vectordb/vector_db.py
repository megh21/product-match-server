"""
Vector Database module for efficient product matching using FAISS.

This module provides functionality to build, search, and manage a FAISS vector index
for product embedding vectors.
"""

import os
import asyncio
import faiss
import numpy as np
import pickle
import logging
from typing import Dict, List, Tuple, Optional, Any

from dotenv import load_dotenv

from services.embeddings import (
    get_combined_embedding,
    DIM_COMBINED,
    CURRENT_MODE,
)
from ingest import create_product_documents


# Initialize logging
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Paths
DATA_DIR = "data"
IMAGE_DIR = os.path.join(DATA_DIR, "images")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
MAPPING_PATH = os.path.join(DATA_DIR, "id_mapping.pkl")
PARTIAL_INDEX_PATH = os.path.join(DATA_DIR, "partial_index_new.pkl")

# FAISS Setup
DIM = DIM_COMBINED  # Get dimension from embeddings module
index = None
id_mapping = {}

# Configuration for different modes
CONFIG = {
    "local": {
        "batch_size": int(os.getenv("LOCAL_BATCH_SIZE", "16")),
        "max_workers": int(os.getenv("LOCAL_MAX_WORKERS", "4")),
        "rate_limit_delay": 0,
        "max_retries": 2,
        "retry_delay": 1,
    },
    "cloud": {
        "batch_size": int(os.getenv("CLOUD_BATCH_SIZE", "5")),
        "max_workers": 1,  # Limited concurrency for cloud API
        "rate_limit_delay": int(os.getenv("CLOUD_RATE_LIMIT_DELAY", "60")),
        "max_retries": int(os.getenv("CLOUD_MAX_RETRIES", "3")),
        "retry_delay": 5,
    },
}


# File management functions
def ensure_data_dir() -> None:
    """Ensure data directory exists."""
    os.makedirs(DATA_DIR, exist_ok=True)


def save_index() -> None:
    """Save FAISS index and id mapping to disk."""
    ensure_data_dir()
    if index is None or index.ntotal == 0:
        logger.warning("Index is None or empty, nothing to save.")
        return

    try:
        faiss.write_index(index, INDEX_PATH)
        with open(MAPPING_PATH, "wb") as f:
            pickle.dump(id_mapping, f)
        logger.info(f"Saved FAISS index ({index.ntotal} vectors) and mapping.")
    except Exception as e:
        logger.error(f"Error saving index: {e}", exc_info=True)


def load_index() -> bool:
    """
    Load FAISS index and id mapping from disk.

    Returns:
        bool: True if index was successfully loaded, False otherwise
    """
    global index, id_mapping

    # Initialize index if it doesn't exist
    if index is None:
        if DIM <= 0:
            logger.error(
                f"Invalid embedding dimension: {DIM}. Cannot initialize FAISS."
            )
            return False
        index = faiss.IndexFlatL2(DIM)

    # Load existing index if available
    if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
        try:
            logger.info(f"Loading FAISS index from {INDEX_PATH}")
            index = faiss.read_index(INDEX_PATH)

            # Validate index dimension
            if index.d != DIM:
                logger.error(
                    f"Dimension mismatch: expected {DIM}, got {index.d}. Rebuilding required."
                )
                _reset_index_files()
                return False

            # Load ID mapping
            with open(MAPPING_PATH, "rb") as f:
                id_mapping = pickle.load(f)
            logger.info(f"Loaded existing FAISS index with {index.ntotal} vectors.")
            return True

        except Exception as e:
            logger.error(f"Error loading index files: {e}", exc_info=True)
            index = faiss.IndexFlatL2(DIM)
            id_mapping = {}
            return False
    else:
        logger.info("Index files not found.")
        return False


def _reset_index_files() -> None:
    """Reset index to a clean state and remove any existing files."""
    global index, id_mapping

    # Reset in-memory state
    index = faiss.IndexFlatL2(DIM)
    id_mapping = {}

    # Remove files
    for path in [INDEX_PATH, MAPPING_PATH, PARTIAL_INDEX_PATH]:
        if os.path.exists(path):
            try:
                os.remove(path)
            except OSError as e:
                logger.warning(f"Could not remove file {path}: {e}")


def save_partial_progress(
    vectors: List[np.ndarray], current_mapping: Dict[int, int]
) -> None:
    """Save partial progress during index building."""
    ensure_data_dir()
    data = {
        "vectors": np.array(vectors) if vectors else np.array([]),
        "mapping": current_mapping,
    }

    try:
        with open(PARTIAL_INDEX_PATH, "wb") as f:
            pickle.dump(data, f)
        logger.info(f"Saved partial progress: {len(vectors)} vectors")
    except Exception as e:
        logger.error(f"Error saving partial progress: {e}")


def load_partial_progress() -> Tuple[List[np.ndarray], Dict[int, int]]:
    """Load partial progress for resuming index building."""
    if not os.path.exists(PARTIAL_INDEX_PATH):
        return [], {}

    try:
        with open(PARTIAL_INDEX_PATH, "rb") as f:
            data = pickle.load(f)
        vectors = data["vectors"].tolist() if len(data["vectors"]) > 0 else []
        mapping = data["mapping"]
        logger.info(f"Loaded partial progress: {len(vectors)} vectors")
        return vectors, mapping
    except Exception as e:
        logger.error(f"Error loading partial progress: {e}. Starting fresh.")
        return [], {}


# Product processing functions
async def process_product_async(
    product: dict, index: int, config: Dict[str, Any]
) -> Optional[Tuple[np.ndarray, int]]:
    """Process a single product to generate embeddings asynchronously."""
    retries = 0
    max_retries = config["max_retries"]
    retry_delay = config["retry_delay"]

    while retries < max_retries:
        try:
            image_path = product["image_path"]
            emb = await asyncio.to_thread(
                get_combined_embedding, product["name"], image_path
            )
            return emb, product["product_id"]
        except Exception as e:
            retries += 1
            if retries == max_retries:
                logger.error(
                    f"Failed to process product {product['product_id']} after {max_retries} retries: {e}"
                )
            else:
                # Exponential backoff
                wait_time = retry_delay * (2**retries)
                logger.warning(
                    f"Retrying product {product['product_id']} in {wait_time} seconds..."
                )
                await asyncio.sleep(wait_time)

    return None


async def process_batch_async(
    products: List[dict], start_idx: int, config: Dict[str, Any]
) -> Tuple[List[np.ndarray], Dict[int, int]]:
    """Process a batch of products for embedding generation asynchronously."""
    tasks = [
        process_product_async(prod, start_idx + i, config)
        for i, prod in enumerate(products)
    ]

    results = await asyncio.gather(*tasks)

    vectors = []
    batch_mapping = {}

    for i, result in enumerate(results):
        if result:
            emb, product_id = result
            vectors.append(emb)
            batch_mapping[start_idx + i] = product_id

    return vectors, batch_mapping


# Vector search functions
def normalize_distance_to_similarity(
    distance: float, scale_factor: float = 5.0
) -> float:
    """Convert FAISS L2 distance to a similarity score between 0 and 1."""
    return 1.0 / (1.0 + distance / scale_factor)


async def search_faiss_async(
    query_vector: np.ndarray, k: int = 3
) -> List[Tuple[str, float]]:
    """
    Search FAISS index asynchronously.

    Args:
        query_vector: Embedding vector to search with
        k: Number of results to return

    Returns:
        List of tuples (product_id, similarity_score)
    """
    global index, id_mapping

    # Validate index state
    if index is None or index.ntotal == 0 or not id_mapping:
        logger.warning(
            "Search called but FAISS index is not loaded, empty, or mapping missing"
        )
        return []

    if DIM <= 0:
        logger.error(f"Search called with invalid dimension {DIM}")
        return []

    try:
        # Define search function to run in a separate thread
        def _search():
            # Prepare query vector
            q_vector = query_vector.copy()
            if q_vector.ndim == 1:
                q_vector = q_vector.reshape(1, -1)
            if q_vector.shape[1] != DIM:
                raise ValueError(
                    f"Query vector dimension mismatch! Expected {DIM}, got {q_vector.shape[1]}."
                )
            q_vector = q_vector.astype("float32")

            # Run FAISS search
            distances, indices = index.search(q_vector, k)

            # Process results
            results = []
            for i, idx in enumerate(indices[0]):
                if idx != -1:  # FAISS uses -1 for invalid index
                    product_id = id_mapping.get(int(idx))
                    if product_id is not None:
                        similarity = int(
                            normalize_distance_to_similarity(float(distances[0][i]))
                            * 100
                        )
                        results.append((product_id, similarity))
            return results

        # Run search in a thread to avoid blocking the event loop
        return await asyncio.to_thread(_search)

    except Exception as e:
        logger.error(f"FAISS search error: {e}", exc_info=True)
        return []


def search_faiss(query_vector: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
    """
    Synchronous wrapper for FAISS search.

    Args:
        query_vector: Embedding vector to search with
        k: Number of results to return

    Returns:
        List of tuples (product_id, similarity_score)
    """
    try:
        return asyncio.run(search_faiss_async(query_vector, k))
    except RuntimeError:
        # If already in an event loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(search_faiss_async(query_vector, k))


# Index building functions
async def build_faiss_index_async(force_rebuild: bool = False) -> bool:
    """
    Build or update FAISS index asynchronously.

    Args:
        force_rebuild: Whether to rebuild the index from scratch

    Returns:
        bool: True if index was built successfully, False otherwise
    """
    global index, id_mapping

    # Validate embedding dimension
    if DIM <= 0:
        logger.error(f"Cannot build index with invalid dimension: {DIM}")
        return False

    # Use existing index if available and not forcing rebuild
    if not force_rebuild and load_index():
        logger.info("Using existing FAISS index.")
        return True

    logger.info("Building new FAISS index...")

    # Initialize index if needed
    if index is None or index.d != DIM:
        index = faiss.IndexFlatL2(DIM)
        id_mapping = {}
        logger.info(f"Initialized FAISS index with dimension {DIM}")

    # Load partial progress or start fresh
    vectors, id_mapping = load_partial_progress() if not force_rebuild else ([], {})
    start_faiss_idx = len(vectors)
    processed_product_ids = set(id_mapping.values())

    # Get products to process
    products = create_product_documents()
    products_to_process = [
        p for p in products if p["product_id"] not in processed_product_ids
    ]

    logger.info(f"Total products: {len(products)}")
    logger.info(f"Products already indexed: {len(processed_product_ids)}")
    logger.info(f"Products to process: {len(products_to_process)}")

    # Get configuration based on current mode
    config = CONFIG[CURRENT_MODE]
    batch_size = config["batch_size"]
    rate_limit_delay = config["rate_limit_delay"]

    # Process products in batches
    total_batches = (len(products_to_process) + batch_size - 1) // batch_size

    for i in range(0, len(products_to_process), batch_size):
        batch_products = products_to_process[i : i + batch_size]
        batch_num = (i // batch_size) + 1
        logger.info(
            f"Processing batch {batch_num}/{total_batches} ({len(batch_products)} products)"
        )

        # Process batch
        batch_vectors, batch_mapping_update = await process_batch_async(
            batch_products, start_faiss_idx + len(vectors), config
        )

        # Update vectors and mapping
        if batch_vectors:
            vectors.extend(batch_vectors)
            id_mapping.update(batch_mapping_update)
            logger.info(f"Added {len(batch_vectors)} vectors from this batch")
            save_partial_progress(vectors, id_mapping)
        else:
            logger.info("No vectors generated in this batch")

        # Apply rate limiting if needed
        if rate_limit_delay > 0 and i + batch_size < len(products_to_process):
            logger.info(f"Rate limiting: waiting {rate_limit_delay}s before next batch")
            await asyncio.sleep(rate_limit_delay)

    # Finalize index
    if vectors:
        logger.info(f"Finalizing index with {len(vectors)} total vectors...")

        # Concatenate vectors into a single numpy array
        try:
            final_embeddings_np = np.concatenate(vectors, axis=0).astype("float32")

            if final_embeddings_np.shape[1] != DIM:
                raise ValueError(
                    f"Vector dimension mismatch: expected {DIM}, got {final_embeddings_np.shape[1]}"
                )

            # Add vectors to index
            if force_rebuild or index.ntotal == 0:
                if index.ntotal > 0:
                    index.reset()
                index.add(final_embeddings_np)
                logger.info(f"Added {index.ntotal} vectors to new index")
            else:
                index.add(final_embeddings_np)
                logger.info(
                    f"Added {len(final_embeddings_np)} new vectors (total: {index.ntotal})"
                )

            # Save index to disk
            save_index()

            # Clean up partial progress
            if os.path.exists(PARTIAL_INDEX_PATH):
                try:
                    os.remove(PARTIAL_INDEX_PATH)
                except OSError as e:
                    logger.warning(f"Could not remove partial progress file: {e}")

            return True

        except Exception as e:
            logger.error(f"Error finalizing index: {e}", exc_info=True)
            return False
    else:
        logger.info("No vectors generated during build process")
        return False


def build_faiss_index(force_rebuild: bool = False) -> bool:
    """
    Synchronous wrapper for building FAISS index.

    Args:
        force_rebuild: Whether to rebuild the index from scratch

    Returns:
        bool: True if index was built successfully, False otherwise
    """
    try:
        return asyncio.run(build_faiss_index_async(force_rebuild))
    except RuntimeError:
        # If already in an event loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(build_faiss_index_async(force_rebuild))


# Database synchronization functions
async def sync_databases_async() -> bool:
    """
    Check if MongoDB and FAISS databases are in sync asynchronously.

    Returns:
        bool: True if databases are in sync, False otherwise
    """
    from mongo import products_col  # Import here to avoid circular imports

    # Load index if needed
    if index is None or not id_mapping:
        logger.warning("FAISS index not loaded, attempting to load...")
        if not load_index():
            logger.error("Failed to load index for sync check")
            return False

    try:
        # Get MongoDB product IDs
        def _get_mongo_ids():
            return set(
                doc["product_id"] for doc in products_col.find({}, {"product_id": 1})
            )

        mongo_product_ids = await asyncio.to_thread(_get_mongo_ids)

        # Get FAISS product IDs
        faiss_product_ids = set(id_mapping.values())

        # Check for mismatches
        missing_in_faiss = mongo_product_ids - faiss_product_ids
        missing_in_mongo = faiss_product_ids - mongo_product_ids

        if missing_in_faiss or missing_in_mongo:
            if missing_in_faiss:
                logger.warning(
                    f"Products in MongoDB but not in FAISS: {len(missing_in_faiss)}"
                )
            if missing_in_mongo:
                logger.warning(
                    f"Products in FAISS but not in MongoDB: {len(missing_in_mongo)}"
                )
            return False

        logger.info(f"Databases are in sync ({len(mongo_product_ids)} products)")
        return True

    except Exception as e:
        logger.error(f"Error checking database sync: {e}", exc_info=True)
        return False


def sync_databases() -> bool:
    """
    Synchronous wrapper for database sync check.

    Returns:
        bool: True if databases are in sync, False otherwise
    """
    try:
        return asyncio.run(sync_databases_async())
    except RuntimeError:
        # If already in an event loop
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(sync_databases_async())


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    logger.info(f"Running vector_db in {CURRENT_MODE} mode")
    asyncio.run(build_faiss_index_async(force_rebuild=False))
