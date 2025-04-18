import os
import faiss
import numpy as np
from embedding_v_zero import get_combined_embedding
from ingest import create_product_documents
import pickle
import time
from typing import Dict, List, Tuple

# Paths 
DATA_DIR = "data"  # Base data directory
IMAGE_DIR = os.path.join(DATA_DIR, "images")
INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
MAPPING_PATH = os.path.join(DATA_DIR, "id_mapping.pkl")
PARTIAL_INDEX_PATH = os.path.join(DATA_DIR, "partial_index.pkl")

# FAISS Setup
DIM = 2048  # Combined embedding dimension
index = None
id_mapping = {}

# Constants
BATCH_SIZE = 5  # Process 5 products at a time
RATE_LIMIT_DELAY = 60  # Wait 60 seconds between batches
MAX_RETRIES = 3

def ensure_data_dir():
    """Ensure data directory exists"""
    os.makedirs(DATA_DIR, exist_ok=True)

def save_index():
    """Save FAISS index and id mapping to disk"""
    ensure_data_dir()
    faiss.write_index(index, INDEX_PATH)
    with open(MAPPING_PATH, 'wb') as f:
        pickle.dump(id_mapping, f)

def load_index():
    """Load FAISS index and id mapping from disk"""
    global index, id_mapping
    
    # Initialize index if None
    if index is None:
        index = faiss.IndexFlatL2(DIM)
    
    try:
        if os.path.exists(INDEX_PATH) and os.path.exists(MAPPING_PATH):
            index = faiss.read_index(INDEX_PATH)
            with open(MAPPING_PATH, 'rb') as f:
                id_mapping = pickle.load(f)
            return True
        return False
    except (FileNotFoundError, EOFError, RuntimeError):
        return False

def process_batch(products: List[dict], start_idx: int) -> Tuple[List[np.ndarray], Dict[int, int]]:
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
                    print(f'error: {e}')    
                    print(f"Failed to process product {prod['product_id']} after {MAX_RETRIES} retries")
                else:
                    wait_time = (2 ** retries) * 5  # Exponential backoff
                    print(f"Retrying product {prod['product_id']} in {wait_time} seconds...")
                    time.sleep(wait_time)
    
    return vectors, batch_mapping

def save_partial_progress(vectors: List[np.ndarray], current_mapping: Dict[int, int]):
    """Save partial progress during index building"""
    ensure_data_dir() # Ensure directory exists
    data = {
        'vectors': np.array(vectors) if vectors else np.array([]),
        'mapping': current_mapping
    }
    try:
        with open(PARTIAL_INDEX_PATH, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved partial progress: {len(vectors)} vectors")
    except Exception as e:
        print(f"Error saving partial progress: {e}")

def load_partial_progress() -> Tuple[List[np.ndarray], Dict[int, int]]:
    """Load partial progress if available"""
    if not os.path.exists(PARTIAL_INDEX_PATH):
        return [], {}
        
    try:
        with open(PARTIAL_INDEX_PATH, 'rb') as f:
            data = pickle.load(f)
            vectors = data['vectors'].tolist() if len(data['vectors']) > 0 else []
            mapping = data['mapping']
            print(f"Loaded partial progress: {len(vectors)} vectors")
            return vectors, mapping
    except Exception as e:
        print(f"Error loading partial progress: {e}")
        return [], {}

def build_faiss_index(force_rebuild=False):
    """Build or load FAISS index with rate limiting and resume capability"""
    global index, id_mapping
    
    if not force_rebuild and load_index():
        print("Loaded existing FAISS index")
        return

    print("Building new index...")
    if index is None:
        index = faiss.IndexFlatL2(DIM)
    
    # Try loading partial progress
    vectors, id_mapping = load_partial_progress() if not force_rebuild else ([], {})
    start_idx = len(vectors)
    print(f"Resuming from index {start_idx}...")
    if start_idx > 0:
        print(f"Loaded {len(vectors)} vectors from partial progress")
    else:
        print("No partial progress found, starting from scratch")
    # Create product documents
    products = create_product_documents()
    print(f"Total products to process: {len(products)}")
    # print(products)
    remaining_products = products[start_idx:]
    
    print(f"Processing {len(remaining_products)} products in batches of {BATCH_SIZE}...")
    
    for i in range(0, len(remaining_products), BATCH_SIZE):
        batch = remaining_products[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{len(remaining_products)//BATCH_SIZE + 1}")
        
        batch_vectors, batch_mapping = process_batch(batch, start_idx + i)
        vectors.extend(batch_vectors)
        id_mapping.update(batch_mapping)
        
        # Save partial progress
        save_partial_progress(vectors, id_mapping)
        
        if i + BATCH_SIZE < len(remaining_products):
            print(f"Waiting {RATE_LIMIT_DELAY} seconds before next batch...")
            time.sleep(RATE_LIMIT_DELAY)
    #print
    print("Batch processing complete")
    if vectors:
        print(f"Indexing {len(vectors)} product embeddings... 1")
        print('Original vectors shape:', np.array(vectors).shape)
        
        # Reshape vectors to correct dimension
        embeddings_np = np.array(vectors).reshape(len(vectors), -1).astype("float32")
        
        # Verify shape
        print(f"Reshaped embeddings shape: {embeddings_np.shape}")
        if embeddings_np.shape[1] != DIM:
            raise ValueError(f"Expected vector dimension {DIM}, got {embeddings_np.shape[1]}")
            
        index.add(embeddings_np)
        print(f"Indexed {len(vectors)} product embeddings 2")
        save_index()
        print("Saved index to disk 3")
        
        # Clean up partial progress file
        if os.path.exists(PARTIAL_INDEX_PATH):
            os.remove(PARTIAL_INDEX_PATH)
    else:
        print("No vectors to index! 4")

def search_faiss(query_vector: np.ndarray, k: int = 3) -> List[Tuple[int, float]]:
    """Search FAISS index for similar vectors"""
    # Ensure vector is 2D array with correct shape
    query_vector = np.array(query_vector).reshape(1, -1).astype('float32')
    
    try:
        # Perform search
        distances, indices = index.search(query_vector, k)
        
        # Convert results to list of (product_id, score) tuples
        results = []
        for idx, dist in zip(indices[0], distances[0]):
            if idx != -1:  # Valid index
                product_id = id_mapping.get(idx)
                if product_id is not None:
                    results.append((product_id, float(dist)))
        
        return results
    except Exception as e:
        print(f"Search error: {e}")
        return []

def sync_databases():
    """Synchronize MongoDB and FAISS index"""
    from mongo import products_col  # Import here to avoid circular imports
    
    # Get all product IDs from MongoDB
    mongo_product_ids = set(doc["product_id"] for doc in products_col.find({}, {"product_id": 1}))
    
    # Get all product IDs from FAISS index
    faiss_product_ids = set(id_mapping.values())
    
    # Find mismatches
    missing_in_faiss = mongo_product_ids - faiss_product_ids
    missing_in_mongo = faiss_product_ids - mongo_product_ids
    
    if missing_in_faiss or missing_in_mongo:
        print(f"Data mismatch detected:")
        print(f"Products in MongoDB but not in FAISS: {len(missing_in_faiss)}")
        print(f"Products in FAISS but not in MongoDB: {len(missing_in_mongo)}")
        return False
    
    print("Databases are in sync")
    return True

if __name__ == "__main__":
    build_faiss_index()
    # Example search
    image_in = os.path.join(IMAGE_DIR, "10000.jpg")
    text_in = "Sample text"
    query_emb = get_combined_embedding(text_in, image_in)
    top_ids = search_faiss(query_emb, k=3)
    print("Top IDs:", top_ids)