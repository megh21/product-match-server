from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from embedding_v_zero import get_combined_embedding
from vector_db import build_faiss_index, search_faiss
from mongo import get_product_by_id
from mongo import insert_sample_products
from ingest import create_product_documents
import shutil
import uuid
import os
from dotenv import load_dotenv
from vector_db import sync_databases

load_dotenv()

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        # First, ensure MongoDB has all products
        insert_sample_products(create_product_documents())
        print("MongoDB data verified")
        
        # Then build/load FAISS index
        build_faiss_index(force_rebuild=not os.path.exists("data/faiss_index.bin"))
        print("FAISS index ready")
        
        # Check synchronization
        if not sync_databases():
            print("WARNING: Databases are out of sync")
            # Optionally force rebuild
            build_faiss_index(force_rebuild=True)
            if not sync_databases():
                raise RuntimeError("Failed to synchronize databases")
    
    except Exception as e:
        print(f"Startup error: {str(e)}")
        raise e
    
    yield

v = os.getenv("VERSION")    
app = FastAPI( title="AI Image match",
              lifespan=lifespan,
              version=v if v else "dev")

@app.post("/match/")
async def match_product(image: UploadFile = File(...), name: str = "Query Product"):
    temp_file = None
    try:
        # Save uploaded file
        temp_file = os.path.join(UPLOAD_FOLDER, f"{uuid.uuid4()}.jpg")
        with open(temp_file, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        # Get embeddings and search
        query_emb = get_combined_embedding(name, temp_file)
        search_results = search_faiss(query_emb, k=3)
        
        if not search_results:
            return {"error": "No matches found"}

        # Get product details with scores
        matches = []
        for product_id, score in search_results:
            product = get_product_by_id(product_id)
            if product:
                matches.append({
                    "product": product,
                    "similarity_score": float(score)  # Ensure score is float
                })

        return {
            "matches": matches,
            "query": {"name": name}
        }

    except Exception as e:
        return {"error": str(e)}
    
    finally:
        # Cleanup temporary file
        if temp_file and os.path.exists(temp_file):
            os.remove(temp_file)

#for app.get / redirect to and direclty load /docs
@app.get("/")
async def root():
    return {"message": "Welcome to the AI Image Match API. Visit /docs for documentation."}