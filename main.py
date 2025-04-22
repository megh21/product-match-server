from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException
#add cors
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
# Import from the new dispatcher
from services.embeddings import get_combined_embedding, CURRENT_MODE, DIM_COMBINED
from vectordb.vector_db import build_faiss_index, search_faiss, sync_databases
from mongo import (
    get_product_by_id,
    get_db_stats,
)  # Keep insert_sample_products for potential manual trigger?

# from ingest import create_product_documents # Maybe call ingest from command line now?
import shutil
import uuid
import os
from dotenv import load_dotenv
from PIL import Image  # Need PIL here
import logging

logging.basicConfig(level=logging.INFO)
load_dotenv()

UPLOAD_FOLDER = "uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logging.info("Application lifespan startup...")
    logging.info(f"Running in {CURRENT_MODE} embedding mode.")
    logging.info(f"Expected combined embedding dimension: {DIM_COMBINED}")

    if DIM_COMBINED <= 0:
        logging.error(
            "Fatal: Invalid embedding dimension configured. Check .env and embeddings module."
        )
        # Application should probably not start
        raise RuntimeError("Invalid embedding dimension configured.")

    try:
        # logging.info("Verifying MongoDB data (optional step)...")
        # insert_sample_products(create_product_documents())

        mongo_stats = get_db_stats()
        logging.info(
            f"MongoDB Check: {mongo_stats.get('product_count', 0)} products found."
        )
        if mongo_stats.get("product_count", 0) == 0:
            logging.warning(
                "MongoDB product collection is empty. Run ingestion script."
            )
            # Optionally raise error or proceed cautiously

        # Build/load FAISS index using vector_db logic
        logging.info("Initializing FAISS index...")
        # Set force_rebuild=True only for debugging/first run if needed
        build_faiss_index(force_rebuild=False)
        logging.info("FAISS index initialization complete.")

        # Check synchronization after index build/load
        if not sync_databases():
            logging.warning("Databases are out of sync after startup!")
            # Decide action: Log warning? Force rebuild? Stop server?
            # For now, just log the warning. A periodic check might be better.
            # Consider adding a flag to force rebuild on sync failure at startup
            # build_faiss_index(force_rebuild=True)
            # if not sync_databases():
            #     raise RuntimeError("Failed to synchronize databases even after rebuild.")

    except Exception as e:
        logging.error(f"Fatal error during startup: {e}", exc_info=True)
        # Re-raise to prevent FastAPI from starting in a bad state
        raise e

    yield
    # Cleanup logic if needed on shutdown
    logging.info("Application lifespan shutdown.")


v = os.getenv("VERSION", "dev")
app = FastAPI(
    title=f"AI-product-Match({CURRENT_MODE.lower()})", lifespan=lifespan, version=v
)

# Static files for serving images
app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_FOLDER),
    name="uploads",
)



@app.post("/api/match/")
async def match_product(image: UploadFile = File(...), name: str = "Query Product"):
    temp_file_path = None
    try:
        # Generate unique filename for uploaded image
        _, ext = os.path.splitext(image.filename or ".jpg")
        temp_file_name = f"{uuid.uuid4()}{ext}"
        temp_file_path = os.path.join(UPLOAD_FOLDER, temp_file_name)

        # Save uploaded file safely
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(image.file, buffer)
            logging.info(f"Uploaded file saved to {temp_file_path}")
        except Exception as e:
            logging.error(f"Failed to save uploaded file: {e}")
            raise HTTPException(status_code=500, detail="Error saving uploaded image.")
        finally:
            # Ensure file object is closed even if copy fails
            if image.file and not image.file.closed:
                image.file.close()

        # --- Get Embeddings using the dispatcher ---
        try:
            # Load image using PIL
            with Image.open(temp_file_path) as img:
                img = img.convert("RGB")  # Ensure RGB format
                # Use the imported embedding function
                logging.info(f"Generating embedding for {name}...")
                query_emb = get_combined_embedding(name, img)
                logging.info(f"Generated embedding for {name})")
        except ValueError as ve:
            # too many tries
            logging.error(f"ValueError: {ve}", exc_info=True)
            raise HTTPException(
                status_code=429,
                detail="Error generating embedding. could be too many tries.",
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=400, detail="Uploaded file not found after saving."
            )
        except Exception as e:
            logging.error(
                f"Failed to process image or get embedding: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Error processing image or generating embedding.",
            )

        if query_emb is None or query_emb.size == 0:
            logging.error("Generated query embedding is empty.")
            raise HTTPException(
                status_code=500, detail="Failed to generate query embedding."
            )

        # Perform FAISS search
        try:
            logging.info(f"Searching FAISS index for {name}...")
            search_results = search_faiss(query_emb, k=5)  # Get top 5
        except Exception as e:
            logging.error(f"FAISS search failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error during product search.")

        if not search_results:
            # Return 200 OK but indicate no matches found
            return {"matches": [], "query": {"name": name}}

        # Get product details from MongoDB with scores
        matches = []
        for product_id, score in search_results:
            try:
                product = get_product_by_id(product_id)  # This returns dict or None
                if product:
                    # Optionally remove sensitive or large fields before returning
                    product.pop("image_path", None)  # Don't leak internal paths
                    matches.append(
                        {
                            "product": product,
                            "similarity_score": float(
                                score
                            ),  # Ensure score is JSON serializable
                        }
                    )
                    logging.info(f"Found product ID {product_id} with score {score}.")
                else:
                    logging.warning(
                        f"Product ID {product_id} found in FAISS but not in MongoDB."
                    )
            except Exception as e:
                logging.error(
                    f"Error fetching product details for ID {product_id}: {e}"
                )
                # Decide whether to skip this match or raise an error
        logging.info(f"Total matches found: {len(matches)}")
        return {"matches": matches, "query": {"name": name}}

    except HTTPException as http_exc:
        # Re-raise HTTP exceptions directly
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors
        logging.error(f"Unexpected error in /match endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )

    finally:
        # Cleanup temporary uploaded file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                logging.warning(
                    f"Could not remove temporary upload file {temp_file_path}: {e}"
                )


@app.get("/")
async def root():
    return JSONResponse(
        content={
            "message": "Welcome to the AI-product-Match API!",
            "version": v,
            "embedding_mode": CURRENT_MODE,
            "documentation_url": "/docs",
        }
    )


@app.get("/health")
async def health_check():
    # Basic health check
    status = {"status": "ok", "embedding_mode": CURRENT_MODE}
    # Check Triton connection if in local mode
    if CURRENT_MODE == "local":
        try:
            from services.embeddings_local import triton_client_instance

            if (
                triton_client_instance
                and triton_client_instance.client.is_server_live()
            ):
                status["triton_connection"] = "live"
            else:
                status["triton_connection"] = "down"
        except Exception:
            status["triton_connection"] = "error"
    # Check MongoDB connection
    try:
        mongo_stats = get_db_stats()
        status["mongodb_status"] = "connected"
        status["mongodb_products"] = mongo_stats.get("product_count", "error")
    except Exception:
        status["mongodb_status"] = "error"
    # Check FAISS index
    try:
        from vectordb.vector_db import index as faiss_index

        if faiss_index and faiss_index.ntotal > 0:
            status["faiss_status"] = "loaded"
            status["faiss_vectors"] = faiss_index.ntotal
        else:
            status["faiss_status"] = "not_loaded_or_empty"
    except Exception:
        status["faiss_status"] = "error"

    return status


@app.get("/sync_check")
async def run_sync_check():
    # Manual trigger for sync check
    is_synced = sync_databases()
    return {"databases_in_sync": is_synced}
