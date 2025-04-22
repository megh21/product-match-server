from contextlib import asynccontextmanager
import os
import uuid
import shutil
import logging
from functools import lru_cache
from typing import Dict, Any

from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from dotenv import load_dotenv
from PIL import Image

from services.embeddings import get_combined_embedding, CURRENT_MODE, DIM_COMBINED
from mongo import get_product_by_id, get_db_stats
from model import ProductMatch, MatchResponse, HealthStatus, SyncCheckResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Constants
UPLOAD_FOLDER = "uploads/"
CACHE_SIZE = int(os.getenv("CACHE_SIZE", "100"))
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
        mongo_stats = get_db_stats()
        logging.info(
            f"MongoDB Check: {mongo_stats.get('product_count', 0)} products found."
        )
        if mongo_stats.get("product_count", 0) == 0:
            logging.warning(
                "MongoDB product collection is empty. Run ingestion script."
            )

        # Build/load FAISS index using vector_db logic
        logging.info("Initializing FAISS index...")
        # Use the async version directly since we're already in an async context
        from vectordb.vector_db import build_faiss_index_async

        await build_faiss_index_async(force_rebuild=False)
        logging.info("FAISS index initialization complete.")

        # Check synchronization after index build/load
        from vectordb.vector_db import sync_databases_async

        if not await sync_databases_async():
            logging.warning("Databases are out of sync after startup!")

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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update this to specific origins if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Static files for serving images
app.mount(
    "/uploads",
    StaticFiles(directory=UPLOAD_FOLDER),
    name="uploads",
)

# Add this after your existing StaticFiles mounts
app.mount("/_next", StaticFiles(directory="static/_next"), name="next_static")


@lru_cache(maxsize=CACHE_SIZE)
def get_cached_product_by_id(product_id: str) -> Dict[str, Any]:
    """
    Retrieve product from MongoDB with caching for improved performance.

    Args:
        product_id: The product ID to retrieve

    Returns:
        Product document as a dictionary
    """
    return get_product_by_id(product_id)


@app.post("/api/match/", response_model=MatchResponse, tags=["API"])
async def match_product(
    image: UploadFile = File(...),
    name: str = "Query Product",
    background_tasks: BackgroundTasks = None,
):
    """
    Match a product image against the vector database.

    This endpoint accepts an image file and returns similar products from the database,
    ranked by similarity score.

    Args:
        image: The product image to match
        name: Optional name for the query product
        background_tasks: FastAPI background tasks handler

    Returns:
        MatchResponse: Object containing matched products and query info

    Raises:
        HTTPException: For various error conditions with appropriate status codes
    """
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
            logger.info(f"Uploaded file saved to {temp_file_path}")
        except Exception as e:
            logger.error(f"Failed to save uploaded file: {e}")
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
                logger.info(f"Generating embedding for {name}...")
                query_emb = get_combined_embedding(name, img)
                logger.info(f"Generated embedding for {name}")
        except ValueError as ve:
            logger.error(f"ValueError: {ve}", exc_info=True)
            raise HTTPException(
                status_code=429, detail="Error generating embedding. Too many requests."
            )
        except FileNotFoundError:
            raise HTTPException(
                status_code=400, detail="Uploaded file not found after saving."
            )
        except Exception as e:
            logger.error(
                f"Failed to process image or get embedding: {e}", exc_info=True
            )
            raise HTTPException(
                status_code=500,
                detail="Error processing image or generating embedding.",
            )

        if query_emb is None or query_emb.size == 0:
            logger.error("Generated query embedding is empty.")
            raise HTTPException(
                status_code=500, detail="Failed to generate query embedding."
            )

        # Perform FAISS search
        try:
            logger.info(f"Searching FAISS index for {name}...")
            from vectordb.vector_db import search_faiss_async

            search_results = await search_faiss_async(query_emb, k=5)
        except Exception as e:
            logger.error(f"FAISS search failed: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail="Error during product search.")

        if not search_results:
            return MatchResponse(matches=[], query={"name": name})

        # Get product details from MongoDB with scores
        matches = []
        for product_id, score in search_results:
            try:
                product = get_cached_product_by_id(product_id)
                if product:
                    # Remove sensitive fields
                    product.pop("image_path", None)
                    matches.append(
                        ProductMatch(product=product, similarity_score=float(score))
                    )
                    logger.debug(f"Found product ID {product_id} with score {score}")
                else:
                    logger.warning(
                        f"Product ID {product_id} found in FAISS but not in MongoDB."
                    )
            except Exception as e:
                logger.error(f"Error fetching product details for ID {product_id}: {e}")

        logger.info(f"Total matches found: {len(matches)}")

        # Add cleanup to background tasks instead of using finally
        if background_tasks and temp_file_path:
            background_tasks.add_task(_remove_temp_file, temp_file_path)

        return MatchResponse(matches=matches, query={"name": name})

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Unexpected error in /match endpoint: {e}", exc_info=True)
        raise HTTPException(
            status_code=500, detail="An internal server error occurred."
        )
    finally:
        # If not using background tasks, clean up immediately
        if not background_tasks and temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
            except OSError as e:
                logger.warning(
                    f"Could not remove temporary upload file {temp_file_path}: {e}"
                )


def _remove_temp_file(file_path: str) -> None:
    """Helper function to remove temporary files in background tasks."""
    try:
        if os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Removed temporary file: {file_path}")
    except Exception as e:
        logger.warning(f"Failed to remove temporary file {file_path}: {e}")


@app.get("/")
async def root():
    """Serve the main HTML page."""
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthStatus, tags=["System"])
async def health_check():
    """
    Health check endpoint that reports system status.

    Checks the health of all system components:
    - API status
    - Embedding mode
    - MongoDB connection
    - FAISS vector database
    - Triton Inference Server (if applicable)

    Returns:
        HealthStatus: Object containing status of all components
    """
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
        except Exception as e:
            logger.warning(f"Error checking Triton status: {str(e)}")
            status["triton_connection"] = "error"

    # Check MongoDB connection
    try:
        mongo_stats = get_db_stats()
        status["mongodb_status"] = "connected"
        status["mongodb_products"] = str(mongo_stats.get("product_count", 0))
    except Exception as e:
        logger.error(f"MongoDB connection error: {str(e)}")
        status["mongodb_status"] = "error"

    # Check FAISS index
    try:
        from vectordb.vector_db import index as faiss_index

        if faiss_index and faiss_index.ntotal > 0:
            status["faiss_status"] = "loaded"
            status["faiss_vectors"] = faiss_index.ntotal
        else:
            status["faiss_status"] = "not_loaded_or_empty"
    except Exception as e:
        logger.error(f"FAISS check error: {str(e)}")
        status["faiss_status"] = "error"

    return HealthStatus(**status)


@app.get("/sync_check", response_model=SyncCheckResponse, tags=["System"])
async def run_sync_check():
    """
    Check if MongoDB and FAISS databases are in sync.

    This endpoint verifies that all products in MongoDB have corresponding
    vectors in the FAISS index and vice versa.

    Returns:
        SyncCheckResponse: Object containing sync status
    """
    from vectordb.vector_db import sync_databases_async

    is_synced = await sync_databases_async()
    return SyncCheckResponse(databases_in_sync=is_synced)
