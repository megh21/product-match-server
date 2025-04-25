# AI Product Matching System - Technical Documentation

## System Overview

The AI Product Matching System is a sophisticated platform designed to match products based on both visual and textual similarity. It leverages state-of-the-art vision and language models, vector databases, and NoSQL databases to create an efficient end-to-end product matching pipeline.

### Core Functionality

1. **Image and Text Embedding Generation**: Extracts meaningful features from product images and text descriptions
2. **Vector Database Indexing**: Stores embeddings in a FAISS vector database for efficient similarity search
3. **Metadata Storage**: Stores detailed product information in MongoDB
4. **Optimized Inference**: Uses NVIDIA Triton Inference Server and TensorRT to serve optimized models
5. **API Service**: Provides REST API endpoints for product matching

## System Architecture

### Component Breakdown

#### 1. Data Ingestion Layer

The data ingestion layer processes product data and prepares it for indexing:

- **Product Data Collection**: Imports product images and metadata
- **Embedding Generation**: Processes images and text through the embedding pipeline
- **Database Storage**: Saves product information in MongoDB and embeddings in FAISS

Key files:
- `ingest.py`: Orchestrates the ingestion process
- `mongo.py`: Handles MongoDB operations

#### 2. Vector Database Layer

The vector database layer manages the storage and retrieval of product embeddings:

- **FAISS Index**: Stores embeddings in an efficient searchable structure
- **ID Mapping**: Maps FAISS vector IDs to product IDs
- **Similarity Search**: Performs nearest neighbor search for query embeddings

Key files:
- `vectordb/vector_db.py`: Manages FAISS index operations

#### 3. Model Serving Layer

The model serving layer handles AI model deployment and inference:

- **Model Quantization**: Converts PyTorch models to optimized TensorRT engines
- **Triton Inference Server**: Serves the models with efficient resource utilization
- **Batching and Caching**: Optimizes inference throughput and latency

Key files:
- `tensorrt/`: Contains model export, calibration and engine building scripts
- `model_repository/`: Contains Triton model configuration
- `services/embeddings_local.py`: Local Triton-based embedding service

#### 4. API Service Layer

The API service layer handles external requests and business logic:

- **FastAPI Endpoints**: Exposes REST API endpoints for product matching
- **Request Handling**: Processes image uploads and generates query embeddings
- **Response Formatting**: Returns matched products with metadata and scores

Key files:
- `main.py`: FastAPI application and endpoints
- `model.py`: API data models and schemas

### Data Flow

1. **Ingestion Flow**:
   - Product data is collected → Embeddings are generated → Data is stored in databases

2. **Query Flow**:
   - Client uploads image → Image is processed → Embeddings are generated → 
     Vector search is performed → Product metadata is retrieved → Results are returned

## Key Components in Detail

### 1. MongoDB Integration

MongoDB is used to store product metadata and provides quick access to product information:

```python
# From mongo.py
def get_product_by_id(product_id):
    """
    Retrieve product from MongoDB with caching for improved performance.

    Args:
        product_id: The product ID to retrieve

    Returns:
        Product document as a dictionary
    """
    product = products_col.find_one({"product_id": product_id})
    if product:
        # Convert ObjectId to string
        product["_id"] = str(product["_id"])
    return product
```

The system uses standard MongoDB operations and includes error handling and connection management.

### 2. FAISS Vector Database

FAISS (Facebook AI Similarity Search) is used for efficient vector similarity search:

```python
# From vector_db.py
async def search_faiss_async(query_vector: np.ndarray, k: int = 3) -> List[Tuple[str, float]]:
    """
    Search FAISS index asynchronously.

    Args:
        query_vector: Embedding vector to search with
        k: Number of results to return

    Returns:
        List of tuples (product_id, similarity_score)
    """
    # ... code omitted for brevity ...
    
    # Run FAISS search
    distances, indices = index.search(q_vector, k)
    
    # Process results
    results = []
    for i, idx in enumerate(indices[0]):
        if idx != -1:  # FAISS uses -1 for invalid index
            product_id = id_mapping.get(int(idx))
            if product_id is not None:
                similarity = normalize_distance_to_similarity(float(distances[0][i]))
                results.append((product_id, similarity))
    return results
```

The system supports both synchronous and asynchronous search operations and includes utilities for distance-to-similarity conversion and index management.

### 3. Embedding Generation

The embedding generation component is designed to be flexible, supporting both local and cloud-based embedding strategies:

```python
# From embeddings.py
if EMBEDDING_MODE == "local":
    logging.info("Initializing LOCAL mode with Triton")
    try:
        from .embeddings_local import (
            get_text_embedding as _get_text_embedding,
            get_vision_embedding as _get_vision_embedding,
            get_combined_embedding as _get_combined_embedding,
        )
        # ... code omitted for brevity ...
    except ImportError as e:
        logging.error(f"Failed to import local embedding module: {e}")
        raise RuntimeError("Local embedding module failed to load")

elif EMBEDDING_MODE == "cloud":
    logging.info("Initializing CLOUD mode with Cohere")
    # ... code omitted for brevity ...
```

The system supports multiple model configurations and seamlessly switches between modes based on configuration.

### 4. Triton Integration

The system uses NVIDIA Triton Inference Server for efficient model serving:

```python
# From embeddings_local.py
class TritonClient:
    def __init__(self, url: str):
        self.url = url
        self.text_tokenizer, self.vision_processor = load_processors()
        self.client = None
        self._initialize_client()
        try:
            self.client = httpclient.InferenceServerClient(url=self.url, verbose=False)
            if not self.client.is_server_live():
                raise ConnectionError(f"Triton server at {url} is not live.")
            logging.info(f"Triton client connected to {url}")
        except Exception as e:
            logging.error(f"Failed to initialize Triton client: {e}", exc_info=True)
            raise
    
    # ... methods for text and vision inference ...
```

The system includes a comprehensive client implementation that handles preprocessing, inference, and result extraction.

### 5. TensorRT Model Optimization

The system includes scripts for model quantization using TensorRT:

```python
# From tensorrt/internvl/build_engine_vision.py
def build_engine(
    onnx_path,
    engine_path_prefix,
    input_names,
    profile_shapes,
    precision="fp16",
    calibrator_factory=None,
    cache_file=None,
):
    """Builds and saves a TensorRT engine."""
    # ... code omitted for brevity ...
    
    # Set precision mode (FP16, INT8, etc.)
    if precision == "fp16":
        if builder.platform_has_fast_fp16:
            config.set_flag(trt.BuilderFlag.FP16)
            logging.info("FP16 mode enabled.")
        else:
            logging.warning("FP16 not supported/fast.")
            precision = "fp32"
    elif precision == "int8":
        if builder.platform_has_fast_int8:
            logging.info("Enabling INT8 mode.")
            config.set_flag(trt.BuilderFlag.INT8)
            if builder.platform_has_fast_fp16:
                logging.info("Enabling FP16 mode alongside INT8.")
                config.set_flag(trt.BuilderFlag.FP16)
            if calibrator_factory is None:
                logging.error("Calibrator factory needed for INT8.")
                return False
            logging.info("Creating INT8 calibrator...")
            calibrator = calibrator_factory()  # Call the factory function
            if calibrator is None:
                logging.error("Failed to create calibrator.")
                return False
            config.int8_calibrator = calibrator
            logging.info("INT8 calibrator set.")
    
    # ... code for building and saving engine ...
```

The system supports both FP16 and INT8 precision modes and includes calibration scripts for INT8 quantization.

### 6. FastAPI Implementation

The system uses FastAPI for the web service:

```python
# From main.py
@app.post("/api/match", response_model=MatchResponse, tags=["API"])
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
    """
    # ... code for image processing, embedding generation, and vector search ...
```

The system includes comprehensive error handling, response modeling, and asynchronous processing.

## Optimization Features

### 1. Caching

The system implements caching to improve performance for frequently accessed products:

```python
# From main.py
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
```

### 2. Asynchronous Processing

The system uses async/await patterns to improve throughput and responsiveness:

```python
# From vector_db.py
async def process_batch_async(
    products: List[dict], start_idx: int, config: Dict[str, Any]
) -> Tuple[List[np.ndarray], Dict[int, int]]:
    """Process a batch of products for embedding generation asynchronously."""
    tasks = [
        process_product_async(prod, start_idx + i, config)
        for i, prod in enumerate(products)
    ]

    results = await asyncio.gather(*tasks)
    
    # ... code for processing results ...
```

### 3. Batch Processing

The system implements batch processing for efficiency:

```python
# From vector_db.py
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
    
    # ... code for updating vectors and mappings ...
```

### 4. Error Handling and Retry Logic

The system includes robust error handling and retry mechanisms:

```python
# From vector_db.py
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
```

## Docker Deployment

The system is containerized using Docker for easy deployment:

```yaml
# From docker-compose.yml
services:
  app:
    build:
      context: .
      dockerfile: ./docker/Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./uploads:/app/uploads
    environment:
      - MONGODB_URI=mongodb://mongodb:27017/product_db
      - CURRENT_MODE=local
      - LOCAL_BATCH_SIZE=16
      - LOCAL_MAX_WORKERS=4
      - CACHE_SIZE=150
    depends_on:
      - mongodb
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  mongodb:
    image: mongo:6.0
    # ... configuration for MongoDB ...

  triton:
    image: nvcr.io/nvidia/tritonserver:25.03-py3
    # ... configuration for Triton Inference Server ...
```

The Docker setup includes proper configuration for GPU access, volume mounts, and container networking.

## Model Configuration

The system includes configuration files for Triton models:

```
# From model_repository/internvl3_vision/config.pbtxt
name: "internvl3_vision"
platform: "tensorrt_plan"
max_batch_size: 16 # Or your preferred max batch

input [
  {
    # Must match the input name in your vision ONNX export
    name: "pixel_values" 
    data_type: TYPE_FP16 # Input usually FP16/FP32 from preprocessor
    # Shape EXCLUDING batch dim: (Channels, Height, Width)
    dims: [ 3, 448, 448 ] # Verify image size (448x448 for InternVL3)
  }
]

output [
  {
    # Must match the output name in your vision ONNX export
    name: "vision_output" # Or "vision_embeddings", "last_hidden_state" etc. 
    data_type: TYPE_FP16 # Output type, usually FP16/FP32
    # Shape EXCLUDING batch dim: (Num_Tokens, Hidden_Size)
    dims: [ 256, 1536 ] # Verify (256 tokens, 1536 hidden for InternVL3)
  }
]

instance_group [
  {
    count: 1
    kind: KIND_GPU
  }
]

dynamic_batching { } # Enable dynamic batching with default settings
```

## API Endpoints Documentation

### 1. POST /api/match

**Description**: Upload an image to find similar products.

**Request**:
- Content-Type: multipart/form-data
- Parameters:
  - `image`: File (required) - The product image to match
  - `name`: String (optional) - Name for the query product

**Response**:
- Content-Type: application/json
- Schema:
  ```json
  {
    "matches": [
      {
        "product": {
          "product_id": "integer",
          "name": "string",
          "image_url": "string",
          "metadata": {
            "gender": "string",
            "masterCategory": "string",
            "subCategory": "string",
            "articleType": "string",
            "baseColour": "string",
            "season": "string",
            "year": "integer",
            "usage": "string"
          }
        },
        "similarity_score": "float"
      }
    ],
    "query": {
      "name": "string"
    }
  }
  ```

**Example Request**:
```bash
curl -X POST "http://localhost:8000/api/match" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/product_image.jpg" \
  -F "name=Blue T-Shirt"
```

### 2. GET /health

**Description**: Check the health of the system components.

**Response**:
- Content-Type: application/json
- Schema:
  ```json
  {
    "status": "string",
    "embedding_mode": "string",
    "mongodb_status": "string",
    "mongodb_products": "string",
    "faiss_status": "string",
    "faiss_vectors": "integer",
    "triton_connection": "string"
  }
  ```

**Example Request**:
```bash
curl -X GET "http://localhost:8000/health"
```

### 3. GET /sync_check

**Description**: Verify that MongoDB and FAISS databases are in sync.

**Response**:
- Content-Type: application/json
- Schema:
  ```json
  {
    "databases_in_sync": "boolean"
  }
  ```

**Example Request**:
```bash
curl -X GET "http://localhost:8000/sync_check"
```

## Configuration Options

The system supports various configuration options through environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `EMBEDDING_MODE` | Embedding mode (`local` or `cloud`) | `local` |
| `LOCAL_BATCH_SIZE` | Batch size for local embedding generation | `16` |
| `LOCAL_MAX_WORKERS` | Maximum workers for local processing | `4` |
| `CLOUD_BATCH_SIZE` | Batch size for cloud embedding generation | `5` |
| `CLOUD_RATE_LIMIT_DELAY` | Delay between cloud API requests (seconds) | `60` |
| `CACHE_SIZE` | Size of the LRU cache | `150` |
| `MONGODB_URI` | MongoDB connection URI | `mongodb://mongodb:27017/product_db` |
| `TRITON_URL` | Triton Inference Server URL | `localhost:8000` |


## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   - **Symptom**: Error messages about CUDA out of memory
   - **Solution**: Reduce batch size in configuration or increase GPU memory

2. **MongoDB Connection Failures**
   - **Symptom**: Cannot connect to MongoDB
   - **Solution**: Check MongoDB is running and connection string is correct

3. **Triton Server Not Responding**
   - **Symptom**: Timeout errors when connecting to Triton
   - **Solution**: Check Triton logs, ensure models are loaded correctly

4. **Missing Model Files**
   - **Symptom**: Triton reports missing model files
   - **Solution**: Check model repository structure, ensure TensorRT engines are built

### Debugging Tips

1. Check logs in each container:
   ```bash
   docker logs <container_name>
   ```

2. Verify Triton model status:
   ```bash
   curl -X GET "http://localhost:8000/v2/models/internvl3_vision/config"
   ```

3. Test MongoDB connection:
   ```bash
   docker exec -it mongodb mongosh
   ```

## Security Considerations

The system implements several security best practices:

1. **Input Validation**: All API inputs are validated
2. **Safe File Handling**: Uploaded files are handled securely
3. **Error Handling**: Errors are logged without exposing sensitive information
4. **Docker Isolation**: Each component runs in an isolated container

## Future Enhancements

Potential areas for future development:

1. **Distributed Vector Database**: Shard FAISS index across multiple nodes
2. **Fine-tuning Pipeline**: Add capability to fine-tune models on domain-specific data
3. **Advanced Search Filters**: Combine vector search with metadata filtering
4. **Model A/B Testing**: Framework for comparing different embedding models
5. **User Feedback Loop**: Incorporate user feedback to improve matching

## Conclusion

The AI Product Matching System provides a robust, scalable solution for finding visually and semantically similar products. By combining state-of-the-art AI models with efficient vector search and metadata storage, it delivers accurate matching results with low latency.

---

## Appendix

### A. Model Export Process

The process for exporting models to TensorRT involves:

1. **ONNX Export**: Convert PyTorch models to ONNX format
2. **INT8 Calibration**: Generate calibration cache for INT8 quantization
3. **TensorRT Engine Building**: Build optimized TensorRT engines

### B. Embedding Format

The system uses combined embeddings with the following format:

- **Text Embedding**: 1024-dimensional vector
- **Vision Embedding**: 1024-dimensional vector
- **Combined Embedding**: 2048-dimensional vector (concatenation)

### C. Reference Documentation

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MongoDB Documentation](https://docs.mongodb.com/)
- [FAISS Documentation](https://github.com/facebookresearch/faiss/wiki)
- [Triton Inference Server Documentation](https://github.com/triton-inference-server/server)
- [TensorRT Documentation](https://docs.nvidia.com/deeplearning/tensorrt/)
