# API Usage Guidelines

This document provides guidelines and examples for using the AI Product Matching System API.

## Base URL

```
http://localhost:8000
```

For production deployments, replace with your domain.

## Authentication

*Note: The current version does not implement authentication. For production use, implement an authentication mechanism such as API keys or OAuth.*

## API Endpoints

### 1. Product Matching

**Endpoint**: `/api/match`

**Method**: POST

**Description**: Upload an image to find similar products in the database.

#### Request

**Content-Type**: `multipart/form-data`

**Parameters**:

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| image | File | Yes | The product image to match against the database |
| name | String | No | A name for the query product (default: "Query Product") |

**Example Request using cURL**:

```bash
curl -X POST "http://localhost:8000/api/match" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "image=@/path/to/product_image.jpg" \
  -F "name=Blue T-Shirt"
```

**Example Request using Python**:

```python
import requests

url = "http://localhost:8000/api/match"
files = {
    "image": ("product.jpg", open("/path/to/product_image.jpg", "rb"), "image/jpeg")
}
data = {
    "name": "Blue T-Shirt"
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

#### Response

**Content-Type**: `application/json`

**Response Body**:

```json
{
  "matches": [
    {
      "product": {
        "product_id": 12345,
        "name": "Men's Blue Cotton T-Shirt",
        "image_filename": "12345.jpg",
        "image_url": "http://example.com/images/12345.jpg",
        "metadata": {
          "gender": "Men",
          "masterCategory": "Apparel",
          "subCategory": "Topwear",
          "articleType": "Tshirts",
          "baseColour": "Blue",
          "season": "Summer",
          "year": 2023,
          "usage": "Casual"
        }
      },
      "similarity_score": 0.89
    },
    {
      "product": {
        "product_id": 12346,
        "name": "Men's Navy Blue Cotton T-Shirt",
        "image_filename": "12346.jpg",
        "image_url": "http://example.com/images/12346.jpg",
        "metadata": {
          "gender": "Men",
          "masterCategory": "Apparel",
          "subCategory": "Topwear",
          "articleType": "Tshirts",
          "baseColour": "Navy Blue",
          "season": "Summer",
          "year": 2023,
          "usage": "Casual"
        }
      },
      "similarity_score": 0.85
    }
  ],
  "query": {
    "name": "Blue T-Shirt"
  }
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| matches | Array | List of matching products |
| matches[].product | Object | Product information |
| matches[].product.product_id | Integer | Unique product identifier |
| matches[].product.name | String | Product name |
| matches[].product.image_filename | String | Image filename |
| matches[].product.image_url | String | URL to product image |
| matches[].product.metadata | Object | Product metadata |
| matches[].similarity_score | Float | Similarity score between 0 and 1 |
| query | Object | Information about the query |
| query.name | String | Name provided for the query product |

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad Request (e.g., invalid image format) |
| 429 | Too Many Requests (rate limit exceeded) |
| 500 | Internal Server Error |

#### Error Response

```json
{
  "detail": "Error message describing the issue"
}
```

### 2. Health Check

**Endpoint**: `/health`

**Method**: GET

**Description**: Check the health status of system components.

#### Request

**Example Request using cURL**:

```bash
curl -X GET "http://localhost:8000/health"
```

**Example Request using Python**:

```python
import requests

url = "http://localhost:8000/health"
response = requests.get(url)
print(response.json())
```

#### Response

**Content-Type**: `application/json`

**Response Body**:

```json
{
  "status": "ok",
  "embedding_mode": "local",
  "mongodb_status": "connected",
  "mongodb_products": "5000",
  "faiss_status": "loaded",
  "faiss_vectors": 5000,
  "triton_connection": "live"
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| status | String | Overall system status ("ok" or "error") |
| embedding_mode | String | Current embedding mode ("local" or "cloud") |
| mongodb_status | String | MongoDB connection status |
| mongodb_products | String | Number of products in MongoDB |
| faiss_status | String | FAISS index status |
| faiss_vectors | Integer | Number of vectors in FAISS index |
| triton_connection | String | Triton server connection status (if applicable) |

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 500 | Internal Server Error |

### 3. Database Synchronization Check

**Endpoint**: `/sync_check`

**Method**: GET

**Description**: Verify that MongoDB and FAISS databases are in sync.

#### Request

**Example Request using cURL**:

```bash
curl -X GET "http://localhost:8000/sync_check"
```

**Example Request using Python**:

```python
import requests

url = "http://localhost:8000/sync_check"
response = requests.get(url)
print(response.json())
```

#### Response

**Content-Type**: `application/json`

**Response Body**:

```json
{
  "databases_in_sync": true
}
```

#### Response Fields

| Field | Type | Description |
|-------|------|-------------|
| databases_in_sync | Boolean | Whether MongoDB and FAISS are in sync |

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 500 | Internal Server Error |

## Best Practices

### Image Requirements

- **Supported formats**: JPEG, PNG, WebP
- **Maximum size**: 10MB
- **Recommended dimensions**: At least 224x224 pixels

### Rate Limiting

The API implements rate limiting to prevent abuse. Current limits:

- **Product matching**: 10 requests per minute
- **Health check**: 60 requests per minute
- **Sync check**: 60 requests per minute

If rate limits are exceeded, the API will return a 429 status code.

### Performance Considerations

- The first request may be slower due to model loading
- Larger images take longer to process
- Consider compressing images before sending for faster processing

### Error Handling

Always check for error responses and handle them appropriately. Common errors include:

- **400 Bad Request**: Invalid parameters or image format
- **429 Too Many Requests**: Rate limit exceeded
- **500 Internal Server Error**: Server-side error

### Example Integration (JavaScript)

```javascript
async function findSimilarProducts(imageFile, productName) {
  try {
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('name', productName);
    
    const response = await fetch('http://localhost:8000/api/match', {
      method: 'POST',
      body: formData,
    });
    
    if (!response.ok) {
      if (response.status === 429) {
        console.error('Rate limit exceeded. Please try again later.');
        return;
      }
      
      const errorData = await response.json();
      console.error('Error:', errorData.detail);
      return;
    }
    
    const data = await response.json();
    return data.matches;
  } catch (error) {
    console.error('Error:', error);
  }
}

// Usage
document.getElementById('product-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const imageFile = document.getElementById('product-image').files[0];
  const productName = document.getElementById('product-name').value;
  
  const matches = await findSimilarProducts(imageFile, productName);
  
  // Display matches in UI
  if (matches) {
    const matchesContainer = document.getElementById('matches-container');
    matchesContainer.innerHTML = matches.map(match => `
      <div class="product-card">
        <img src="${match.product.image_url}" alt="${match.product.name}">
        <h3>${match.product.name}</h3>
        <p>Similarity: ${(match.similarity_score * 100).toFixed(1)}%</p>
      </div>
    `).join('');
  }
});
```

## Troubleshooting

### Common Issues

1. **Image upload fails**
   - Ensure the image is in a supported format (JPEG, PNG, WebP)
   - Check that the image size is under the 10MB limit
   - Verify the Content-Type is set to `multipart/form-data`

2. **No matches found**
   - Ensure the database has been populated with product data
   - Try a clearer image with better lighting
   - Check if the product category exists in the database

3. **Slow response times**
   - First request may be slow due to model loading
   - Consider using smaller images for faster processing
   - Check server load and resource usage

### Support

For further assistance, please contact the system administrator or maintainer.

---

*This document is subject to updates. Last updated: April 2025*