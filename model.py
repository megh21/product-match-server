from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class ProductMatch(BaseModel):
    product: Dict[str, Any]
    similarity_score: float


class MatchResponse(BaseModel):
    matches: List[ProductMatch]
    query: Dict[str, str]


class HealthStatus(BaseModel):
    status: str
    embedding_mode: str
    mongodb_status: Optional[str] = None
    mongodb_products: Optional[str] = None
    faiss_status: Optional[str] = None
    faiss_vectors: Optional[int] = None
    triton_connection: Optional[str] = None


class SyncCheckResponse(BaseModel):
    databases_in_sync: bool
