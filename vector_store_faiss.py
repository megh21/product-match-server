from pathlib import Path
from typing import List, Dict, Optional
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
import numpy as np

class VectorStore:
    """Simplified vector store implementation using LangChain's FAISS wrapper"""
    
    def __init__(self, persist_dir: str = "data/vector_store"):
        self.persist_dir = Path(persist_dir)
        self.store: Optional[FAISS] = None
        self.dimension = 2048  # Combined embedding dimension

    def initialize(self, force_rebuild: bool = False):
        """Initialize or load existing vector store"""
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        if not force_rebuild and self._exists():
            self._load()
        else:
            self.store = FAISS.from_texts(
                texts=[],  # Empty initial store
                embedding=None,  # We'll use our own embeddings
                metadatas=[]
            )

    def _exists(self) -> bool:
        """Check if vector store exists"""
        return (self.persist_dir / "index.faiss").exists()

    def _load(self):
        """Load existing vector store"""
        self.store = FAISS.load_local(
            self.persist_dir, 
            None  # No embedding function needed
        )

    def add_embeddings(self, embeddings: List[np.ndarray], documents: List[dict]):
        """Add embeddings and metadata to store"""
        if not self.store:
            self.initialize()

        docs = []
        for emb, doc in zip(embeddings, documents):
            # Create document with metadata
            docs.append(Document(
                page_content=doc["name"],
                metadata={
                    "product_id": doc["product_id"],
                    **doc["metadata"],
                    "image_path": doc["image_path"],
                    "image_url": doc["image_url"]
                }
            ))
            # Add embedding
            self.store.add_embeddings(
                text_embeddings=[(str(doc["product_id"]), emb)],
                metadatas=[docs[-1].metadata]
            )

        # Save after batch
        self.persist()

    def search(self, query_embedding: np.ndarray, k: int = 5) -> List[dict]:
        """Search for similar vectors"""
        if not self.store:
            raise RuntimeError("Vector store not initialized")

        results = self.store.similarity_search_with_score_by_vector(
            query_embedding,
            k=k
        )

        return [{
            "product_id": doc.metadata["product_id"],
            "score": float(score),
            "metadata": doc.metadata
        } for doc, score in results]

    def persist(self):
        """Save vector store to disk"""
        if self.store:
            self.store.save_local(self.persist_dir)