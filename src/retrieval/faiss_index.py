"""
FAISS Index Manager for semantic nearest-neighbor retrieval.
Uses IndexFlatIP with normalized vectors for exact cosine similarity search.
"""

import os
import json
import logging
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
import faiss

logger = logging.getLogger(__name__)


class FaissIndexManager:
    """Manages creation, serialization, and querying of FAISS vector indices."""

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: List[Dict[str, Any]] = []

    def build_index(self, embeddings: np.ndarray, metadata: List[Dict[str, Any]]):
        """Builds index from dense float32 embeddings and associated metadata dicts."""
        assert embeddings.shape[0] == len(metadata), f"Mismatch: {embeddings.shape[0]} vectors vs {len(metadata)} metadata items"
        assert embeddings.shape[1] == self.dimension, f"Vector dimension mismatch: expected {self.dimension}, got {embeddings.shape[1]}"

        # Ensure float32 and contiguous
        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        self.index.reset()
        self.index.add(embeddings)
        self.metadata = list(metadata)
        logger.info(f"Built FAISS IndexFlatIP with {self.index.ntotal} vectors of dimension {self.dimension}")

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Queries the FAISS index for top_k nearest neighbors.
        Returns a list of dicts with match text, category, similarity score, and index.
        """
        if self.index.ntotal == 0:
            raise ValueError("FAISS index is empty. Call build_index or load first.")

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)

        query_embedding = np.ascontiguousarray(query_embedding, dtype=np.float32)
        top_k = min(top_k, self.index.ntotal)

        distances, indices = self.index.search(query_embedding, top_k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.metadata):
                item = dict(self.metadata[idx])
                item["similarity"] = float(dist)
                item["pool_index"] = int(idx)
                results.append(item)

        return results

    def save(self, index_path: str, metadata_path: Optional[str] = None):
        """Persists the FAISS index and metadata JSON to disk."""
        os.makedirs(os.path.dirname(index_path), exist_ok=True)
        faiss.write_index(self.index, index_path)

        if metadata_path is None:
            metadata_path = os.path.splitext(index_path)[0] + "_meta.json"

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

        logger.info(f"Saved FAISS index ({self.index.ntotal} items) to {index_path} and {metadata_path}")

    def load(self, index_path: str, metadata_path: Optional[str] = None):
        """Loads a persisted FAISS index and metadata from disk."""
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"FAISS index file not found at {index_path}")

        self.index = faiss.read_index(index_path)

        if metadata_path is None:
            metadata_path = os.path.splitext(index_path)[0] + "_meta.json"

        if os.path.exists(metadata_path):
            with open(metadata_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = []

        logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors and {len(self.metadata)} metadata records.")
