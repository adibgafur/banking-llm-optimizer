"""
Embedding generation using Sentence-Transformers.
Supports batch encoding, L2 normalization, and persistent caching.
"""

import os
import logging
from typing import List, Optional, Union
import numpy as np
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """Generates normalized dense vector representations for text queries and demonstration pools."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        device: Optional[str] = None
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.device = device
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        self.model = SentenceTransformer(model_name, device=device)

    def encode(
        self,
        texts: Union[str, List[str]],
        batch_size: int = 64,
        normalize: bool = True,
        show_progress_bar: bool = False
    ) -> np.ndarray:
        """
        Encodes a single string or list of strings into float32 numpy embeddings.
        When normalize=True, vectors have unit L2 norm, making inner product equivalent to cosine similarity.
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )

        embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
        assert embeddings.shape[-1] == self.dimension, f"Expected dim {self.dimension}, got {embeddings.shape[-1]}"
        return embeddings

    def save_embeddings(self, embeddings: np.ndarray, file_path: str):
        """Saves embeddings array to a .npy file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        np.save(file_path, embeddings)
        logger.info(f"Saved {embeddings.shape} embeddings to {file_path}")

    def load_embeddings(self, file_path: str) -> np.ndarray:
        """Loads embeddings array from a .npy file."""
        embeddings = np.load(file_path)
        logger.info(f"Loaded {embeddings.shape} embeddings from {file_path}")
        return np.ascontiguousarray(embeddings, dtype=np.float32)
