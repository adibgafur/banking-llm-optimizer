"""
Example selector for dynamic few-shot prompting.
Provides similarity-only retrieval and diversity-aware retrieval from the training pool.
"""

import logging
from typing import List, Dict, Any, Optional
import numpy as np

from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.faiss_index import FaissIndexManager

logger = logging.getLogger(__name__)


class ExampleSelector:
    """Selects dynamic few-shot demonstrations via similarity or diversity-aware search."""

    def __init__(
        self,
        embedding_generator: EmbeddingGenerator,
        index_manager: FaissIndexManager
    ):
        self.encoder = embedding_generator
        self.index_manager = index_manager

    def retrieve_similar(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """
        Similarity-only retrieval: Encodes query and retrieves top-k nearest neighbors.
        """
        query_vec = self.encoder.encode(query, normalize=True)
        return self.index_manager.search(query_vec, top_k=k)

    def retrieve_diverse(
        self,
        query: str,
        k: int = 5,
        candidate_pool_size: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Diversity-aware retrieval:
        1. Retrieves candidate pool of top M (e.g. M=20) most similar examples.
        2. Applies deterministic diversity selection:
           - Always takes the #1 most similar candidate.
           - Greedily selects subsequent candidates prioritizing distinct intents
             to expose the LLM to boundary distinctions rather than redundant duplicates.
           - If needed, fills remaining slots with next highest similarity candidates.
        """
        candidate_pool_size = max(candidate_pool_size, k)
        query_vec = self.encoder.encode(query, normalize=True)
        candidates = self.index_manager.search(query_vec, top_k=candidate_pool_size)

        if not candidates:
            return []

        selected: List[Dict[str, Any]] = [candidates[0]]
        selected_intents = {candidates[0]["category"]}

        # First pass: Add top candidates with novel intents to promote boundary contrast
        for cand in candidates[1:]:
            if len(selected) >= k:
                break
            if cand["category"] not in selected_intents:
                selected.append(cand)
                selected_intents.add(cand["category"])

        # Second pass: If slots remain, fill with next best similarity
        if len(selected) < k:
            selected_ids = {s.get("pool_index") for s in selected}
            for cand in candidates[1:]:
                if len(selected) >= k:
                    break
                if cand.get("pool_index") not in selected_ids:
                    selected.append(cand)
                    selected_ids.add(cand.get("pool_index"))

        return selected[:k]

    @staticmethod
    def format_demonstrations(examples: List[Dict[str, Any]]) -> str:
        """
        Formats retrieved examples into standard prompt exemplar text.
        """
        if not examples:
            return "None available."

        formatted_blocks = []
        for i, ex in enumerate(examples, 1):
            text = ex.get("text", "").strip()
            category = ex.get("category", "").strip()
            formatted_blocks.append(f"Example {i}:\nCustomer: \"{text}\"\nIntent: {category}")

        return "\n\n".join(formatted_blocks)
