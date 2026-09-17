"""
Unit tests for semantic embeddings, FAISS indexing, and demonstration selection.
"""

import os
import pytest
import numpy as np
from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.faiss_index import FaissIndexManager
from src.retrieval.example_selector import ExampleSelector


@pytest.fixture(scope="module")
def encoder():
    return EmbeddingGenerator()


@pytest.fixture
def mini_index_and_metadata(encoder):
    metadata = [
        {"text": "My new debit card has not arrived in the post yet.", "category": "card_arrival"},
        {"text": "How can I track the delivery status of my card?", "category": "card_delivery_estimate"},
        {"text": "I forgot my card pin code number at the ATM.", "category": "pin_blocked"},
        {"text": "The transfer has not reached the recipient bank account.", "category": "transfer_not_received_by_recipient"},
        {"text": "Why was I charged an extra fee on my recent transfer?", "category": "transfer_fee_charged"},
    ]
    texts = [m["text"] for m in metadata]
    embeddings = encoder.encode(texts, normalize=True)
    index_mgr = FaissIndexManager(dimension=384)
    index_mgr.build_index(embeddings, metadata)
    return index_mgr, metadata


def test_embedding_generator_shape(encoder):
    emb = encoder.encode("Test query for banking")
    assert isinstance(emb, np.ndarray)
    assert emb.shape == (1, 384)
    # Check L2 norm is approximately 1.0
    norm = np.linalg.norm(emb[0])
    assert pytest.approx(norm, 0.01) == 1.0


def test_faiss_search(encoder, mini_index_and_metadata):
    index_mgr, _ = mini_index_and_metadata
    query_vec = encoder.encode("Where is my card? Still waiting", normalize=True)
    results = index_mgr.search(query_vec, top_k=2)

    assert len(results) == 2
    assert "text" in results[0]
    assert "category" in results[0]
    assert "similarity" in results[0]
    # The top result should be card related
    assert results[0]["category"] in ("card_arrival", "card_delivery_estimate")


def test_example_selector_similarity_and_diversity(encoder, mini_index_and_metadata):
    index_mgr, _ = mini_index_and_metadata
    selector = ExampleSelector(encoder, index_mgr)

    # Similarity retrieval
    sim_res = selector.retrieve_similar("Where is my card?", k=2)
    assert len(sim_res) == 2

    # Diversity retrieval
    div_res = selector.retrieve_diverse("Where is my card?", k=3, candidate_pool_size=5)
    assert len(div_res) == 3
    # Check diverse intents were prioritized
    distinct_intents = {r["category"] for r in div_res}
    assert len(distinct_intents) >= 2


def test_format_demonstrations(encoder, mini_index_and_metadata):
    index_mgr, _ = mini_index_and_metadata
    selector = ExampleSelector(encoder, index_mgr)
    results = selector.retrieve_similar("Where is my card?", k=2)
    demo_str = selector.format_demonstrations(results)

    assert "Example 1:" in demo_str
    assert "Customer:" in demo_str
    assert "Intent:" in demo_str
