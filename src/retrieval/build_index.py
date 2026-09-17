"""
Script to precompute embeddings and build the FAISS index for the training pool.
"""

import os
import logging
import pandas as pd
from src.config import AppConfig
from src.data.loader import BankingDataLoader
from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.faiss_index import FaissIndexManager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def build_and_save_index():
    config = AppConfig()
    loader = BankingDataLoader()
    train_pool, val_df, test_df = loader.load_processed_splits()

    embeddings_path = config.raw_config.get("retrieval", {}).get(
        "embeddings_path", "data/processed/train_pool_embeddings.npy"
    )
    index_path = config.raw_config.get("retrieval", {}).get(
        "index_path", "data/processed/faiss_train_pool.index"
    )

    encoder = EmbeddingGenerator(
        model_name=config.embedding_model_name,
        dimension=config.embedding_dimension
    )

    if os.path.exists(embeddings_path):
        logger.info(f"Loading existing embeddings from {embeddings_path}")
        embeddings = encoder.load_embeddings(embeddings_path)
    else:
        logger.info(f"Generating embeddings for {len(train_pool)} training pool examples...")
        embeddings = encoder.encode(train_pool["text"].tolist(), batch_size=128, show_progress_bar=True)
        encoder.save_embeddings(embeddings, embeddings_path)

    metadata = train_pool[["text", "category"]].to_dict(orient="records")
    index_mgr = FaissIndexManager(dimension=config.embedding_dimension)
    index_mgr.build_index(embeddings, metadata)
    index_mgr.save(index_path)
    logger.info("FAISS index successfully built and saved!")


if __name__ == "__main__":
    build_and_save_index()
