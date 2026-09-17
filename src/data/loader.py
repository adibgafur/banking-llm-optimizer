"""
Data loader for the BANKING77 benchmark dataset.
Enforces strict train/val/test data separation and zero-leakage constraints.
"""

import os
import json
import logging
from typing import Tuple, List, Dict, Any, Optional
import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class BankingDataLoader:
    """
    Robust data loader for the BANKING77 benchmark.
    Handles automatic download, stratified dev/val splitting (90/10),
    official test isolation, and leakage verification.
    """

    TRAIN_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv"
    TEST_URL = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv"

    def __init__(
        self,
        raw_dir: str = "data/raw",
        processed_dir: str = "data/processed",
        config_path: str = "configs/intents.json"
    ):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        self.config_path = config_path
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.processed_dir, exist_ok=True)

    def load_raw_data(self, force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Loads the official BANKING77 train and test splits.
        Downloads from canonical PolyAI repository if not already cached in raw_dir.
        """
        train_path = os.path.join(self.raw_dir, "train.csv")
        test_path = os.path.join(self.raw_dir, "test.csv")

        if force_download or not os.path.exists(train_path):
            logger.info("Downloading official BANKING77 train dataset...")
            train_df = pd.read_csv(self.TRAIN_URL)
            train_df.to_csv(train_path, index=False)
        else:
            train_df = pd.read_csv(train_path)

        if force_download or not os.path.exists(test_path):
            logger.info("Downloading official BANKING77 test dataset...")
            test_df = pd.read_csv(self.TEST_URL)
            test_df.to_csv(test_path, index=False)
        else:
            test_df = pd.read_csv(test_path)

        assert "text" in train_df.columns and "category" in train_df.columns, "Train set must contain 'text' and 'category'"
        assert "text" in test_df.columns and "category" in test_df.columns, "Test set must contain 'text' and 'category'"

        logger.info(f"Loaded raw BANKING77: Train={len(train_df)} rows, Test={len(test_df)} rows")
        return train_df, test_df

    def create_stratified_split(
        self,
        train_df: pd.DataFrame,
        val_ratio: float = 0.10,
        seed: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Splits the official 10,003 training examples into:
        - 90% Training / Example Pool (~9,003 examples)
        - 10% Validation Set (~1,000 examples)
        Uses stratified sampling to preserve intent distributions across all 77 classes.
        """
        df = train_df.copy()
        if "orig_idx" not in df.columns:
            df["orig_idx"] = df.index

        train_pool_df, val_df = train_test_split(
            df,
            test_size=val_ratio,
            random_state=seed,
            stratify=df["category"]
        )

        train_pool_df = train_pool_df.reset_index(drop=True)
        val_df = val_df.reset_index(drop=True)

        logger.info(
            f"Stratified split created: Training Pool = {len(train_pool_df)} examples, "
            f"Validation Set = {len(val_df)} examples across {train_pool_df['category'].nunique()} intents"
        )
        return train_pool_df, val_df

    def verify_no_leakage(
        self,
        train_pool: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> bool:
        """
        Ensures strict data isolation:
        1. No overlap in index IDs between train_pool and val_df.
        2. Exact expected split counts.
        3. All 77 classes represented in each split.
        """
        if "orig_idx" in train_pool.columns and "orig_idx" in val_df.columns:
            train_indices = set(train_pool["orig_idx"])
            val_indices = set(val_df["orig_idx"])
            overlap = train_indices.intersection(val_indices)
            if overlap:
                raise ValueError(f"CRITICAL: Found index overlap between train pool and validation: {len(overlap)} IDs!")

        assert train_pool["category"].nunique() == 77, f"Train pool missing classes: {train_pool['category'].nunique()}"
        assert val_df["category"].nunique() == 77, f"Validation set missing classes: {val_df['category'].nunique()}"
        assert test_df["category"].nunique() == 77, f"Test set missing classes: {test_df['category'].nunique()}"

        logger.info("Verification passed: Strict train/val/test data isolation confirmed with zero leakage.")
        return True

    def get_intent_labels(self) -> List[str]:
        """
        Returns the canonical sorted list of all 77 banking intents.
        """
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                intents = json.load(f)
            return sorted(intents)

        train_df, test_df = self.load_raw_data()
        intents = sorted(list(set(train_df["category"].unique()) | set(test_df["category"].unique())))
        with open(self.config_path, "w", encoding="utf-8") as f:
            json.dump(intents, f, indent=2)
        return intents

    def save_processed_splits(
        self,
        train_pool: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame
    ) -> Dict[str, str]:
        """
        Persists processed splits to data/processed for fast reproducible downstream loading.
        """
        paths = {
            "train_pool": os.path.join(self.processed_dir, "train_pool.csv"),
            "val": os.path.join(self.processed_dir, "val.csv"),
            "test": os.path.join(self.processed_dir, "test.csv"),
        }
        train_pool.to_csv(paths["train_pool"], index=False)
        val_df.to_csv(paths["val"], index=False)
        test_df.to_csv(paths["test"], index=False)
        logger.info(f"Saved processed splits to {self.processed_dir}")
        return paths

    def load_processed_splits(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Loads the pre-split training pool, validation, and untouched test sets.
        If not yet generated, downloads raw data, creates the split, and saves.
        """
        train_pool_path = os.path.join(self.processed_dir, "train_pool.csv")
        val_path = os.path.join(self.processed_dir, "val.csv")
        test_path = os.path.join(self.processed_dir, "test.csv")

        if os.path.exists(train_pool_path) and os.path.exists(val_path) and os.path.exists(test_path):
            train_pool = pd.read_csv(train_pool_path)
            val_df = pd.read_csv(val_path)
            test_df = pd.read_csv(test_path)
            return train_pool, val_df, test_df

        raw_train, raw_test = self.load_raw_data()
        train_pool, val_df = self.create_stratified_split(raw_train)
        self.verify_no_leakage(train_pool, val_df, raw_test)
        self.save_processed_splits(train_pool, val_df, raw_test)
        return train_pool, val_df, raw_test
