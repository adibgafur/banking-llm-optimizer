"""
Data preprocessing, validation, and exploratory data analysis utilities.
"""

import re
import logging
from typing import Dict, Any, List, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class TextPreprocessor:
    """Utilities for cleaning text and analyzing data distributions."""

    @staticmethod
    def clean_text(text: str) -> str:
        """Standardizes whitespaces and trims leading/trailing spaces."""
        if not isinstance(text, str):
            return ""
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @staticmethod
    def validate_query(query: str) -> str:
        """Ensures query is non-empty and well-formed."""
        cleaned = TextPreprocessor.clean_text(query)
        if not cleaned:
            raise ValueError("Query string cannot be empty or solely whitespace.")
        return cleaned

    @staticmethod
    def compute_class_distribution(df: pd.DataFrame, label_col: str = "category") -> pd.DataFrame:
        """Calculates frequency counts and percentage for each intent."""
        dist = df[label_col].value_counts().reset_index()
        dist.columns = ["intent", "count"]
        dist["percentage"] = (dist["count"] / len(df)) * 100.0
        return dist

    @staticmethod
    def compute_text_length_stats(df: pd.DataFrame, text_col: str = "text") -> Dict[str, Any]:
        """Calculates word-level and character-level summary statistics."""
        char_lens = df[text_col].astype(str).str.len()
        word_lens = df[text_col].astype(str).str.split().str.len()

        return {
            "num_samples": len(df),
            "char_mean": float(char_lens.mean()),
            "char_std": float(char_lens.std()),
            "char_min": int(char_lens.min()),
            "char_max": int(char_lens.max()),
            "word_mean": float(word_lens.mean()),
            "word_std": float(word_lens.std()),
            "word_min": int(word_lens.min()),
            "word_max": int(word_lens.max()),
            "word_p50": float(word_lens.median()),
            "word_p95": float(np.percentile(word_lens, 95)),
        }

    @staticmethod
    def plot_class_distribution(
        df: pd.DataFrame,
        output_path: str = "results/figures/class_distribution.png",
        label_col: str = "category"
    ):
        """Generates and saves class distribution bar chart using matplotlib."""
        counts = df[label_col].value_counts()
        plt.figure(figsize=(16, 6))
        plt.bar(range(len(counts)), counts.values, color="#1f77b4", edgecolor="none", width=0.8)
        plt.xticks(range(len(counts)), counts.index, rotation=90, fontsize=6)
        plt.title(f"Class Distribution Across {len(counts)} Intents (Total = {len(df)})", fontsize=14)
        plt.xlabel("Banking Intent", fontsize=10)
        plt.ylabel("Number of Samples", fontsize=10)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved class distribution plot to {output_path}")

    @staticmethod
    def plot_text_length_distribution(
        df: pd.DataFrame,
        output_path: str = "results/figures/text_length_distribution.png",
        text_col: str = "text"
    ):
        """Generates and saves token/word length histogram using matplotlib."""
        word_lens = df[text_col].astype(str).str.split().str.len()
        plt.figure(figsize=(10, 5))
        plt.hist(word_lens, bins=30, color="#2ca02c", edgecolor="black", alpha=0.7, density=False)
        plt.axvline(word_lens.mean(), color="red", linestyle="--", linewidth=1.5, label=f"Mean: {word_lens.mean():.1f}")
        plt.axvline(word_lens.median(), color="blue", linestyle=":", linewidth=1.5, label=f"Median: {word_lens.median():.1f}")
        plt.title("Distribution of Customer Query Word Counts", fontsize=14)
        plt.xlabel("Word Count", fontsize=11)
        plt.ylabel("Frequency", fontsize=11)
        plt.grid(axis="y", linestyle="--", alpha=0.5)
        plt.legend(fontsize=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved text length distribution plot to {output_path}")
