"""
Confusion matrix computation, visualization, and error-pair extraction.
"""

import os
import logging
from typing import List, Tuple, Optional
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)


class ConfusionAnalyzer:
    """Analyzes and plots intent confusion across the 77 classes."""

    @staticmethod
    def extract_top_confusion_pairs(
        y_true: List[str],
        y_pred: List[str],
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Finds pairs where True Intent != Predicted Intent,
        sorted in descending order of frequency.
        """
        confusions = {}
        for yt, yp in zip(y_true, y_pred):
            if yt != yp:
                pair = (yt, yp)
                confusions[pair] = confusions.get(pair, 0) + 1

        rows = [
            {"true_intent": pair[0], "predicted_intent": pair[1], "count": cnt}
            for pair, cnt in confusions.items()
        ]

        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(by="count", ascending=False).reset_index(drop=True)
        else:
            df = pd.DataFrame(columns=["true_intent", "predicted_intent", "count"])

        return df.head(top_n)

    @staticmethod
    def plot_confusion_matrix(
        y_true: List[str],
        y_pred: List[str],
        labels: List[str],
        output_path: str = "results/figures/confusion_matrix.png"
    ):
        """Plots high-resolution 77x77 confusion matrix heatmap using matplotlib."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        plt.figure(figsize=(24, 22))
        plt.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.title(f"Confusion Matrix ({len(labels)} Banking Intents)", fontsize=18, pad=20)
        plt.colorbar(fraction=0.046, pad=0.04)

        tick_marks = np.arange(len(labels))
        plt.xticks(tick_marks, labels, rotation=90, fontsize=6)
        plt.yticks(tick_marks, labels, fontsize=6)

        plt.xlabel("Predicted Intent", fontsize=14, labelpad=10)
        plt.ylabel("True Intent", fontsize=14, labelpad=10)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300)
        plt.close()
        logger.info(f"Saved confusion matrix plot to {output_path}")
