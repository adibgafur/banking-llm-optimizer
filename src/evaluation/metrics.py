"""
Classification evaluation metrics for BankingLLM-Optimizer.
Computes Accuracy, Macro/Weighted F1, Precision, Recall, and Per-Class breakdown.
"""

from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)


class ClassificationMetrics:
    """Computes comprehensive multi-class evaluation metrics."""

    @staticmethod
    def compute_all_metrics(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Calculates global summary classification metrics."""
        acc = float(accuracy_score(y_true, y_pred))
        macro_prec = float(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        macro_rec = float(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        macro_f1 = float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0))
        weighted_f1 = float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0))

        return {
            "accuracy": acc,
            "macro_precision": macro_prec,
            "macro_recall": macro_rec,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "total_samples": len(y_true)
        }

    @staticmethod
    def compute_per_class_metrics(
        y_true: List[str],
        y_pred: List[str],
        labels: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """Returns per-class precision, recall, f1, and support in a structured DataFrame."""
        report = classification_report(
            y_true,
            y_pred,
            labels=labels,
            output_dict=True,
            zero_division=0
        )
        rows = []
        for class_name, metrics in report.items():
            if class_name in ("accuracy", "macro avg", "weighted avg"):
                continue
            rows.append({
                "intent": class_name,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1_score": float(metrics["f1-score"]),
                "support": int(metrics["support"])
            })

        df = pd.DataFrame(rows)
        return df.sort_values(by="f1_score", ascending=True).reset_index(drop=True)
