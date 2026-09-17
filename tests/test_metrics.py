"""
Unit tests for evaluation metrics, confusion matrix, error taxonomy, and cost analysis.
"""

import pytest
import pandas as pd
from src.evaluation.metrics import ClassificationMetrics
from src.evaluation.confusion import ConfusionAnalyzer
from src.evaluation.cost_analysis import CostAnalyzer
from src.evaluation.error_analysis import ErrorAnalyzer
from src.pipeline import BankingIntentPipeline


def test_classification_metrics():
    y_true = ["card_arrival", "card_arrival", "pin_blocked", "transfer_fee_charged"]
    y_pred = ["card_arrival", "pin_blocked", "pin_blocked", "transfer_fee_charged"]

    metrics = ClassificationMetrics.compute_all_metrics(y_true, y_pred)
    assert metrics["accuracy"] == 0.75
    assert "macro_f1" in metrics
    assert "weighted_f1" in metrics
    assert metrics["total_samples"] == 4

    per_class = ClassificationMetrics.compute_per_class_metrics(y_true, y_pred)
    assert isinstance(per_class, pd.DataFrame)
    assert len(per_class) == 3


def test_confusion_top_pairs():
    y_true = ["card_arrival", "card_arrival", "card_arrival", "pin_blocked"]
    y_pred = ["card_delivery_estimate", "card_delivery_estimate", "card_arrival", "pin_blocked"]

    top_pairs = ConfusionAnalyzer.extract_top_confusion_pairs(y_true, y_pred, top_n=5)
    assert len(top_pairs) == 1
    assert top_pairs.iloc[0]["true_intent"] == "card_arrival"
    assert top_pairs.iloc[0]["predicted_intent"] == "card_delivery_estimate"
    assert top_pairs.iloc[0]["count"] == 2


def test_cost_analyzer():
    analyzer = CostAnalyzer(input_price_per_million=0.20, output_price_per_million=0.20)
    # 1,000,000 input tokens + 1,000,000 output tokens = $0.40
    single_cost = analyzer.calculate_query_cost(1_000_000, 1_000_000)
    assert pytest.approx(single_cost, 0.0001) == 0.40

    summary = analyzer.summarize_benchmark_run(
        latencies_ms=[100.0, 200.0, 300.0],
        input_tokens=[1000, 1000, 1000],
        output_tokens=[100, 100, 100]
    )
    assert summary["num_queries"] == 3
    assert summary["latency_mean_ms"] == 200.0
    assert summary["latency_median_ms"] == 200.0
    assert summary["avg_input_tokens"] == 1000.0
    assert summary["avg_output_tokens"] == 100.0


def test_error_categorization():
    # Semantic similarity
    err_sem = ErrorAnalyzer.categorize_error(
        query="I want to track my card delivery status",
        true_intent="card_delivery_estimate",
        predicted_intent="card_arrival"
    )
    assert err_sem == "semantic_similarity"

    # Ambiguous query
    err_amb = ErrorAnalyzer.categorize_error(
        query="help please",
        true_intent="card_arrival",
        predicted_intent="pin_blocked"
    )
    assert err_amb == "ambiguous_query"

    # Format error
    err_fmt = ErrorAnalyzer.categorize_error(
        query="My card is missing",
        true_intent="lost_or_stolen_card",
        predicted_intent="unknown",
        parsing_error="Malformed JSON"
    )
    assert err_fmt == "format_error"


def test_confidence_threshold_routing():
    df = pd.DataFrame([
        {"true_intent": "card_arrival", "predicted_intent": "card_arrival", "confidence": 0.95},
        {"true_intent": "card_arrival", "predicted_intent": "card_arrival", "confidence": 0.85},
        {"true_intent": "pin_blocked", "predicted_intent": "card_arrival", "confidence": 0.65},
        {"true_intent": "pin_blocked", "predicted_intent": "pin_blocked", "confidence": 0.50},
    ])
    res = BankingIntentPipeline.analyze_confidence_thresholds(df, thresholds=[0.60, 0.70, 0.80, 0.90])
    assert len(res) == 4
    # At threshold 0.80: samples with conf >= 0.80 are 2 (both correct -> acc=1.0, coverage=0.5)
    row_80 = res[res["threshold"] == 0.80].iloc[0]
    assert row_80["coverage"] == 0.5
    assert row_80["accepted_accuracy"] == 1.0
