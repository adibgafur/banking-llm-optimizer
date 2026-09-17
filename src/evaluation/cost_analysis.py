"""
Cost, token usage, and latency profiling for BankingLLM-Optimizer.
Provides granular expense estimates based on configurable pricing tables.
"""

from typing import List, Dict, Any, Optional
import numpy as np
import pandas as pd


class CostAnalyzer:
    """Calculates latency distributions, token consumption, and financial cost estimates."""

    def __init__(self, input_price_per_million: float = 0.20, output_price_per_million: float = 0.20):
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million

    def calculate_query_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Computes estimated USD cost for an individual API call."""
        cost_in = (input_tokens / 1_000_000.0) * self.input_price_per_million
        cost_out = (output_tokens / 1_000_000.0) * self.output_price_per_million
        return cost_in + cost_out

    def summarize_benchmark_run(
        self,
        latencies_ms: List[float],
        input_tokens: List[int],
        output_tokens: List[int]
    ) -> Dict[str, Any]:
        """
        Calculates aggregate statistics over a full evaluation run:
        - Latency: mean, median (P50), P95
        - Tokens: mean input, mean output, mean total
        - Cost: total cost, avg cost per query, cost per 1,000 queries
        """
        n = len(latencies_ms)
        if n == 0:
            return {
                "num_queries": 0,
                "latency_mean_ms": 0.0,
                "latency_median_ms": 0.0,
                "latency_p95_ms": 0.0,
                "avg_input_tokens": 0.0,
                "avg_output_tokens": 0.0,
                "avg_total_tokens": 0.0,
                "total_estimated_cost_usd": 0.0,
                "cost_per_query_usd": 0.0,
                "cost_per_1k_queries_usd": 0.0,
            }

        arr_lat = np.array(latencies_ms, dtype=np.float64)
        arr_in = np.array(input_tokens, dtype=np.float64)
        arr_out = np.array(output_tokens, dtype=np.float64)
        arr_tot = arr_in + arr_out

        total_cost = float(
            (np.sum(arr_in) / 1_000_000.0) * self.input_price_per_million
            + (np.sum(arr_out) / 1_000_000.0) * self.output_price_per_million
        )

        cost_per_query = total_cost / n
        cost_per_1k = cost_per_query * 1000.0

        return {
            "num_queries": n,
            "latency_mean_ms": float(np.mean(arr_lat)),
            "latency_median_ms": float(np.median(arr_lat)),
            "latency_p95_ms": float(np.percentile(arr_lat, 95)),
            "avg_input_tokens": float(np.mean(arr_in)),
            "avg_output_tokens": float(np.mean(arr_out)),
            "avg_total_tokens": float(np.mean(arr_tot)),
            "total_estimated_cost_usd": total_cost,
            "cost_per_query_usd": cost_per_query,
            "cost_per_1k_queries_usd": cost_per_1k,
        }
