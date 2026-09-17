"""
Error analysis and failure taxonomy classification.
Categorizes failure modes into semantic, retrieval, ambiguity, and formatting errors.
"""

import os
import logging
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)

ERROR_TYPES = [
    "semantic_similarity",
    "ambiguous_query",
    "missing_context",
    "retrieval_error",
    "classification_error",
    "format_error",
    "other"
]


class ErrorAnalyzer:
    """Categorizes misclassifications into an empirical taxonomy."""

    @staticmethod
    def categorize_error(
        query: str,
        true_intent: str,
        predicted_intent: str,
        retrieved_examples: Optional[List[Dict[str, Any]]] = None,
        parsing_error: Optional[str] = None
    ) -> str:
        """
        Applies deterministic heuristic rules to classify failure modes:
        1. format_error: parsing failed or output invalid
        2. retrieval_error: retrieved examples misled the model towards wrong intent
        3. semantic_similarity: predicted intent shares stem/domain with true intent
        4. ambiguous_query: query is ultra-short (<4 words)
        5. missing_context: contains pronouns or vague references without entity
        6. classification_error: default misclassification
        """
        if parsing_error or predicted_intent == "unknown":
            return "format_error"

        # Check if retrieval introduced distractor
        if retrieved_examples:
            retrieved_categories = [ex.get("category") for ex in retrieved_examples]
            if predicted_intent in retrieved_categories and true_intent not in retrieved_categories:
                return "retrieval_error"

        # Check semantic similarity / lexical root overlap
        true_tokens = set(true_intent.split("_"))
        pred_tokens = set(predicted_intent.split("_"))
        shared = true_tokens.intersection(pred_tokens)
        if len(shared) >= 1:
            return "semantic_similarity"

        words = query.strip().split()
        if len(words) <= 3:
            return "ambiguous_query"

        vague_phrases = ["it", "this", "that", "what happened", "not working", "why"]
        if any(vp == query.lower().strip() for vp in vague_phrases):
            return "missing_context"

        return "classification_error"

    @classmethod
    def build_error_dataframe(
        cls,
        queries: List[str],
        true_intents: List[str],
        predicted_intents: List[str],
        confidences: List[float],
        strategy: str,
        retrieved_examples_list: Optional[List[List[Dict[str, Any]]]] = None,
        parsing_errors: Optional[List[Optional[str]]] = None
    ) -> pd.DataFrame:
        """Constructs and returns error analysis DataFrame for misclassified samples."""
        records = []
        n = len(queries)
        for i in range(n):
            yt = true_intents[i]
            yp = predicted_intents[i]
            if yt != yp:
                q = queries[i]
                conf = confidences[i]
                ret_ex = retrieved_examples_list[i] if retrieved_examples_list else None
                err_msg = parsing_errors[i] if parsing_errors else None

                err_type = cls.categorize_error(
                    query=q,
                    true_intent=yt,
                    predicted_intent=yp,
                    retrieved_examples=ret_ex,
                    parsing_error=err_msg
                )

                ex_summary = ""
                if ret_ex:
                    ex_summary = "; ".join([f"{e.get('category')}:{e.get('text')[:30]}" for e in ret_ex[:3]])

                records.append({
                    "query": q,
                    "true_intent": yt,
                    "predicted_intent": yp,
                    "confidence": conf,
                    "prompt_strategy": strategy,
                    "retrieved_examples": ex_summary,
                    "error_type": err_type
                })

        df = pd.DataFrame(records)
        if df.empty:
            df = pd.DataFrame(columns=[
                "query", "true_intent", "predicted_intent", "confidence",
                "prompt_strategy", "retrieved_examples", "error_type"
            ])
        return df

    @staticmethod
    def save_error_analysis(df: pd.DataFrame, output_path: str = "results/error_analysis/error_analysis.csv"):
        """Saves error analysis table to CSV."""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved error analysis ({len(df)} records) to {output_path}")
