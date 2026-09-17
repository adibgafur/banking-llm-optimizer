"""
End-to-End Banking Intent Classification Pipeline.
Integrates retrieval, prompt construction, LLM generation, structured parsing,
confidence-based routing, and cost estimation.
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
import pandas as pd
from tqdm import tqdm

from src.config import AppConfig
from src.retrieval.embeddings import EmbeddingGenerator
from src.retrieval.faiss_index import FaissIndexManager
from src.retrieval.example_selector import ExampleSelector
from src.prompts.templates import PromptBuilder
from src.llm.groq_client import GroqClient
from src.llm.response_parser import ResponseParser, BankingPrediction
from src.llm.response_generator import BankingResponseGenerator
from src.utils.caching import LLMCache
from src.evaluation.cost_analysis import CostAnalyzer

logger = logging.getLogger(__name__)


class BankingIntentPipeline:
    """Master pipeline orchestrating intent classification across multiple strategies."""

    def __init__(
        self,
        config: Optional[AppConfig] = None,
        mock_mode: bool = False,
        embedding_generator: Optional[EmbeddingGenerator] = None,
        index_manager: Optional[FaissIndexManager] = None
    ):
        self.config = config or AppConfig()
        self.mock_mode = mock_mode

        # Initialize Caching
        self.cache = LLMCache(
            db_path=self.config.cache_db_path,
            enabled=self.config.cache_enabled
        )

        # Initialize Groq Client
        self.groq_client = GroqClient(
            api_key=self.config.groq_api_key,
            cache=self.cache,
            mock_mode=self.mock_mode
        )
        self.response_generator = BankingResponseGenerator(self.groq_client)

        # Load Intent Labels & Response Parser
        intents_path = os.path.join("configs", "intents.json")
        if os.path.exists(intents_path):
            import json
            with open(intents_path, "r", encoding="utf-8") as f:
                self.valid_intents = json.load(f)
        else:
            self.valid_intents = []
        self.parser = ResponseParser(self.valid_intents)
        self.prompt_builder = PromptBuilder(self.valid_intents)

        # Initialize Retrieval Layer
        self.encoder = embedding_generator
        self.index_manager = index_manager
        self.selector: Optional[ExampleSelector] = None

        if self.encoder is None:
            self.encoder = EmbeddingGenerator(
                model_name=self.config.embedding_model_name,
                dimension=self.config.embedding_dimension
            )

        if self.index_manager is None:
            self.index_manager = FaissIndexManager(dimension=self.config.embedding_dimension)
            index_path = self.config.raw_config.get("retrieval", {}).get("index_path", "data/processed/faiss_train_pool.index")
            if os.path.exists(index_path):
                self.index_manager.load(index_path)

        if self.index_manager.index.ntotal > 0:
            self.selector = ExampleSelector(self.encoder, self.index_manager)

    def classify_query(
        self,
        query: str,
        strategy: str = "dynamic_few_shot",
        model: Optional[str] = None,
        k: int = 5,
        confidence_threshold: Optional[float] = None,
        diversity: bool = False,
        use_cache: bool = True,
        generate_reply: bool = False
    ) -> Dict[str, Any]:
        """
        Executes classification on a single customer query.
        Strategies: 'zero_shot', 'one_shot', 'few_shot', 'dynamic_few_shot', 'optimized'
        """
        target_model = model or self.config.primary_model
        threshold = confidence_threshold if confidence_threshold is not None else self.config.default_confidence_threshold
        pricing = self.config.get_pricing(target_model)
        cost_analyzer = CostAnalyzer(
            input_price_per_million=pricing.get("input_per_million", 0.20),
            output_price_per_million=pricing.get("output_per_million", 0.20)
        )

        retrieved_examples: List[Dict[str, Any]] = []
        demonstrations_str = ""

        # Step 1: Retrieval (if strategy requires dynamic exemplars)
        if strategy in ("dynamic_few_shot", "optimized"):
            if self.selector:
                if diversity:
                    retrieved_examples = self.selector.retrieve_diverse(
                        query, k=k, candidate_pool_size=self.config.diversity_candidate_pool
                    )
                else:
                    retrieved_examples = self.selector.retrieve_similar(query, k=k)
                demonstrations_str = self.selector.format_demonstrations(retrieved_examples)

        # Step 2: Build Prompt
        strat_lower = strategy.lower()
        if strat_lower == "zero_shot":
            prompt_text = self.prompt_builder.build_zero_shot(query)
        elif strat_lower == "one_shot":
            prompt_text = self.prompt_builder.build_one_shot(query)
        elif strat_lower == "few_shot":
            prompt_text = self.prompt_builder.build_few_shot(query)
        elif strat_lower == "dynamic_few_shot":
            prompt_text = self.prompt_builder.build_dynamic_few_shot(query, demonstrations_str)
        elif strat_lower == "optimized":
            prompt_text = self.prompt_builder.build_optimized(query, demonstrations_str)
        else:
            raise ValueError(f"Unsupported strategy: {strategy}")

        # Step 3: Call Groq API
        raw_completion = self.groq_client.generate(
            prompt=prompt_text,
            model=target_model,
            temperature=self.config.raw_config.get("generation", {}).get("temperature", 0.0),
            use_cache=use_cache
        )

        # Step 4: Structured Output Validation
        prediction = self.parser.parse(raw_completion.get("text", ""))

        # Step 5: Confidence Routing
        # if confidence < threshold, flag for human review
        if prediction.confidence < threshold or prediction.intent not in self.valid_intents:
            prediction.needs_review = True
        else:
            prediction.needs_review = False

        # Step 6: Cost Estimation
        input_tokens = raw_completion.get("input_tokens", 0)
        output_tokens = raw_completion.get("output_tokens", 0)
        cost_usd = cost_analyzer.calculate_query_cost(input_tokens, output_tokens)

        # Step 7: Automated Customer Reply Generation
        automated_reply = None
        if generate_reply:
            automated_reply = self.response_generator.generate_reply(
                query=query,
                intent=prediction.intent,
                confidence=prediction.confidence,
                needs_review=prediction.needs_review,
                model=target_model
            )

        return {
            "query": query,
            "prediction": prediction,
            "predicted_intent": prediction.intent,
            "confidence": prediction.confidence,
            "needs_review": prediction.needs_review,
            "strategy": strategy,
            "model": target_model,
            "retrieved_examples": retrieved_examples,
            "latency_ms": raw_completion.get("latency_ms", 0.0),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": raw_completion.get("total_tokens", 0),
            "estimated_cost_usd": cost_usd,
            "cached": raw_completion.get("cached", False),
            "raw_text": raw_completion.get("text", ""),
            "automated_reply": automated_reply
        }

    def evaluate_dataset(
        self,
        df: pd.DataFrame,
        strategy: str = "dynamic_few_shot",
        model: Optional[str] = None,
        k: int = 5,
        diversity: bool = False,
        confidence_threshold: Optional[float] = None,
        max_samples: Optional[int] = None
    ) -> pd.DataFrame:
        """Runs the pipeline across a dataset partition and returns results in a DataFrame."""
        eval_df = df.copy()
        if max_samples and len(eval_df) > max_samples:
            eval_df = eval_df.head(max_samples)

        results = []
        for _, row in tqdm(eval_df.iterrows(), total=len(eval_df), desc=f"Evaluating {strategy}"):
            text = row["text"]
            true_category = row["category"]
            out = self.classify_query(
                query=text,
                strategy=strategy,
                model=model,
                k=k,
                diversity=diversity,
                confidence_threshold=confidence_threshold
            )
            out["true_intent"] = true_category
            results.append(out)

        return pd.DataFrame(results)

    @staticmethod
    def analyze_confidence_thresholds(
        results_df: pd.DataFrame,
        thresholds: Optional[List[float]] = None
    ) -> pd.DataFrame:
        """
        Evaluates confidence routing across thresholds:
        - coverage: fraction of queries accepted without review
        - accepted_accuracy: accuracy on accepted queries
        - review_rate: fraction of queries routed to review
        - overall_accuracy: accuracy across all queries
        """
        if thresholds is None:
            thresholds = [0.60, 0.70, 0.80, 0.90]

        rows = []
        total = len(results_df)
        if total == 0:
            return pd.DataFrame()

        for t in thresholds:
            accepted = results_df[results_df["confidence"] >= t]
            reviewed = results_df[results_df["confidence"] < t]

            coverage = len(accepted) / total
            review_rate = len(reviewed) / total

            if len(accepted) > 0:
                acc_accepted = float((accepted["predicted_intent"] == accepted["true_intent"]).mean())
            else:
                acc_accepted = 0.0

            acc_overall = float((results_df["predicted_intent"] == results_df["true_intent"]).mean())

            rows.append({
                "threshold": t,
                "coverage": round(coverage, 4),
                "review_rate": round(review_rate, 4),
                "accepted_accuracy": round(acc_accepted, 4),
                "overall_accuracy": round(acc_overall, 4)
            })

        return pd.DataFrame(rows)
