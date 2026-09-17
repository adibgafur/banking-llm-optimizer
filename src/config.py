"""
Configuration management for BankingLLM-Optimizer.
Reads YAML configurations, environment variables, and pricing tables.
"""

import os
import yaml
from typing import Dict, Any, List, Optional
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_CONFIG_PATH = CONFIG_DIR / "experiments.yaml"
DEFAULT_INTENTS_PATH = CONFIG_DIR / "intents.json"


class AppConfig:
    """Central configuration class reading configs/experiments.yaml."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self.raw_config = self._load_yaml()

    def _load_yaml(self) -> Dict[str, Any]:
        if not self.config_path.exists():
            return {}
        with open(self.config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    @property
    def groq_api_key(self) -> Optional[str]:
        return os.getenv("GROQ_API_KEY")

    @property
    def primary_model(self) -> str:
        return self.raw_config.get("model", {}).get("primary", "qwen/qwen3.8-27b")

    @property
    def secondary_model(self) -> str:
        return self.raw_config.get("model", {}).get("secondary", "openai/gpt-oss-20b")

    @property
    def tertiary_model(self) -> str:
        return self.raw_config.get("model", {}).get("tertiary", "openai/gpt-oss-120b")

    @property
    def available_models(self) -> List[str]:
        models = [self.primary_model, self.secondary_model, self.tertiary_model]
        return [m for m in models if m]

    @property
    def embedding_model_name(self) -> str:
        return self.raw_config.get("retrieval", {}).get("embedding_model", "sentence-transformers/all-MiniLM-L6-v2")

    @property
    def embedding_dimension(self) -> int:
        return self.raw_config.get("retrieval", {}).get("dimension", 384)

    @property
    def k_values(self) -> List[int]:
        return self.raw_config.get("retrieval", {}).get("k_values", [1, 3, 5, 8, 10])

    @property
    def diversity_candidate_pool(self) -> int:
        return self.raw_config.get("retrieval", {}).get("diversity_candidate_pool", 20)

    @property
    def default_confidence_threshold(self) -> float:
        return self.raw_config.get("confidence", {}).get("default_threshold", 0.80)

    @property
    def confidence_thresholds(self) -> List[float]:
        return self.raw_config.get("confidence", {}).get("thresholds", [0.60, 0.70, 0.80, 0.90])

    @property
    def validation_samples(self) -> int:
        return self.raw_config.get("evaluation", {}).get("validation_samples", 500)

    @property
    def final_test_samples(self) -> int:
        return self.raw_config.get("evaluation", {}).get("final_test_samples", 3080)

    @property
    def random_seed(self) -> int:
        return self.raw_config.get("evaluation", {}).get("random_seed", 42)

    @property
    def cache_enabled(self) -> bool:
        return self.raw_config.get("cache", {}).get("enabled", True)

    @property
    def cache_db_path(self) -> str:
        return self.raw_config.get("cache", {}).get("db_path", "cache/llm_cache.sqlite")

    def get_pricing(self, model_name: Optional[str] = None) -> Dict[str, float]:
        """Returns input and output token pricing per 1,000,000 tokens."""
        pricing_table = self.raw_config.get("pricing", {})
        target = model_name or self.primary_model
        if target in pricing_table:
            return pricing_table[target]
        return pricing_table.get("default", {"input_per_million": 0.20, "output_per_million": 0.20})
