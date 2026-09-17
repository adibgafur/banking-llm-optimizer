"""
Structured logging utilities for experiment tracking and auditability.
"""

import os
import sys
import logging
from typing import Dict, Any, Optional
import json

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, "experiments.log")


def setup_logger(name: str = "banking_llm", log_file: str = DEFAULT_LOG_FILE, level: int = logging.INFO) -> logging.Logger:
    """Configures a standardized console and file logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


class ExperimentLogger:
    """Logs structured experiment events without recording sensitive keys or credentials."""

    def __init__(self, experiment_name: str, logger_instance: Optional[logging.Logger] = None):
        self.experiment_name = experiment_name
        self.logger = logger_instance or setup_logger(experiment_name)

    def log_query_event(
        self,
        query_id: Any,
        strategy: str,
        model: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        total_tokens: int,
        predicted_intent: str,
        confidence: float,
        cached: bool = False,
        error: Optional[str] = None
    ):
        """Records a single prediction invocation event in JSON-formatted log entry."""
        event = {
            "experiment": self.experiment_name,
            "query_id": str(query_id),
            "strategy": strategy,
            "model": model,
            "latency_ms": round(latency_ms, 2),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "predicted_intent": predicted_intent,
            "confidence": round(confidence, 4),
            "cached": cached,
            "error": error
        }
        if error:
            self.logger.warning(f"Experiment Event: {json.dumps(event)}")
        else:
            self.logger.info(f"Experiment Event: {json.dumps(event)}")
