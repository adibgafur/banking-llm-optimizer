"""
Persistent SQLite caching layer for LLM completions.
Ensures deterministic reproducibility and prevents redundant API costs.
"""

import os
import sqlite3
import hashlib
import json
import logging
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


class LLMCache:
    """Persistent disk-based SQLite cache for LLM inference requests."""

    def __init__(self, db_path: str = "cache/llm_cache.sqlite", enabled: bool = True):
        self.db_path = db_path
        self.enabled = enabled
        if self.enabled:
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            self._init_db()

    def _init_db(self):
        """Initializes schema if not already present."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS response_cache (
                    cache_key TEXT PRIMARY KEY,
                    model TEXT NOT NULL,
                    prompt_hash TEXT NOT NULL,
                    temperature REAL NOT NULL,
                    completion_text TEXT NOT NULL,
                    input_tokens INTEGER,
                    output_tokens INTEGER,
                    total_tokens INTEGER,
                    latency_ms REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_cache_key ON response_cache(cache_key)")

    @staticmethod
    def compute_key(model: str, prompt: str, temperature: float) -> str:
        """Generates SHA-256 hash key from request parameters."""
        raw_payload = f"{model.strip()}||{temperature:.4f}||{prompt.strip()}"
        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

    def get(self, model: str, prompt: str, temperature: float = 0.0) -> Optional[Dict[str, Any]]:
        """Retrieves cached response dict if available, else None."""
        if not self.enabled:
            return None

        key = self.compute_key(model, prompt, temperature)
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT completion_text, input_tokens, output_tokens, total_tokens, latency_ms
                    FROM response_cache
                    WHERE cache_key = ?
                """, (key,))
                row = cursor.fetchone()
                if row:
                    return {
                        "text": row[0],
                        "input_tokens": row[1],
                        "output_tokens": row[2],
                        "total_tokens": row[3],
                        "latency_ms": row[4],
                        "cached": True
                    }
        except Exception as e:
            logger.warning(f"Cache read error for key {key}: {e}")

        return None

    def set(
        self,
        model: str,
        prompt: str,
        temperature: float,
        completion_text: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        total_tokens: int = 0,
        latency_ms: float = 0.0
    ):
        """Saves a completion and its metadata to the SQLite cache."""
        if not self.enabled:
            return

        key = self.compute_key(model, prompt, temperature)
        prompt_hash = hashlib.sha256(prompt.strip().encode("utf-8")).hexdigest()

        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO response_cache
                    (cache_key, model, prompt_hash, temperature, completion_text, input_tokens, output_tokens, total_tokens, latency_ms)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (key, model, prompt_hash, temperature, completion_text, input_tokens, output_tokens, total_tokens, latency_ms))
        except Exception as e:
            logger.warning(f"Cache write error for key {key}: {e}")

    def size(self) -> int:
        """Returns the number of cached items."""
        if not self.enabled or not os.path.exists(self.db_path):
            return 0
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM response_cache")
                return cursor.fetchone()[0]
        except Exception:
            return 0
