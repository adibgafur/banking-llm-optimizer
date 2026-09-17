"""
Groq API Client with exponential backoff retries, latency timing,
token usage tracking, SQLite caching, and offline mock mode.
"""

import os
import time
import json
import logging
from typing import Dict, Any, Optional
from groq import Groq

from src.utils.caching import LLMCache

logger = logging.getLogger(__name__)


class GroqClient:
    """Client for querying Groq LLMs with automatic retry, caching, and offline simulation."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        cache: Optional[LLMCache] = None,
        mock_mode: bool = False,
        max_retries: int = 3,
        backoff_factor: float = 1.2
    ):
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.mock_mode = mock_mode or (not self.api_key or self.api_key.startswith("your_"))
        self.cache = cache
        self.max_retries = max_retries
        self.backoff_factor = backoff_factor

        if not self.mock_mode:
            self.client = Groq(api_key=self.api_key)
            logger.info("Initialized live Groq client.")
        else:
            self.client = None
            logger.info("Initialized Groq client in MOCK MODE (offline simulation enabled).")

    def generate(
        self,
        prompt: str,
        model: str = "qwen/qwen3.8-27b",
        temperature: float = 0.0,
        max_tokens: int = 600,
        system_prompt: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Executes a completion request with cache lookup, retry logic, and latency profiling.
        Returns a dict containing text, input_tokens, output_tokens, total_tokens, latency_ms, cached.
        """
        # 1. Check persistent cache
        if use_cache and self.cache:
            cached_result = self.cache.get(model=model, prompt=prompt, temperature=temperature)
            if cached_result and cached_result.get("text", "").strip():
                cached_result["model"] = model
                return cached_result

        # 2. Mock mode if offline or no key
        if self.mock_mode:
            return self._mock_completion(prompt, model, temperature)

        # 3. Live API call with adaptive JSON format
        messages = [
            {"role": "system", "content": system_prompt or "You are a banking customer support intent classification system. Always return valid JSON only."}
        ]
        messages.append({"role": "user", "content": prompt})

        last_exception: Optional[Exception] = None
        use_json_format = True
        active_model = model

        for attempt in range(1, self.max_retries + 1):
            start_time = time.perf_counter()
            try:
                kwargs = {
                    "model": active_model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if use_json_format:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000.0

                msg = response.choices[0].message
                completion_text = msg.content or ""

                # If content is empty but model emitted reasoning (e.g. gpt-oss models), extract JSON from reasoning
                if not completion_text.strip() and hasattr(msg, "reasoning") and msg.reasoning:
                    import re
                    match = re.search(r"(\{.*?\})", msg.reasoning, re.DOTALL)
                    if match:
                        completion_text = match.group(1).strip()

                input_tokens = response.usage.prompt_tokens if response.usage else len(prompt.split()) * 2
                output_tokens = response.usage.completion_tokens if response.usage else len(completion_text.split()) * 2
                total_tokens = response.usage.total_tokens if response.usage else input_tokens + output_tokens

                # Write to cache only if valid non-empty response
                if use_cache and self.cache and completion_text.strip():
                    self.cache.set(
                        model=active_model,
                        prompt=prompt,
                        temperature=temperature,
                        completion_text=completion_text,
                        input_tokens=input_tokens,
                        output_tokens=output_tokens,
                        total_tokens=total_tokens,
                        latency_ms=latency_ms
                    )

                return {
                    "text": completion_text,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "total_tokens": total_tokens,
                    "latency_ms": latency_ms,
                    "cached": False,
                    "model": active_model
                }

            except Exception as e:
                last_exception = e
                err_str = str(e).lower()
                # If rate-limited on gpt-oss, automatically fallback to qwen/qwen3.8-27b
                if ("rate_limit" in err_str or "429" in err_str or "tpd" in err_str) and active_model != "qwen/qwen3.8-27b":
                    logger.warning(f"Rate limit on {active_model}, falling back to qwen/qwen3.8-27b: {e}")
                    active_model = "qwen/qwen3.8-27b"
                    continue

                # If json_validate_failed, immediately disable json_object format and retry without sleeping
                if "json_validate_failed" in err_str or "response_format" in err_str:
                    use_json_format = False
                    continue

                logger.warning(f"Groq API attempt {attempt}/{self.max_retries} failed: {e}")
                if attempt < self.max_retries:
                    sleep_time = self.backoff_factor ** attempt
                    time.sleep(sleep_time)

        # Fallback if all retries fail
        logger.error(f"All {self.max_retries} Groq API retries failed: {last_exception}")
        return {
            "text": json.dumps({"intent": "unknown", "confidence": 0.0, "needs_review": True}),
            "input_tokens": 0,
            "output_tokens": 0,
            "total_tokens": 0,
            "latency_ms": 0.0,
            "cached": False,
            "model": active_model,
            "error": str(last_exception)
        }

    def _mock_completion(self, prompt: str, model: str, temperature: float) -> Dict[str, Any]:
        """Offline simulator for zero-cost testing and automated test suites."""
        time.sleep(0.01)
        prompt_lower = prompt.lower()
        if "card" in prompt_lower and ("arrive" in prompt_lower or "waiting" in prompt_lower):
            simulated_intent = "card_arrival"
        elif "pin" in prompt_lower:
            simulated_intent = "get_physical_card"
        elif "transfer" in prompt_lower:
            simulated_intent = "transfer_not_received_by_recipient"
        else:
            simulated_intent = "card_arrival"

        simulated_json = json.dumps({
            "intent": simulated_intent,
            "confidence": 0.92,
            "needs_review": False
        })
        input_tokens = len(prompt.split()) * 2
        output_tokens = 25
        total_tokens = input_tokens + output_tokens

        return {
            "text": simulated_json,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "total_tokens": total_tokens,
            "latency_ms": 42.5,
            "cached": False,
            "model": model
        }
