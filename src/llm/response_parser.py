"""
Structured response parsing and Pydantic validation for BankingLLM-Optimizer.
Extracts and validates structured predictions from raw LLM completions.
"""

import re
import json
import logging
from typing import Optional, Set, Dict, Any, List
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class BankingPrediction(BaseModel):
    """Pydantic model representing validated intent classification output."""

    intent: str
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    needs_review: bool = Field(default=False)
    raw_response: Optional[str] = Field(default=None, exclude=True)
    parsing_error: Optional[str] = Field(default=None, exclude=True)

    @field_validator("confidence", mode="before")
    @classmethod
    def clamp_confidence(cls, v: Any) -> float:
        try:
            val = float(v)
            return max(0.0, min(1.0, val))
        except (ValueError, TypeError):
            return 0.0

    @field_validator("needs_review", mode="before")
    @classmethod
    def parse_bool(cls, v: Any) -> bool:
        if isinstance(v, bool):
            return v
        if isinstance(v, str):
            return v.strip().lower() in ("true", "1", "yes")
        return bool(v)


class ResponseParser:
    """Extracts JSON and validates BankingPrediction against canonical 77 intents."""

    def __init__(self, valid_intents: List[str]):
        self.valid_intents: Set[str] = set(valid_intents)
        # Case-insensitive mapping to handle dataset anomalies (e.g. 'Refund_not_showing_up')
        self.canonical_map: Dict[str, str] = {i.lower(): i for i in valid_intents}

    def parse(self, raw_text: str) -> BankingPrediction:
        """
        Extracts JSON from raw LLM text and validates it.
        Guaranteed to never raise an unhandled exception or crash the pipeline.
        """
        if not raw_text or not raw_text.strip():
            return BankingPrediction(
                intent="unknown",
                confidence=0.0,
                needs_review=True,
                raw_response=raw_text,
                parsing_error="Empty or whitespace response"
            )

        extracted_json = self._extract_json_substring(raw_text)
        if not extracted_json:
            return BankingPrediction(
                intent="unknown",
                confidence=0.0,
                needs_review=True,
                raw_response=raw_text,
                parsing_error="No JSON structure detected"
            )

        try:
            data = json.loads(extracted_json)
        except json.JSONDecodeError as err:
            logger.warning(f"Failed to decode JSON from model completion: {err}")
            return BankingPrediction(
                intent="unknown",
                confidence=0.0,
                needs_review=True,
                raw_response=raw_text,
                parsing_error=f"JSONDecodeError: {str(err)}"
            )

        cleaned_data: Dict[str, Any] = {}
        for k, v in data.items():
            cleaned_data[k.lower().strip()] = v

        raw_intent = str(cleaned_data.get("intent", "unknown")).strip()
        confidence = cleaned_data.get("confidence", 0.0)
        needs_review = cleaned_data.get("needs_review", False)

        # Normalize via case-insensitive canonical map
        canonical_intent = self.canonical_map.get(raw_intent.lower())
        if not canonical_intent:
            logger.warning(f"Model generated invalid intent '{raw_intent}' not in canonical 77 list.")
            return BankingPrediction(
                intent=raw_intent,
                confidence=float(confidence) if isinstance(confidence, (int, float)) else 0.0,
                needs_review=True,
                raw_response=raw_text,
                parsing_error=f"Intent '{raw_intent}' is not one of the 77 valid BANKING77 intents"
            )

        try:
            prediction = BankingPrediction(
                intent=canonical_intent,
                confidence=confidence,
                needs_review=needs_review,
                raw_response=raw_text
            )
            return prediction
        except Exception as e:
            logger.error(f"Pydantic instantiation failed: {e}")
            return BankingPrediction(
                intent=canonical_intent if canonical_intent in self.valid_intents else "unknown",
                confidence=0.0,
                needs_review=True,
                raw_response=raw_text,
                parsing_error=str(e)
            )

    @staticmethod
    def _extract_json_substring(text: str) -> Optional[str]:
        """Extracts JSON object from markdown code blocks or raw braces."""
        code_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if code_block:
            return code_block.group(1).strip()

        brace_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if brace_match:
            candidate = brace_match.group(1).strip()
            first_brace = candidate.find("{")
            last_brace = candidate.rfind("}")
            if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
                return candidate[first_brace : last_brace + 1]

        return None
