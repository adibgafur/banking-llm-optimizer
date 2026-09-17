"""
Unit tests for structured output parsing, regex extraction, and Pydantic validation.
"""

import pytest
from src.llm.response_parser import BankingPrediction, ResponseParser


@pytest.fixture
def valid_intents():
    return ["card_arrival", "pin_blocked", "transfer_into_account", "balance_not_updated_after_bank_transfer"]


def test_banking_prediction_pydantic_validation():
    # Valid model
    pred = BankingPrediction(intent="card_arrival", confidence=0.95, needs_review=False)
    assert pred.intent == "card_arrival"
    assert pred.confidence == 0.95
    assert pred.needs_review is False

    # Clamping out-of-range confidence
    pred_high = BankingPrediction(intent="card_arrival", confidence=1.5)
    assert pred_high.confidence == 1.0

    pred_low = BankingPrediction(intent="card_arrival", confidence=-0.5)
    assert pred_low.confidence == 0.0


def test_response_parser_markdown_block(valid_intents):
    parser = ResponseParser(valid_intents)
    raw_completion = """Here is the classification result:
```json
{
  "intent": "card_arrival",
  "confidence": 0.88,
  "needs_review": false
}
```
Hope that helps!"""
    pred = parser.parse(raw_completion)
    assert pred.intent == "card_arrival"
    assert pred.confidence == 0.88
    assert pred.needs_review is False


def test_response_parser_raw_json(valid_intents):
    parser = ResponseParser(valid_intents)
    raw = '{"intent": "pin_blocked", "confidence": 0.99, "needs_review": false}'
    pred = parser.parse(raw)
    assert pred.intent == "pin_blocked"
    assert pred.confidence == 0.99
    assert pred.needs_review is False


def test_response_parser_unknown_intent(valid_intents):
    parser = ResponseParser(valid_intents)
    raw = '{"intent": "fake_intent_not_in_list", "confidence": 0.90, "needs_review": false}'
    pred = parser.parse(raw)
    assert pred.intent == "fake_intent_not_in_list"
    assert pred.needs_review is True
    assert "not one of the 77 valid" in str(pred.parsing_error)


def test_response_parser_malformed_json_fallback(valid_intents):
    parser = ResponseParser(valid_intents)
    raw = "I think the intent is card_arrival but I won't format as json."
    pred = parser.parse(raw)
    assert pred.intent == "unknown"
    assert pred.confidence == 0.0
    assert pred.needs_review is True
    assert pred.parsing_error is not None
