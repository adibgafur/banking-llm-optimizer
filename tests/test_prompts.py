"""
Unit tests for prompt construction and template formatting.
"""

import pytest
from src.prompts.templates import PromptBuilder, STATIC_FEW_SHOT_EXEMPLARS
from src.prompts.optimizer import PromptOptimizer


@pytest.fixture
def sample_intents():
    return ["card_arrival", "pin_blocked", "transfer_not_received_by_recipient", "balance_not_updated_after_bank_transfer"]


def test_zero_shot_prompt_builder(sample_intents):
    builder = PromptBuilder(sample_intents)
    prompt = builder.build_zero_shot("When will my card arrive?")
    assert "When will my card arrive?" in prompt
    assert "card_arrival" in prompt
    assert "Return only valid JSON." in prompt


def test_one_shot_prompt_builder(sample_intents):
    builder = PromptBuilder(sample_intents)
    prompt = builder.build_one_shot("I lost my card.")
    assert "I lost my card." in prompt
    assert "I am still waiting for my card." in prompt
    assert "card_arrival" in prompt


def test_few_shot_prompt_builder(sample_intents):
    builder = PromptBuilder(sample_intents)
    prompt = builder.build_few_shot("Need help with PIN")
    assert "Need help with PIN" in prompt
    assert "pin_blocked" in prompt
    assert "Example 1:" in prompt


def test_dynamic_few_shot_prompt_builder(sample_intents):
    builder = PromptBuilder(sample_intents)
    demos = "Customer: \"Where is my card?\"\nIntent: card_arrival"
    prompt = builder.build_dynamic_few_shot("Still waiting", demos)
    assert "Still waiting" in prompt
    assert demos in prompt


def test_optimized_prompt_builder(sample_intents):
    builder = PromptBuilder(sample_intents)
    demos = "Customer: \"Where is my card?\"\nIntent: card_arrival"
    prompt = builder.build_optimized("Still waiting", demos)
    assert "You are an expert banking intent classification system." in prompt
    assert "RULES:" in prompt
    assert "RELEVANT TRAINING EXAMPLES:" in prompt


def test_ablation_variants(sample_intents):
    builder = PromptBuilder(sample_intents)
    for variant in ["A", "B", "C", "D", "E", "F"]:
        p = builder.build_ablation(variant, "My transaction failed")
        assert len(p) > 0
        assert "My transaction failed" in p
