"""
Unit tests for automated customer response generation and escalation.
"""

import pytest
from src.llm.groq_client import GroqClient
from src.llm.response_generator import BankingResponseGenerator, ESCALATION_TEMPLATE, DOMAIN_PLAYBOOKS


@pytest.fixture
def response_generator():
    mock_client = GroqClient(mock_mode=True)
    return BankingResponseGenerator(mock_client)


def test_automated_reply_high_confidence(response_generator):
    reply = response_generator.generate_reply(
        query="When is my card arriving?",
        intent="card_arrival",
        confidence=0.95,
        needs_review=False
    )
    assert isinstance(reply, str)
    assert len(reply) > 20
    assert "card" in reply.lower() or "5-7 business days" in reply.lower()


def test_escalation_reply_low_confidence(response_generator):
    reply = response_generator.generate_reply(
        query="Something weird happened to my money",
        intent="unknown",
        confidence=0.30,
        needs_review=True
    )
    assert reply == ESCALATION_TEMPLATE
    assert "PIN" in reply
    assert "banking specialist" in reply


def test_domain_playbook_coverage():
    assert "card_arrival" in DOMAIN_PLAYBOOKS
    assert "pin_blocked" in DOMAIN_PLAYBOOKS
    assert "transfer_not_received_by_recipient" in DOMAIN_PLAYBOOKS
