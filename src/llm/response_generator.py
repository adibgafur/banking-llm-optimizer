"""
Automated customer response generation for BankingLLM-Optimizer.
Synthesizes professional, intent-grounded replies for customer support inquiries.
"""

import logging
from typing import Optional, Dict, Any

from src.llm.groq_client import GroqClient

logger = logging.getLogger(__name__)

# Standard fallback guidance when offline or in human review escalation
ESCALATION_TEMPLATE = (
    "Thank you for reaching out to our banking support team. We have received your inquiry regarding "
    "your account. Because your request requires specialized verification, we have routed this to "
    "a dedicated banking specialist who will assist you shortly. For your security, our team will "
    "never ask you for your PIN, CVV, or full online banking password."
)

DOMAIN_PLAYBOOKS = {
    "card_arrival": (
        "Standard debit and credit cards typically arrive within 5-7 business days from the dispatch date. "
        "You can track real-time delivery status in the mobile app under 'Cards' > 'Card Delivery'. If it has "
        "been over 10 business days, please confirm your mailing address with us so we can issue a free replacement."
    ),
    "card_linking": (
        "To link your card, open the mobile app, navigate to 'Cards', and select 'Link Card'. Enter your 16-digit "
        "card number and expiry date, then authenticate with SMS or biometric verification to activate it instantly."
    ),
    "pin_blocked": (
        "If your PIN has been blocked due to multiple incorrect attempts, you can easily view your PIN or reset it "
        "by navigating to 'Cards' > 'Security & PIN' in your mobile banking app, followed by two-factor authentication."
    ),
    "transfer_not_received_by_recipient": (
        "Domestic transfers typically clear within 2 hours, though some inter-bank networks may take up to 1 business day. "
        "Please check your transaction history for the transfer reference ID and ensure the recipient's sort code/routing number "
        "and account number were entered correctly."
    ),
    "card_payment_fee_charged": (
        "Card payment fees may apply when making transactions in foreign currencies or when international merchant surcharges "
        "are assessed. You can review the itemized breakdown of any fee in the transaction details within your app."
    ),
    "automatic_top_up": (
        "Automatic top-ups trigger whenever your balance falls below your configured minimum threshold. You can modify or "
        "disable auto top-up rules anytime under 'Add Money' > 'Automatic Top-up Settings'."
    )
}


class BankingResponseGenerator:
    """Generates context-aware, professional customer support responses."""

    def __init__(self, groq_client: GroqClient):
        self.client = groq_client

    def generate_reply(
        self,
        query: str,
        intent: str,
        confidence: float,
        needs_review: bool,
        model: str = "openai/gpt-oss-20b"
    ) -> str:
        """
        Generates an automated customer response based on intent and review status.
        If flagged for human review, returns a secure escalation acknowledgment.
        """
        # If in mock mode or offline, use domain playbook or default guidance
        if self.client.mock_mode:
            if needs_review or intent == "unknown":
                return ESCALATION_TEMPLATE
            return DOMAIN_PLAYBOOKS.get(
                intent,
                f"Thank you for contacting us regarding {intent.replace('_', ' ')}. "
                "Our team is actively processing your inquiry according to our standard operating guidelines. "
                "You can also manage your account features 24/7 directly in the mobile banking app."
            )

        # Generate custom empathetic response using Groq LLM
        if needs_review or intent == "unknown":
            prompt = (
                f"You are a friendly, professional banking customer service representative.\n"
                f"The customer asked an inquiry that requires human specialist review or falls outside automated workflows: \"{query}\"\n"
                f"Write a concise, polite, and reassuring response directly to the customer (2-3 sentences).\n"
                f"Requirements:\n"
                f"1. Acknowledge their specific question directly (e.g., creating an account or checking limits).\n"
                f"2. Provide helpful general guidance (e.g. they can start account registration in the mobile app with a valid ID, or check limits in the app).\n"
                f"3. Reassure them that our specialist banking team has also received their request and will assist them shortly.\n"
                f"4. Return only the plain customer-facing text, no JSON or quotes.\n"
                f"Response:"
            )
        else:
            prompt = (
                f"You are a friendly, professional banking customer service representative.\n"
                f"Write a concise, polite, and helpful reply directly to the customer based on their query and confirmed intent.\n\n"
                f"Customer Message: \"{query}\"\n"
                f"Confirmed Intent: {intent}\n"
                f"Requirements:\n"
                f"1. Directly address the customer's specific inquiry.\n"
                f"2. Provide clear next steps, timeframes, or instructions.\n"
                f"3. Keep the response to 2-3 sentences.\n"
                f"4. Do NOT output JSON or meta-commentary; return only the customer-facing reply text.\n"
                f"Response:"
            )

        try:
            res = self.client.generate(
                prompt=prompt,
                model=model,
                temperature=0.3,
                max_tokens=300,
                system_prompt="You are an empathetic, highly trained banking customer support agent. Reply directly to the customer."
            )
            reply_text = res.get("text", "").strip()

            # Clean any stray JSON wrapping if present
            if reply_text.startswith("{") and reply_text.endswith("}"):
                import json
                try:
                    data = json.loads(reply_text)
                    reply_text = data.get("response") or data.get("reply") or reply_text
                except Exception:
                    pass

            if not reply_text:
                return DOMAIN_PLAYBOOKS.get(intent, ESCALATION_TEMPLATE)

            return reply_text
        except Exception as e:
            logger.warning(f"Failed to generate LLM reply, using domain fallback: {e}")
            return DOMAIN_PLAYBOOKS.get(intent, ESCALATION_TEMPLATE)
