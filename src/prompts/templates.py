"""
Prompt template engine for Zero-Shot, One-Shot, Few-Shot, Dynamic Few-Shot,
and prompt component ablations (Prompts A through F).
"""

from typing import List, Optional

ZERO_SHOT_TEMPLATE = """You are a banking customer-support intent classification system.
Your task is to classify the customer's message into exactly one intent from the provided list.
Do not invent a new intent.
Select the intent that best matches the customer's request.

Available intents:
{INTENT_LIST}

Customer message:
"{QUERY}"

Return only valid JSON.
Expected schema:
{{
  "intent": "one_intent_from_the_list",
  "confidence": 0.0,
  "needs_review": false
}}"""

ONE_SHOT_TEMPLATE = """You are a banking customer-support intent classification system.
Your task is to classify the customer's message into exactly one intent from the provided list.
Do not invent a new intent.
Select the intent that best matches the customer's request.

Available intents:
{INTENT_LIST}

Example:
Customer: "I am still waiting for my card."
Intent: card_arrival

Customer message:
"{QUERY}"

Return only valid JSON.
Expected schema:
{{
  "intent": "one_intent_from_the_list",
  "confidence": 0.0,
  "needs_review": false
}}"""

FEW_SHOT_TEMPLATE = """You are a banking customer-support intent classification system.
Your task is to classify the customer's message into exactly one intent from the provided list.
Do not invent a new intent.
Select the intent that best matches the customer's request.

Available intents:
{INTENT_LIST}

Examples from training pool:
{FEW_SHOT_EXAMPLES}

Customer message:
"{QUERY}"

Return only valid JSON.
Expected schema:
{{
  "intent": "one_intent_from_the_list",
  "confidence": 0.0,
  "needs_review": false
}}"""

DYNAMIC_FEW_SHOT_TEMPLATE = """You are a banking customer-support intent classification system.
Your task is to classify the customer's message into exactly one intent from the provided list.
Do not invent a new intent.
Select the intent that best matches the customer's request.

GUIDELINES:
1. Negation & Retraction: If the customer mentions what they do NOT want, what did NOT happen, or cancels an action (e.g. "i wanted to change password but now i dont want to change it . i am not finding my card"), ignore the negated statement and classify the active unresolved issue (e.g. lost_or_stolen_card).
2. Card Limits vs Loss: If the customer asks about card spending/contactless limits and clarifies their card is NOT lost, do NOT classify as lost_or_stolen_card. Use card limit intents (such as disposable_card_limits or top_up_limits).
3. Out of Scope: If the customer asks for something outside the 77 intents (e.g., creating a new bank account), pick the closest relevant category or "unknown" and set "confidence": 0.2, "needs_review": true.

Relevant demonstrations retrieved from training pool:
{RETRIEVED_EXAMPLES}

Available intents:
{INTENT_LIST}

Customer message:
"{QUERY}"

Return only valid JSON.
Expected schema:
{{
  "intent": "one_intent_from_the_list",
  "confidence": 0.0,
  "needs_review": false
}}"""

OPTIMIZED_TEMPLATE = """You are an expert banking intent classification system.

TASK:
Classify the customer's message into exactly one of the provided banking intents.

RULES:
1. Select exactly one intent from the provided list.
2. Never invent an intent.
3. Use the demonstrations as guidance for intent semantics and phrasing.
4. Focus on the customer's actual core request or inquiry.
5. Carefully distinguish semantically similar intents (e.g., card arrival vs card linking vs card delivery estimate).
6. Negation & Retraction: If a customer clarifies what they do NOT want, what did NOT happen, or retracts a previous statement (e.g., "i wanted to change password but now i dont want to . i am not finding my card"), discard the retracted topic and classify the active unresolved request (e.g. lost_or_stolen_card).
7. Limits vs Lost Cards: If the customer asks about daily contactless spending limits or transaction limits and clarifies the card is NOT lost, do NOT select lost_or_stolen_card. Choose the closest limit intent (e.g. disposable_card_limits or top_up_limits).
8. Out-of-Scope Inquiries: If the customer is asking about unsupported features or general onboarding (e.g., how to create a new account), select the closest intent or "unknown", assign low confidence (e.g. 0.20-0.50), and set "needs_review": true.
9. Do not copy an example intent unless the query genuinely shares the same underlying banking intent.
10. Return valid JSON only. Do not include introductory or concluding text.

RELEVANT TRAINING EXAMPLES:
{RETRIEVED_EXAMPLES}

AVAILABLE INTENTS:
{INTENT_LIST}

CUSTOMER MESSAGE:
"{QUERY}"

Return:
{{
  "intent": "exact_intent_name",
  "confidence": 0.0,
  "needs_review": false
}}"""

# Fixed 5 static exemplars for static Few-Shot (strictly from dev pool)
STATIC_FEW_SHOT_EXEMPLARS = """Example 1:
Customer: "I am still waiting for my card to arrive in the post."
Intent: card_arrival

Example 2:
Customer: "Why was I charged an extra fee when making a card payment abroad?"
Intent: card_payment_fee_charged

Example 3:
Customer: "How do I transfer money to an international account?"
Intent: transfer_into_account

Example 4:
Customer: "My PIN code is not working at the cash machine."
Intent: pin_blocked

Example 5:
Customer: "I noticed a pending transaction on my account that I did not authorize."
Intent: pending_card_payment"""


class PromptBuilder:
    """Builds ready-to-use prompts for different strategies and ablation variants."""

    def __init__(self, intent_list: List[str]):
        self.intent_list = intent_list
        self.formatted_intent_list = ", ".join(intent_list)

    def build_zero_shot(self, query: str) -> str:
        return ZERO_SHOT_TEMPLATE.format(
            INTENT_LIST=self.formatted_intent_list,
            QUERY=query
        )

    def build_one_shot(self, query: str) -> str:
        return ONE_SHOT_TEMPLATE.format(
            INTENT_LIST=self.formatted_intent_list,
            QUERY=query
        )

    def build_few_shot(self, query: str, custom_examples: Optional[str] = None) -> str:
        examples = custom_examples or STATIC_FEW_SHOT_EXEMPLARS
        return FEW_SHOT_TEMPLATE.format(
            INTENT_LIST=self.formatted_intent_list,
            FEW_SHOT_EXAMPLES=examples,
            QUERY=query
        )

    def build_dynamic_few_shot(self, query: str, retrieved_examples_str: str) -> str:
        return DYNAMIC_FEW_SHOT_TEMPLATE.format(
            INTENT_LIST=self.formatted_intent_list,
            RETRIEVED_EXAMPLES=retrieved_examples_str,
            QUERY=query
        )

    def build_optimized(self, query: str, retrieved_examples_str: str) -> str:
        return OPTIMIZED_TEMPLATE.format(
            INTENT_LIST=self.formatted_intent_list,
            RETRIEVED_EXAMPLES=retrieved_examples_str,
            QUERY=query
        )

    def build_ablation(
        self,
        prompt_type: str,
        query: str,
        examples_str: Optional[str] = None
    ) -> str:
        """
        Builds ablation variants (Prompts A through F):
        Prompt A: Role + task
        Prompt B: Role + task + intent list
        Prompt C: Role + task + intent list + examples
        Prompt D: Role + task + intent list + examples + classification rules
        Prompt E: Role + task + intent list + examples + rules + structured output
        Prompt F: Optimized dynamic few-shot prompt
        """
        pt = prompt_type.upper()
        if pt == "A":
            return f"""You are a banking customer-support intent classification system.\nClassify the following customer message into a banking intent.\n\nCustomer message: "{query}"\nIntent:"""
        elif pt == "B":
            return f"""You are a banking customer-support intent classification system.\nClassify the customer message into exactly one of the following intents:\n{self.formatted_intent_list}\n\nCustomer message: "{query}"\nIntent:"""
        elif pt == "C":
            ex = examples_str or STATIC_FEW_SHOT_EXEMPLARS
            return f"""You are a banking customer-support intent classification system.\nClassify the customer message into exactly one of the following intents:\n{self.formatted_intent_list}\n\nExamples:\n{ex}\n\nCustomer message: "{query}"\nIntent:"""
        elif pt == "D":
            ex = examples_str or STATIC_FEW_SHOT_EXEMPLARS
            rules = "Rules: Select exactly one intent from the list. Do not invent an intent. Focus on the customer's actual inquiry."
            return f"""You are a banking customer-support intent classification system.\nClassify the customer message into exactly one of the following intents:\n{self.formatted_intent_list}\n\n{rules}\n\nExamples:\n{ex}\n\nCustomer message: "{query}"\nIntent:"""
        elif pt == "E":
            ex = examples_str or STATIC_FEW_SHOT_EXEMPLARS
            rules = "Rules: Select exactly one intent from the list. Do not invent an intent. Focus on the customer's actual inquiry."
            return f"""You are a banking customer-support intent classification system.\nClassify the customer message into exactly one of the following intents:\n{self.formatted_intent_list}\n\n{rules}\n\nExamples:\n{ex}\n\nCustomer message: "{query}"\n\nReturn JSON: {{"intent": "...", "confidence": 0.0, "needs_review": false}}"""
        elif pt == "F":
            return self.build_optimized(query, examples_str or STATIC_FEW_SHOT_EXEMPLARS)
        else:
            raise ValueError(f"Unknown prompt ablation variant: {prompt_type}")
