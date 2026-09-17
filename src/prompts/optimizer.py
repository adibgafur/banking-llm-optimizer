"""
Prompt optimizer utilities for composing, testing, and saving optimized prompts.
"""

import os
from typing import List, Dict, Any, Optional
from src.prompts.templates import (
    ZERO_SHOT_TEMPLATE,
    ONE_SHOT_TEMPLATE,
    FEW_SHOT_TEMPLATE,
    DYNAMIC_FEW_SHOT_TEMPLATE,
    OPTIMIZED_TEMPLATE,
    PromptBuilder
)


class PromptOptimizer:
    """Manages prompt composition and persists canonical prompt files."""

    def __init__(self, output_dir: str = "prompts"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def export_canonical_prompts(self):
        """Saves template texts to the prompts/ directory as required."""
        files = {
            "zero_shot.txt": ZERO_SHOT_TEMPLATE,
            "one_shot.txt": ONE_SHOT_TEMPLATE,
            "few_shot.txt": FEW_SHOT_TEMPLATE,
            "dynamic_few_shot.txt": DYNAMIC_FEW_SHOT_TEMPLATE,
            "optimized.txt": OPTIMIZED_TEMPLATE,
        }
        for filename, content in files.items():
            path = os.path.join(self.output_dir, filename)
            with open(path, "w", encoding="utf-8") as f:
                f.write(content)

    def analyze_token_overhead(self, prompt: str) -> Dict[str, Any]:
        """Rough token approximation for prompt length auditing."""
        words = prompt.split()
        return {
            "char_length": len(prompt),
            "word_count": len(words),
            "estimated_tokens": int(len(words) * 1.33)
        }
