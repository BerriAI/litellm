"""
Gemini Image Edit Cost Calculator
"""

from typing import Any

from litellm.llms.gemini.image_generation.cost_calculator import (
    cost_calculator as image_generation_cost_calculator,
)


def cost_calculator(
    model: str,
    image_response: Any,
) -> float:
    """
    Gemini image edit cost calculator.

    Gemini image edits and generations share image response billing behavior:
    use provider token usage when present, otherwise fall back to per-image pricing.
    """
    return image_generation_cost_calculator(
        model=model,
        image_response=image_response,
    )
