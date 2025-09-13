"""
Cost calculator for Alibaba Chat models.
"""

from typing import Tuple

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.
    
    Uses the standard cost calculation for alibaba models.
    """
    return generic_cost_per_token(
        model=model, usage=usage, custom_llm_provider="alibaba"
    )