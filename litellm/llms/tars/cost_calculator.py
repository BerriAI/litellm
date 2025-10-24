"""
Helper util for handling TARS-specific cost calculation.
- Uses the generic cost calculator which already handles tiered pricing correctly.
- Adds a 5% margin to the base model costs.
- Returns (0.0, 0.0) when no pricing is available.
"""

from typing import Tuple

from litellm.types.utils import Usage
from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.utils import get_model_info


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given TARS model with a 5% margin.
    Uses the generic cost calculator for all pricing logic.

    Input:
        - model: str, the model name without provider prefix.
        - usage: LiteLLM Usage block, containing usage information.

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd.
        Returns (0.0, 0.0) if no pricing is available.
    """
    try:
        # Check if pricing is available for this model.
        model_info = get_model_info(model=model, custom_llm_provider="tars")
        
        # If no pricing is available, return (0.0, 0.0).
        if not model_info or (
            model_info.get("input_cost_per_token", 0) == 0 and 
            model_info.get("output_cost_per_token", 0) == 0
        ):
            return (0.0, 0.0)
        
        # Calculate base cost using generic calculator.
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider="tars"
        )
        
        # Add 5% margin to both costs.
        margin_multiplier = 1.05
        prompt_cost_with_margin = prompt_cost * margin_multiplier
        completion_cost_with_margin = completion_cost * margin_multiplier
        
        return prompt_cost_with_margin, completion_cost_with_margin
    except Exception:
        # If any error occurs (e.g., model not found), return (0.0, 0.0).
        return (0.0, 0.0)

