"""
Cost calculation for Lemonade LLM provider.

Since Lemonade is a local/self-hosted service, all costs default to 0.
This prevents cost calculation errors when using models not in model_prices_and_context_window.json
"""
from typing import Tuple

from litellm.types.utils import Usage


def cost_per_token(
    model: str,
    usage: Usage,
) -> Tuple[float, float]:
    """
    Calculate cost per token for Lemonade models.
    
    Since Lemonade is a local/self-hosted deployment, there are no per-token costs.
    This function returns (0.0, 0.0) for all models to allow cost tracking to work
    without errors for any Lemonade model, regardless of whether it's in the
    model_prices_and_context_window.json file.
    
    Args:
        model: The model name (with or without "lemonade/" prefix)
        usage: Usage object containing token counts
        
    Returns:
        Tuple of (prompt_cost, completion_cost) - always (0.0, 0.0) for Lemonade
    """
    # Lemonade is self-hosted/local, so cost is always 0
    prompt_cost = 0.0
    completion_cost = 0.0
    
    return prompt_cost, completion_cost
