"""
Cost calculation for Lemonade LLM provider.

Since Lemonade is a local/self-hosted service, all costs default to 0.
This prevents cost calculation errors when using models not in model_prices_and_context_window.json
"""

from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import branch_cost
from litellm.types.utils import Usage


def cost_per_token(
    model: str,
    usage: Usage,
) -> tuple[float, float]:
    """
    Calculate cost per token for Lemonade models: (0.0, 0.0) for any model, mapped or not, since Lemonade is
    self-hosted.

    Args:
        model: The model name (with or without "lemonade/" prefix)
        usage: Usage object containing token counts

    Returns:
        Tuple of (prompt_cost, completion_cost)
    """
    return branch_cost(model, "lemonade", usage, None)
