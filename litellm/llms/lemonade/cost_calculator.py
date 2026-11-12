"""
Cost calculation for Lemonade LLM provider.

Since Lemonade is a local/self-hosted service, all costs default to 0.
This prevents cost calculation errors when using models not in model_prices_and_context_window.json
"""

from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import ModelInfo, Usage


def cost_per_token(
    model: str,
    usage: Usage,
    model_info: ModelInfo | None = None,
) -> tuple[float, float]:
    """
    Calculate configured token costs, or return zero for unpriced Lemonade models.

    Since Lemonade is a local/self-hosted deployment, unpriced models have no per-token costs.
    This function returns (0.0, 0.0) for those models, regardless of whether they are in the
    model_prices_and_context_window.json file.

    Args:
        model: The model name (with or without "lemonade/" prefix)
        usage: Usage object containing token counts
        model_info: Optional resolved deployment prices

    Returns:
        Tuple of (prompt_cost, completion_cost)
    """
    if model_info is not None:
        return generic_cost_per_token(model=model, usage=usage, custom_llm_provider="lemonade", model_info=model_info)

    # Lemonade is self-hosted/local, so cost is always 0
    prompt_cost: Final = 0.0
    completion_cost: Final = 0.0

    return prompt_cost, completion_cost
