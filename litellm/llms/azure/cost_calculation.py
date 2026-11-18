"""
Helper util for handling azure openai-specific cost calculation
- e.g.: prompt caching, audio tokens
"""

from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import branch_cost
from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(
    model: str,
    usage: Usage,
    response_time_ms: float | None = 0.0,
    service_tier: str | None = None,
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing caching and audio token information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    return branch_cost(
        model, "azure", usage, get_model_info(model=model, custom_llm_provider="azure"), service_tier=service_tier
    )
