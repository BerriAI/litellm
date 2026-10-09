"""
Helper util for handling azure openai-specific cost calculation
- e.g.: prompt caching, audio tokens
"""

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage


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
    ## Use generic cost calculator for all other cases
    ## This properly handles: text tokens, audio tokens, cached tokens, reasoning tokens, etc.
    return generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="azure",
        service_tier=service_tier,
    )
