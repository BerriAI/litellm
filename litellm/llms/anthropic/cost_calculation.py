"""
Helper util for handling anthropic-specific cost calculation
- e.g.: prompt caching
"""

from typing import Tuple

from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## GET MODEL INFO
    model_info = get_model_info(model=model, custom_llm_provider="anthropic")

    ## CALCULATE INPUT COST

    prompt_cost: float = usage["prompt_tokens"] * model_info["input_cost_per_token"]
    if model_info.get("cache_creation_input_token_cost") is not None:
        prompt_cost += (
            usage._cache_creation_input_tokens  # type: ignore
            * model_info["cache_creation_input_token_cost"]
        )
    if model_info.get("cache_read_input_token_cost") is not None:
        prompt_cost += (
            usage._cache_read_input_tokens * model_info["cache_read_input_token_cost"]  # type: ignore
        )

    ## CALCULATE OUTPUT COST
    completion_cost = usage["completion_tokens"] * model_info["output_cost_per_token"]

    return prompt_cost, completion_cost
