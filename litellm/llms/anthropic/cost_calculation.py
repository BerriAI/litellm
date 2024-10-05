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
    ### Cost of processing (non-cache hit + cache hit) + Cost of cache-writing (cache writing)
    prompt_cost = 0.0
    ### PROCESSING COST
    non_cache_hit_tokens = usage.prompt_tokens
    cache_hit_tokens = 0
    if usage.prompt_tokens_details and usage.prompt_tokens_details.cached_tokens:
        cache_hit_tokens = usage.prompt_tokens_details.cached_tokens
        non_cache_hit_tokens = non_cache_hit_tokens - cache_hit_tokens

    prompt_cost = float(non_cache_hit_tokens) * model_info["input_cost_per_token"]

    _cache_read_input_token_cost = model_info.get("cache_read_input_token_cost")
    if (
        _cache_read_input_token_cost is not None
        and usage.prompt_tokens_details
        and usage.prompt_tokens_details.cached_tokens
    ):
        prompt_cost += (
            float(usage.prompt_tokens_details.cached_tokens)
            * _cache_read_input_token_cost
        )

    ### CACHE WRITING COST
    _cache_creation_input_token_cost = model_info.get("cache_creation_input_token_cost")
    if _cache_creation_input_token_cost is not None:
        prompt_cost += (
            float(usage._cache_creation_input_tokens) * _cache_creation_input_token_cost
        )

    ## CALCULATE OUTPUT COST
    completion_cost = usage["completion_tokens"] * model_info["output_cost_per_token"]

    return prompt_cost, completion_cost
