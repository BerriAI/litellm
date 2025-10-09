"""
Helper util for handling XAI-specific cost calculation
- e.g.: reasoning tokens for grok models
- handles tiered pricing for tokens above 128k
"""

from typing import Tuple, Union

from litellm.types.utils import Usage
from litellm.utils import get_model_info


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given XAI model, prompt tokens, and completion tokens.
    Handles tiered pricing for tokens above 128k.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing XAI-specific usage information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## GET MODEL INFO
    model_info = get_model_info(model=model, custom_llm_provider="xai")

    def _safe_float_cast(value: Union[str, int, float, None, object], default: float = 0.0) -> float:
        """Safely cast a value to float with proper type handling for mypy."""
        if value is None:
            return default
        try:
            return float(value)  # type: ignore
        except (ValueError, TypeError):
            return default

    def _calculate_tiered_cost(tokens: int, base_cost_key: str, tiered_cost_key: str) -> float:
        """Calculate cost using tiered pricing if available.

        For xAI models:
        - First 128k tokens are billed at base rate
        - Tokens beyond 128k are billed at tiered rate
        """
        base_cost = _safe_float_cast(model_info.get(base_cost_key))
        tiered_cost = _safe_float_cast(model_info.get(tiered_cost_key))

        if tokens <= 128000 or tiered_cost <= 0:
            # Use base pricing for all tokens
            return tokens * base_cost
        else:
            # Use tiered pricing: first 128k at base rate, rest at tiered rate
            base_tokens = 128000
            tiered_tokens = tokens - 128000
            return (base_tokens * base_cost) + (tiered_tokens * tiered_cost)

    ## CALCULATE INPUT COST
    prompt_tokens = usage.prompt_tokens or 0
    prompt_cost = _calculate_tiered_cost(
        prompt_tokens, "input_cost_per_token", "input_cost_per_token_above_128k_tokens"
    )

    # Add cached input tokens cost if available
    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details:
        cache_hit_tokens = getattr(usage.prompt_tokens_details, "cache_hit_tokens", 0) or 0
        if cache_hit_tokens > 0:
            cache_read_cost = _safe_float_cast(model_info.get("cache_read_input_token_cost"))
            cache_read_cost_above_128k = _safe_float_cast(
                model_info.get("cache_read_input_token_cost_above_128k_tokens")
            )

            if cache_hit_tokens <= 128000 or cache_read_cost_above_128k <= 0:
                cache_cost = cache_hit_tokens * cache_read_cost
            else:
                # Tiered pricing for cached tokens
                base_cache_tokens = 128000
                tiered_cache_tokens = cache_hit_tokens - 128000
                cache_cost = (base_cache_tokens * cache_read_cost) + (tiered_cache_tokens * cache_read_cost_above_128k)

            prompt_cost += cache_cost

    ## CALCULATE OUTPUT COST
    # For XAI models, completion is billed as (visible completion tokens + reasoning tokens)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
    reasoning_tokens = 0
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = int(getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0)

    total_completion_tokens = completion_tokens + reasoning_tokens

    # Output tokens are billed at tiered rates if INPUT tokens are above 128k
    if prompt_tokens > 128000:
        # Use tiered pricing for ALL output tokens when input tokens > 128k
        output_cost_per_token_above_128k = _safe_float_cast(model_info.get("output_cost_per_token_above_128k_tokens"))
        if output_cost_per_token_above_128k > 0:
            completion_cost = total_completion_tokens * output_cost_per_token_above_128k
        else:
            # Fallback to base pricing if no tiered pricing available
            output_cost_per_token = _safe_float_cast(model_info.get("output_cost_per_token"))
            completion_cost = total_completion_tokens * output_cost_per_token
    else:
        # Use base pricing for output tokens when input tokens <= 128k
        output_cost_per_token = _safe_float_cast(model_info.get("output_cost_per_token"))
        completion_cost = total_completion_tokens * output_cost_per_token

    return prompt_cost, completion_cost
