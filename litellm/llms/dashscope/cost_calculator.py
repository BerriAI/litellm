"""
Cost calculator for Dashscope Chat models. 

Handles tiered pricing and prompt caching scenarios.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


@dataclass
class TokenBreakdown:
    """Token breakdown for cost calculation."""

    text_tokens: int
    cached_tokens: int
    completion_tokens: int
    reasoning_tokens: int


def _extract_token_breakdown(usage: Usage) -> TokenBreakdown:
    """Extract token counts from usage, handling cached and reasoning tokens."""
    cached_tokens = 0
    if usage.prompt_tokens_details and hasattr(
        usage.prompt_tokens_details, "cached_tokens"
    ):
        cached_tokens = usage.prompt_tokens_details.cached_tokens or 0

    text_tokens = usage.prompt_tokens - cached_tokens

    reasoning_tokens = 0
    if (
        hasattr(usage, "completion_tokens_details")
        and usage.completion_tokens_details
        and hasattr(usage.completion_tokens_details, "reasoning_tokens")
    ):
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0

    completion_tokens = (usage.completion_tokens or 0) - reasoning_tokens

    return TokenBreakdown(
        text_tokens, cached_tokens, completion_tokens, reasoning_tokens
    )


def _calculate_tiered_cost(
    tokens: int,
    tiered_pricing: List[dict],
    cost_key: str,
    fallback_cost_key: Optional[str] = None,
) -> float:
    """
    Calculate cost for a given number of tokens based on a true tiered pricing structure.

    This function iterates through sorted pricing tiers, calculates the cost for the
    number of tokens that fall into each tier's range, and sums them up to get the total cost.

    Args:
        tokens (int): The total number of tokens to calculate the cost for.
        tiered_pricing (List[dict]): A list of dictionaries, where each dictionary
            represents a pricing tier.
        cost_key (str): The key in the tier dictionary that holds the per-token cost
            (e.g., 'input_cost_per_token').
        fallback_cost_key (Optional[str], optional): A fallback key to use if the
            primary `cost_key` is not found in a tier. Defaults to None.

    Returns:
        float: The total calculated cost for the given tokens.

    Example:
        >>> tiered_pricing = [
        ...     {"range": [0, 100000], "input_cost_per_token": 0.0001},
        ...     {"range": [100000, 500000], "input_cost_per_token": 0.00005},
        ... ]

        Calculating cost for 150,000 tokens:
        (100,000 * 0.0001) + (50,000 * 0.00005) = $12.5
    """
    if not tiered_pricing or tokens <= 0:
        return 0.0

    total_cost = 0.0
    tokens_processed = 0

    sorted_tiers = sorted(tiered_pricing, key=lambda x: x.get("range", [0, 0])[0])

    for tier in sorted_tiers:
        if tokens_processed >= tokens:
            break

        tier_range = tier.get("range", [])
        if len(tier_range) != 2:
            continue

        range_start, range_end = tier_range

        if tokens <= range_start:
            continue

        tier_start = max(range_start, tokens_processed)
        tier_end = min(range_end, tokens)

        if tier_end > tier_start:
            tokens_in_tier = tier_end - tier_start
            cost_per_token = tier.get(cost_key) or tier.get(fallback_cost_key, 0)
            total_cost += tokens_in_tier * cost_per_token
            tokens_processed = tier_end

    # After loop, check if any tokens remain (i.e., tokens > highest tier's end range)
    # and charge them at the last tier's rate.
    if tokens_processed < tokens and sorted_tiers:
        last_tier = sorted_tiers[-1]
        remaining_tokens = tokens - tokens_processed
        cost_per_token = last_tier.get(cost_key) or last_tier.get(fallback_cost_key, 0)
        total_cost += remaining_tokens * cost_per_token

    return total_cost


def _calculate_prompt_cost(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tiered_pricing: Optional[List[dict]],
) -> float:
    """Calculate total prompt cost including cached tokens."""
    if tiered_pricing:
        text_cost = _calculate_tiered_cost(
            tokens=breakdown.text_tokens,
            tiered_pricing=tiered_pricing,
            cost_key="input_cost_per_token",
        )
        cache_cost = _calculate_tiered_cost(
            tokens=breakdown.cached_tokens,
            tiered_pricing=tiered_pricing,
            cost_key="cache_read_input_token_cost",
            fallback_cost_key="input_cost_per_token",
        )
        return text_cost + cache_cost

    input_cost = float(model_info.get("input_cost_per_token") or 0.0)

    # For cache_cost, first try the specific key, then fall back to input_cost.
    cache_cost_val = model_info.get("cache_read_input_token_cost")
    if cache_cost_val is None:
        cache_cost = input_cost
    else:
        cache_cost = float(cache_cost_val)

    return (breakdown.text_tokens * input_cost) + (breakdown.cached_tokens * cache_cost)


def _calculate_completion_cost(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tiered_pricing: Optional[List[dict]],
) -> float:
    """Calculate total completion cost including reasoning tokens."""
    if tiered_pricing:
        completion_cost = _calculate_tiered_cost(
            tokens=breakdown.completion_tokens,
            tiered_pricing=tiered_pricing,
            cost_key="output_cost_per_token",
        )
        reasoning_cost = _calculate_tiered_cost(
            tokens=breakdown.reasoning_tokens,
            tiered_pricing=tiered_pricing,
            cost_key="output_cost_per_reasoning_token",
            fallback_cost_key="output_cost_per_token",
        )
        return completion_cost + reasoning_cost

    output_cost = float(model_info.get("output_cost_per_token") or 0.0)

    # For reasoning_cost, first try the specific key, then fall back to output_cost.
    reasoning_cost_val = model_info.get("output_cost_per_reasoning_token")
    if reasoning_cost_val is None:
        reasoning_cost = output_cost
    else:
        reasoning_cost = float(reasoning_cost_val)

    return (breakdown.completion_tokens * output_cost) + (
        breakdown.reasoning_tokens * reasoning_cost
    )


def cost_per_token(model: str, usage: Usage) -> Tuple[float, float]:
    """
    Calculate cost per token for Dashscope models.

    Supports both tiered and flat pricing with cached and reasoning tokens.

    Args:
        model: Model name without provider prefix
        usage: LiteLLM Usage block

    Returns:
        Tuple[float, float] - (prompt_cost_in_usd, completion_cost_in_usd)
    """
    model_info = get_model_info(model=model, custom_llm_provider="dashscope")
    breakdown = _extract_token_breakdown(usage)
    tiered_pricing = (
        model_info.get("tiered_pricing")
        if isinstance(model_info.get("tiered_pricing"), list)
        else None
    )

    prompt_cost = _calculate_prompt_cost(
        breakdown=breakdown, model_info=model_info, tiered_pricing=tiered_pricing
    )
    completion_cost = _calculate_completion_cost(
        breakdown=breakdown, model_info=model_info, tiered_pricing=tiered_pricing
    )

    return prompt_cost, completion_cost
