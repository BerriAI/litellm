"""
Cost calculator for Dashscope Chat models.

Handles tiered pricing and prompt caching scenarios.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from litellm.litellm_core_utils.llm_cost_calc.tiered_pricing import calculate_tiered_cost
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
    if usage.prompt_tokens_details and hasattr(usage.prompt_tokens_details, "cached_tokens"):
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

    return TokenBreakdown(text_tokens, cached_tokens, completion_tokens, reasoning_tokens)


def _calculate_prompt_cost(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tiered_pricing: Optional[List[dict]],
) -> float:
    """Calculate total prompt cost including cached tokens."""
    if tiered_pricing:
        text_cost = calculate_tiered_cost(
            tokens=breakdown.text_tokens,
            tiered_pricing=tiered_pricing,
            cost_key="input_cost_per_token",
        )
        cache_cost = calculate_tiered_cost(
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
        completion_cost = calculate_tiered_cost(
            tokens=breakdown.completion_tokens,
            tiered_pricing=tiered_pricing,
            cost_key="output_cost_per_token",
        )
        reasoning_cost = calculate_tiered_cost(
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

    return (breakdown.completion_tokens * output_cost) + (breakdown.reasoning_tokens * reasoning_cost)


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
    tiered_pricing = model_info.get("tiered_pricing") if isinstance(model_info.get("tiered_pricing"), list) else None

    prompt_cost = _calculate_prompt_cost(breakdown=breakdown, model_info=model_info, tiered_pricing=tiered_pricing)
    completion_cost = _calculate_completion_cost(
        breakdown=breakdown, model_info=model_info, tiered_pricing=tiered_pricing
    )

    return prompt_cost, completion_cost
