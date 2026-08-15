"""
Cost calculator for Dashscope Chat models.

Handles tiered pricing and prompt caching scenarios.
"""

from dataclasses import dataclass
from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.tiered_pricing import (
    select_tier_for_input,
    tier_rate,
)
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

    text_tokens: Final = usage.prompt_tokens - cached_tokens

    reasoning_tokens = 0
    if (
        hasattr(usage, "completion_tokens_details")
        and usage.completion_tokens_details
        and hasattr(usage.completion_tokens_details, "reasoning_tokens")
    ):
        reasoning_tokens = usage.completion_tokens_details.reasoning_tokens or 0

    completion_tokens: Final = (usage.completion_tokens or 0) - reasoning_tokens

    return TokenBreakdown(text_tokens, cached_tokens, completion_tokens, reasoning_tokens)


def _calculate_prompt_cost(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tier: dict | None,
) -> float:
    """Calculate total prompt cost including cached tokens."""
    if tier is not None:
        input_rate: Final = tier_rate(tier, "input_cost_per_token")
        cache_rate: Final = tier_rate(tier, "cache_read_input_token_cost", "input_cost_per_token")
        return (breakdown.text_tokens * input_rate) + (breakdown.cached_tokens * cache_rate)

    input_cost: Final = float(model_info.get("input_cost_per_token") or 0.0)

    # For cache_cost, first try the specific key, then fall back to input_cost.
    cache_cost_val: Final = model_info.get("cache_read_input_token_cost")
    cache_cost: Final = input_cost if cache_cost_val is None else float(cache_cost_val)

    return (breakdown.text_tokens * input_cost) + (breakdown.cached_tokens * cache_cost)


def _calculate_completion_cost(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tier: dict | None,
) -> float:
    """Calculate total completion cost including reasoning tokens."""
    if tier is not None:
        output_rate: Final = tier_rate(tier, "output_cost_per_token")
        reasoning_rate: Final = tier_rate(tier, "output_cost_per_reasoning_token", "output_cost_per_token")
        return (breakdown.completion_tokens * output_rate) + (breakdown.reasoning_tokens * reasoning_rate)

    output_cost: Final = float(model_info.get("output_cost_per_token") or 0.0)

    # For reasoning_cost, first try the specific key, then fall back to output_cost.
    reasoning_cost_val: Final = model_info.get("output_cost_per_reasoning_token")
    reasoning_cost: Final = output_cost if reasoning_cost_val is None else float(reasoning_cost_val)

    return (breakdown.completion_tokens * output_cost) + (breakdown.reasoning_tokens * reasoning_cost)


def cost_per_token(model: str, usage: Usage) -> tuple[float, float]:
    """
    Calculate cost per token for Dashscope models.

    Supports both tiered and flat pricing with cached and reasoning tokens.

    Args:
        model: Model name without provider prefix
        usage: LiteLLM Usage block

    Returns:
        Tuple[float, float] - (prompt_cost_in_usd, completion_cost_in_usd)
    """
    model_info: Final = get_model_info(model=model, custom_llm_provider="dashscope")
    breakdown: Final = _extract_token_breakdown(usage)
    raw_tiered: Final = model_info.get("tiered_pricing")
    tiered_pricing: Final = raw_tiered if isinstance(raw_tiered, list) else None

    total_input_tokens: Final = breakdown.text_tokens + breakdown.cached_tokens
    tier: Final = select_tier_for_input(tiered_pricing, total_input_tokens) if tiered_pricing else None

    prompt_cost: Final = _calculate_prompt_cost(breakdown=breakdown, model_info=model_info, tier=tier)
    completion_cost: Final = _calculate_completion_cost(breakdown=breakdown, model_info=model_info, tier=tier)

    return prompt_cost, completion_cost
