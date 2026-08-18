"""
Cost calculator for Dashscope Chat models.

Alibaba Model Studio tiered pricing is all-or-nothing: the tier is picked from the
total input tokens of a single request, and every token of that request (input,
cached, cache-creation, output, reasoning) is billed at that one tier's rate.
See https://help.aliyun.com/zh/model-studio/billing-for-model-studio
"""

from dataclasses import dataclass
from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.tiered_pricing import select_tier_for_input, tier_rate
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    parse_completion_tokens_details,
    parse_prompt_tokens_details,
)
from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


@dataclass(frozen=True, slots=True)
class TokenBreakdown:
    text_tokens: int
    cached_tokens: int
    cache_creation_tokens: int
    completion_tokens: int
    reasoning_tokens: int

    @property
    def total_input_tokens(self) -> int:
        return self.text_tokens + self.cached_tokens + self.cache_creation_tokens


def _extract_token_breakdown(usage: Usage) -> TokenBreakdown:
    prompt_details: Final = parse_prompt_tokens_details(usage)
    cached_tokens: Final = prompt_details["cache_hit_tokens"]
    cache_creation_tokens: Final = prompt_details["cache_creation_tokens"]
    text_tokens: Final = max(usage.prompt_tokens - cached_tokens - cache_creation_tokens, 0)

    reasoning_tokens: Final = parse_completion_tokens_details(usage)["reasoning_tokens"]
    completion_tokens: Final = max((usage.completion_tokens or 0) - reasoning_tokens, 0)

    return TokenBreakdown(
        text_tokens=text_tokens,
        cached_tokens=cached_tokens,
        cache_creation_tokens=cache_creation_tokens,
        completion_tokens=completion_tokens,
        reasoning_tokens=reasoning_tokens,
    )


def _flat_rate(model_info: ModelInfo, cost_key: str, fallback_cost_key: str) -> float:
    value: Final = model_info.get(cost_key)
    if value is None:
        return float(model_info.get(fallback_cost_key) or 0.0)
    return float(value)


def _calculate_prompt_cost(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tier: dict | None,
) -> float:
    if tier is not None:
        return (
            (breakdown.text_tokens * tier_rate(tier, "input_cost_per_token"))
            + (breakdown.cached_tokens * tier_rate(tier, "cache_read_input_token_cost", "input_cost_per_token"))
            + (
                breakdown.cache_creation_tokens
                * tier_rate(tier, "cache_creation_input_token_cost", "input_cost_per_token")
            )
        )

    input_cost: Final = float(model_info.get("input_cost_per_token") or 0.0)
    cache_read_cost: Final = _flat_rate(model_info, "cache_read_input_token_cost", "input_cost_per_token")
    cache_creation_cost: Final = _flat_rate(model_info, "cache_creation_input_token_cost", "input_cost_per_token")

    return (
        (breakdown.text_tokens * input_cost)
        + (breakdown.cached_tokens * cache_read_cost)
        + (breakdown.cache_creation_tokens * cache_creation_cost)
    )


def _calculate_completion_cost(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tier: dict | None,
) -> float:
    # A tier that declares output rates keeps the request on them, all-or-nothing. A tier table
    # spelling out only input rates would serve every completion for free, so there the model's
    # own output rates stand in
    tier_declares_output: Final = tier is not None and "output_cost_per_token" in tier
    output_cost: Final = (
        tier_rate(tier, "output_cost_per_token")
        if tier_declares_output
        else float(model_info.get("output_cost_per_token") or 0.0)
    )
    tier_declares_reasoning: Final = tier is not None and "output_cost_per_reasoning_token" in tier
    model_reasoning_rate: Final = None if tier_declares_output else model_info.get("output_cost_per_reasoning_token")
    reasoning_cost: Final = (
        tier_rate(tier, "output_cost_per_reasoning_token", "output_cost_per_token")
        if tier_declares_reasoning
        else float(model_reasoning_rate)
        if model_reasoning_rate is not None
        else output_cost
    )

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
    raw_tiers: Final = model_info.get("tiered_pricing")
    tiered_pricing: Final = raw_tiers if isinstance(raw_tiers, list) else None
    tier: Final = (
        select_tier_for_input(tiered_pricing=tiered_pricing, input_tokens=breakdown.total_input_tokens)
        if tiered_pricing
        else None
    )

    prompt_cost: Final = _calculate_prompt_cost(breakdown=breakdown, model_info=model_info, tier=tier)
    completion_cost: Final = _calculate_completion_cost(breakdown=breakdown, model_info=model_info, tier=tier)

    return prompt_cost, completion_cost
