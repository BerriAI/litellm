"""
Cost calculator for Dashscope Chat models.

Handles tiered pricing and prompt caching scenarios.

Alibaba Model Studio tiered pricing is step pricing: the total input token count of a
request selects one tier, and every token of that request (plain input, cached input,
completion, reasoning) is billed at that tier's rates.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

from litellm.litellm_core_utils.llm_cost_calc.tiered_pricing import select_tier_for_input, tier_rate
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


@dataclass(frozen=True, slots=True)
class TokenRates:
    """Per-token rates applied to a single request."""

    input: float
    cached_input: float
    output: float
    reasoning_output: float


def _rates_from_tier(tier: dict) -> TokenRates:
    return TokenRates(
        input=tier_rate(tier, "input_cost_per_token"),
        cached_input=tier_rate(tier, "cache_read_input_token_cost", "input_cost_per_token"),
        output=tier_rate(tier, "output_cost_per_token"),
        reasoning_output=tier_rate(tier, "output_cost_per_reasoning_token", "output_cost_per_token"),
    )


def _rates_from_flat_pricing(model_info: ModelInfo) -> TokenRates:
    input_cost = float(model_info.get("input_cost_per_token") or 0.0)
    output_cost = float(model_info.get("output_cost_per_token") or 0.0)
    cache_cost_val = model_info.get("cache_read_input_token_cost")
    reasoning_cost_val = model_info.get("output_cost_per_reasoning_token")

    return TokenRates(
        input=input_cost,
        cached_input=input_cost if cache_cost_val is None else float(cache_cost_val),
        output=output_cost,
        reasoning_output=output_cost if reasoning_cost_val is None else float(reasoning_cost_val),
    )


def _select_rates(model_info: ModelInfo, input_tokens: int) -> TokenRates:
    tiered_pricing: Optional[List[dict]] = (
        model_info.get("tiered_pricing") if isinstance(model_info.get("tiered_pricing"), list) else None
    )
    if tiered_pricing:
        tier = select_tier_for_input(tiered_pricing=tiered_pricing, input_tokens=input_tokens)
        if tier is not None:
            return _rates_from_tier(tier)

    return _rates_from_flat_pricing(model_info)


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
    rates = _select_rates(model_info=model_info, input_tokens=usage.prompt_tokens or 0)

    prompt_cost = (breakdown.text_tokens * rates.input) + (breakdown.cached_tokens * rates.cached_input)
    completion_cost = (breakdown.completion_tokens * rates.output) + (
        breakdown.reasoning_tokens * rates.reasoning_output
    )

    return prompt_cost, completion_cost
