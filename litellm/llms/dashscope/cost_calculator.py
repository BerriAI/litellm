"""
Cost calculator for Dashscope Chat models.

Alibaba Model Studio tiered pricing is all-or-nothing: the tier is picked from the
total input tokens of a single request, and every token of that request (input,
cached, cache-creation, output, reasoning) is billed at that one tier's rate.
See https://help.aliyun.com/zh/model-studio/billing-for-model-studio
"""

from dataclasses import dataclass, replace
from datetime import datetime
from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.tiered_pricing import select_tier_for_input, tier_rate
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    apply_off_peak_pricing,
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


@dataclass(frozen=True, slots=True)
class TokenRates:
    input_rate: float
    cache_read_rate: float
    cache_creation_rate: float
    output_rate: float
    reasoning_rate: float | None

    @property
    def billed_reasoning_rate(self) -> float:
        return self.output_rate if self.reasoning_rate is None else self.reasoning_rate


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


def _flat_rates(model_info: ModelInfo) -> TokenRates:
    reasoning_rate: Final = model_info.get("output_cost_per_reasoning_token")
    return TokenRates(
        input_rate=float(model_info.get("input_cost_per_token") or 0.0),
        cache_read_rate=_flat_rate(model_info, "cache_read_input_token_cost", "input_cost_per_token"),
        cache_creation_rate=_flat_rate(model_info, "cache_creation_input_token_cost", "input_cost_per_token"),
        output_rate=float(model_info.get("output_cost_per_token") or 0.0),
        reasoning_rate=None if reasoning_rate is None else float(reasoning_rate),
    )


def _tier_rates(model_info: ModelInfo, tier: dict) -> TokenRates:
    # A tier that declares output rates keeps the request on them, all-or-nothing. A tier table
    # spelling out only input rates would serve every completion for free, so there the model's
    # own output rates stand in
    flat_rates: Final = _flat_rates(model_info)
    tier_declares_output: Final = "output_cost_per_token" in tier
    tier_declares_reasoning: Final = "output_cost_per_reasoning_token" in tier
    return TokenRates(
        input_rate=tier_rate(tier, "input_cost_per_token"),
        cache_read_rate=tier_rate(tier, "cache_read_input_token_cost", "input_cost_per_token"),
        cache_creation_rate=tier_rate(tier, "cache_creation_input_token_cost", "input_cost_per_token"),
        output_rate=tier_rate(tier, "output_cost_per_token") if tier_declares_output else flat_rates.output_rate,
        reasoning_rate=(
            tier_rate(tier, "output_cost_per_reasoning_token")
            if tier_declares_reasoning
            else None
            if tier_declares_output
            else flat_rates.reasoning_rate
        ),
    )


def _off_peak_rates(model_info: ModelInfo, current_time: datetime | None, rates: TokenRates) -> TokenRates:
    input_rate, output_rate, cache_read_rate = apply_off_peak_pricing(
        model_info, current_time, rates.input_rate, rates.output_rate, rates.cache_read_rate
    )
    return replace(rates, input_rate=input_rate, output_rate=output_rate, cache_read_rate=cache_read_rate)


def _bill(breakdown: TokenBreakdown, rates: TokenRates) -> tuple[float, float]:
    prompt_cost: Final = (
        (breakdown.text_tokens * rates.input_rate)
        + (breakdown.cached_tokens * rates.cache_read_rate)
        + (breakdown.cache_creation_tokens * rates.cache_creation_rate)
    )
    completion_cost: Final = (breakdown.completion_tokens * rates.output_rate) + (
        breakdown.reasoning_tokens * rates.billed_reasoning_rate
    )
    return prompt_cost, completion_cost


def cost_per_token(
    model: str,
    usage: Usage,
    custom_llm_provider: str = "dashscope",
    current_time: datetime | None = None,
) -> tuple[float, float]:
    """
    Calculate cost per token for Dashscope models.

    Supports both tiered and flat pricing with cached and reasoning tokens, and swaps in the
    model's off_peak_pricing rates while one of its windows is open.

    Args:
        model: Model name without provider prefix
        usage: LiteLLM Usage block
        custom_llm_provider: The provider id the request resolved to; dashscope or one of its brand aliases
        current_time: The moment the request is billed at; defaults to now, UTC

    Returns:
        Tuple[float, float] - (prompt_cost_in_usd, completion_cost_in_usd)
    """
    model_info: Final = get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    breakdown: Final = _extract_token_breakdown(usage)
    raw_tiers: Final = model_info.get("tiered_pricing")
    tiered_pricing: Final = raw_tiers if isinstance(raw_tiers, list) else None
    tier: Final = (
        select_tier_for_input(tiered_pricing=tiered_pricing, input_tokens=breakdown.total_input_tokens)
        if tiered_pricing
        else None
    )
    standard_rates: Final = _flat_rates(model_info) if tier is None else _tier_rates(model_info, tier)
    rates: Final = _off_peak_rates(model_info, current_time, standard_rates)

    return _bill(breakdown, rates)
