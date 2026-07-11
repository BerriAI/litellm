"""
Cost calculator for Dashscope Chat models.

Handles tiered pricing and prompt caching scenarios.

Tiered pricing semantics
------------------------
Dashscope (Alibaba Bailian) uses **all-or-nothing** tiered pricing. The unit
price of a single request is determined by its total input-token count, and
*all* tokens of the request (input, cache hits, and output) are billed at that
tier's rate. This is documented in the official help center:

    https://help.aliyun.com/zh/model-studio/billing-for-model-studio

    "百炼部分模型实行阶梯计费。单价取决于单次请求的输入 Token 总量。
     该请求的所有 Token 均按对应阶梯的单价结算。
     例如，某模型设有两档计费区间：0 < Token ≤ 32K 和 32K < Token ≤ 128K。
     若输入 100K Token，因数值落在第二区间（32K < 100K ≤ 128K），
     所有 Token 均按第二档单价结算。"

i.e. a 100K input request priced under [0, 32K]=A, (32K, 128K]=B is billed at
B for the entire request — *not* at A for the first 32K plus B for the next
68K (which would be income-tax / graduated bracket logic).
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


def _select_tier_for_input(
    total_input_tokens: int,
    tiered_pricing: List[dict],
) -> Optional[dict]:
    """
    Return the single tier whose range contains ``total_input_tokens``.

    Per Dashscope rules the tier is selected by total input tokens of the
    request, and that tier's rates apply uniformly to text, cached and output
    tokens. If the request exceeds the highest declared range, the last tier
    is used.

    A tier matches when ``range_start < total_input_tokens <= range_end``, so
    a request of exactly ``range_end`` tokens falls into the lower tier — matching
    the official example ``0 < Token ≤ 32K``. A request with no input tokens
    returns ``None`` so the caller can fall back to flat pricing (the tier
    concept does not apply to an empty request).
    """
    if not tiered_pricing or total_input_tokens <= 0:
        return None

    sorted_tiers = sorted(tiered_pricing, key=lambda x: x.get("range", [0, 0])[0])

    for tier in sorted_tiers:
        tier_range = tier.get("range", [])
        if len(tier_range) != 2:
            continue
        range_start, range_end = tier_range
        if range_start < total_input_tokens <= range_end:
            return tier

    return sorted_tiers[-1]


def _tier_rate(
    tier: dict,
    cost_key: str,
    fallback_cost_key: Optional[str] = None,
) -> float:
    """Read ``cost_key`` from ``tier``, falling back to ``fallback_cost_key`` if absent."""
    rate = tier.get(cost_key)
    if rate is None and fallback_cost_key is not None:
        rate = tier.get(fallback_cost_key)
    return float(rate or 0.0)


def _calculate_prompt_cost(
    breakdown: TokenBreakdown,
    model_info: ModelInfo,
    tiered_pricing: Optional[List[dict]],
) -> float:
    """Calculate total prompt cost including cached tokens."""
    if tiered_pricing:
        total_input = breakdown.text_tokens + breakdown.cached_tokens
        tier = _select_tier_for_input(total_input, tiered_pricing)
        if tier is not None:
            input_rate = _tier_rate(tier, "input_cost_per_token")
            cache_rate = _tier_rate(
                tier, "cache_read_input_token_cost", "input_cost_per_token"
            )
            return (
                breakdown.text_tokens * input_rate
                + breakdown.cached_tokens * cache_rate
            )

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
    """
    Calculate total completion cost including reasoning tokens.

    Tier selection is based on *input* tokens, per Dashscope rules — output
    tokens do not affect which tier is used. This is consistent with the
    examples in the official documentation, which only ever reference input
    Token counts when choosing a tier.
    """
    if tiered_pricing:
        total_input = breakdown.text_tokens + breakdown.cached_tokens
        tier = _select_tier_for_input(total_input, tiered_pricing)
        if tier is not None:
            output_rate = _tier_rate(tier, "output_cost_per_token")
            reasoning_rate = _tier_rate(
                tier, "output_cost_per_reasoning_token", "output_cost_per_token"
            )
            return (
                breakdown.completion_tokens * output_rate
                + breakdown.reasoning_tokens * reasoning_rate
            )

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
