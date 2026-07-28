"""
Cost calculator for Volcengine chat models.

Volcengine selects one pricing tier from the request's total input length and
charges every input/output token at that tier. This is not graduated pricing.
"""

from litellm.litellm_core_utils.llm_cost_calc.tiered_pricing import (
    select_tier_for_input,
    tier_rate,
)
from litellm.types.utils import ModelInfo, Usage
from litellm.utils import get_model_info


def _cached_prompt_tokens(usage: Usage) -> int:
    prompt_details = usage.prompt_tokens_details
    if prompt_details is None:
        return 0
    return int(getattr(prompt_details, "cached_tokens", 0) or 0)


def _select_pricing_tier(
    tiered_pricing: list[dict] | None,
    prompt_tokens: int,
) -> dict | None:
    if not tiered_pricing:
        return None

    # Output-only synthetic usage blocks have no input length. Use the first
    # tier instead of returning zero cost for their completion tokens.
    return select_tier_for_input(
        tiered_pricing=tiered_pricing,
        input_tokens=max(prompt_tokens, 1),
    )


def _output_rate(tier: dict, completion_tokens: int) -> float:
    """Read the output rate, including Seed 1.8's short-output discount."""
    if completion_tokens > 200 and tier.get("output_cost_per_token_above_200_tokens") is not None:
        return tier_rate(tier, "output_cost_per_token_above_200_tokens")
    return tier_rate(tier, "output_cost_per_token")


def cost_per_token(model: str, usage: Usage) -> tuple[float, float]:
    """
    Return ``(prompt_cost_usd, completion_cost_usd)`` for a Volcengine request.

    Tier selection is based on total prompt tokens. Cached prompt tokens still
    count toward the tier boundary, but use the cache-read rate when one is
    declared by the model.
    """
    model_info: ModelInfo = get_model_info(
        model=model,
        custom_llm_provider="volcengine",
    )

    prompt_tokens = int(usage.prompt_tokens or 0)
    completion_tokens = int(usage.completion_tokens or 0)
    cached_tokens = min(_cached_prompt_tokens(usage), prompt_tokens)
    uncached_tokens = prompt_tokens - cached_tokens

    raw_tiered_pricing = model_info.get("tiered_pricing")
    tiered_pricing = raw_tiered_pricing if isinstance(raw_tiered_pricing, list) else None
    tier = _select_pricing_tier(
        tiered_pricing=tiered_pricing,
        prompt_tokens=prompt_tokens,
    )

    if tier is not None:
        input_rate = tier_rate(tier, "input_cost_per_token")
        output_rate = _output_rate(tier=tier, completion_tokens=completion_tokens)
        cache_rate_value = model_info.get("cache_read_input_token_cost")
        cache_rate = float(cache_rate_value) if cache_rate_value is not None else input_rate
    else:
        input_rate = float(model_info.get("input_cost_per_token") or 0.0)
        output_rate = float(model_info.get("output_cost_per_token") or 0.0)
        cache_rate_value = model_info.get("cache_read_input_token_cost")
        cache_rate = float(cache_rate_value) if cache_rate_value is not None else input_rate

    prompt_cost = (uncached_tokens * input_rate) + (cached_tokens * cache_rate)
    completion_cost = completion_tokens * output_rate
    return prompt_cost, completion_cost
