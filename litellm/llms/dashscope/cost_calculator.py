"""
Cost calculator for Dashscope Chat models.

Alibaba Model Studio tiered pricing is all-or-nothing: the tier is picked from the
total input tokens of a single request, and every token of that request (input,
cached, cache-creation, output, reasoning) is billed at that one tier's rate.
See https://help.aliyun.com/zh/model-studio/billing-for-model-studio
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    billed_cache_read_rate,
    call_context,
    parse_completion_tokens_details,
    parse_prompt_tokens_details,
)
from litellm.rust_bridge._native import NativePrices
from litellm.types.utils import Usage
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
    quote: Final = NativePrices(model_info, custom_llm_provider).quote(
        breakdown.total_input_tokens, call_context(None, None, None, current_time)
    )
    prompt_cost: Final = (
        (breakdown.text_tokens * quote.input)
        + (breakdown.cached_tokens * billed_cache_read_rate(quote, usage))
        + (breakdown.cache_creation_tokens * quote.cache_write)
    )
    completion_cost: Final = (breakdown.completion_tokens * quote.output) + (
        breakdown.reasoning_tokens * quote.reasoning
    )
    return prompt_cost, completion_cost
