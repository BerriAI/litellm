"""
Helper util for handling XAI-specific cost calculation
- Prefers the cost xAI reports on the response over recomputing it locally
- Uses the generic cost calculator which already handles tiered pricing correctly
- Handles XAI-specific reasoning token billing (billed as part of completion tokens)
"""

from typing import TYPE_CHECKING, Final

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import Usage

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo

# xAI states what it billed in "USD ticks", where 1 USD = 10^10 ticks.
# https://docs.x.ai/developers/cost-tracking
USD_TICKS_PER_DOLLAR: Final = 10_000_000_000


def _cost_reported_by_xai(usage: "Usage") -> float | None:
    """
    Return what xAI billed for the request in USD, or None if it reported nothing.

    ``usage.cost_in_usd_ticks`` is the total for the whole request -- tokens *and*
    every server-side tool invocation -- so whoever consumes it must not add
    anything on top.
    """
    # Documented as an integer, but this rides in on an untyped extra field, so the
    # conversion below stays defensive about what actually arrives.
    ticks: Final[int | None] = getattr(usage, "cost_in_usd_ticks", None)
    if ticks is None:
        return None
    try:
        return int(ticks) / USD_TICKS_PER_DOLLAR
    except (TypeError, ValueError):
        # A malformed value is not worth failing a request over; fall back to
        # calculating the cost locally.
        return None


def cost_per_token(model: str, usage: Usage) -> tuple[float, float]:
    """
    Calculates the cost per token for a given XAI model, prompt tokens, and completion tokens.
    Uses the generic cost calculator for all pricing logic, with XAI-specific reasoning token handling.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing XAI-specific usage information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    ## USE THE COST XAI REPORTED, IF AVAILABLE
    ## Same shape as the perplexity calculator: when the provider states what it
    ## charged, that number is authoritative and the local token math is skipped.
    ## It is returned as completion cost because xAI does not break the total down
    ## by direction.
    reported_cost: Final = _cost_reported_by_xai(usage)
    if reported_cost is not None:
        return 0.0, reported_cost

    ## FALLBACK: calculate the cost from tokens
    # XAI-specific completion cost: completion is billed as visible + reasoning
    # tokens. Detect when the transformation layer already folded them so we
    # don't double-count; fall back to raw xAI shape for callers that bypass
    # the transformation (e.g. proxy logs replayed into cost calc).
    prompt_tokens: Final = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens: Final = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens: Final = int(getattr(usage, "total_tokens", 0) or 0)
    reasoning_tokens = 0
    if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details:
        reasoning_tokens = int(getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0)

    already_normalised: Final = total_tokens == prompt_tokens + completion_tokens
    total_completion_tokens: Final = completion_tokens if already_normalised else completion_tokens + reasoning_tokens

    modified_usage: Final = Usage(
        prompt_tokens=usage.prompt_tokens,
        completion_tokens=total_completion_tokens,
        total_tokens=usage.total_tokens,
        prompt_tokens_details=usage.prompt_tokens_details,
        completion_tokens_details=None,
    )

    prompt_cost, completion_cost = generic_cost_per_token(model=model, usage=modified_usage, custom_llm_provider="xai")

    return prompt_cost, completion_cost


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    Calculate the cost of web search requests for X.AI models.

    When xAI reports what it billed, that figure already covers the server-side
    search calls and ``cost_per_token`` has returned it, so there is nothing to add
    here.

    Otherwise fall back to the legacy Live Search rate of $25 per 1,000 sources
    used ($0.025 each). The number of sources is stored in
    prompt_tokens_details.web_search_requests by the transformation layer to be
    compatible with the existing detection system.
    """
    # Charging a per-search fee on top of a total xAI already gave us would bill
    # the search calls twice.
    if _cost_reported_by_xai(usage) is not None:
        return 0.0

    # Cost per source used: $25 per 1,000 sources = $0.025 per source
    cost_per_source: Final = 25.0 / 1000.0  # $0.025

    num_sources_used = 0

    if (
        hasattr(usage, "prompt_tokens_details")
        and usage.prompt_tokens_details is not None
        and hasattr(usage.prompt_tokens_details, "web_search_requests")
        and usage.prompt_tokens_details.web_search_requests is not None
    ):
        num_sources_used = int(usage.prompt_tokens_details.web_search_requests)

    # Fallback: try to get from num_sources_used if set directly
    elif hasattr(usage, "num_sources_used") and usage.num_sources_used is not None:
        num_sources_used = int(usage.num_sources_used)

    total_cost: Final = cost_per_source * num_sources_used

    return total_cost
