"""
Helper util for handling XAI-specific cost calculation
- Prefers the cost xAI reports on the response over recomputing it locally
- Uses the generic cost calculator which already handles tiered pricing correctly
- Handles XAI-specific reasoning token billing (billed as part of completion tokens)
"""

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token
from litellm.types.utils import PromptTokensDetailsWrapper, Usage

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo

# https://docs.x.ai/developers/pricing#tools-pricing — default when unset in model map
_DEFAULT_WEB_SEARCH_COST_PER_CALL: Final = 5.0 / 1000.0


def apply_server_side_tool_usage_details_to_usage(usage: Usage, details: Mapping[str, object] | None) -> None:
    """
    Attach server_side_tool_usage_details and mirror web_search_calls onto
    prompt_tokens_details.web_search_requests for built-in tool cost gating.
    """
    if details is None:
        return
    usage.server_side_tool_usage_details = details  # pyright: ignore[reportAttributeAccessIssue]  # extra  # rebind-ok: extras
    try:
        web_search_calls: Final = int(details.get("web_search_calls") or 0)
    except (TypeError, ValueError):
        return
    if web_search_calls <= 0:
        return
    prompt_tokens_details: Final = usage.prompt_tokens_details or PromptTokensDetailsWrapper()
    prompt_tokens_details.web_search_requests = web_search_calls
    usage.prompt_tokens_details = prompt_tokens_details  # rebind-ok: write details onto caller usage


def _cost_reported_by_xai(usage: "Usage") -> float | None:
    """
    Return what xAI billed for the request in USD, or None if it reported nothing usable.

    The xAI transformations restate ``usage.cost_in_usd_ticks`` as ``usage.cost``, the
    field litellm already carries a provider stated cost in and the same one
    ``llms/perplexity/cost_calculator.py`` bills from. That figure is the total for the
    whole request, tokens and every server side tool invocation together, so nothing may
    be added on top of it.

    A negative amount is refused rather than billed: a caller who can point litellm at an
    api_base they control also controls the response body, and a negative cost would
    subtract from their own recorded spend. Those requests are priced from tokens instead.

    NaN and the infinities are refused for the same reason and are the worse case, because
    ``Usage`` stores a provider supplied ``cost`` without validating it and NaN compares
    false against every budget threshold. Billing one would disable spend enforcement for
    the key rather than mispricing a single request.
    """
    reported_cost: Final[object] = getattr(usage, "cost", None)
    if not isinstance(reported_cost, (int, float)) or isinstance(reported_cost, bool):
        return None
    if not math.isfinite(reported_cost):
        return None
    if reported_cost < 0:
        return None
    return float(reported_cost)


def cost_per_token(model: str, usage: Usage) -> tuple[float, float]:
    """
    Prefers the amount xAI reported for the request, matching how the perplexity
    calculator treats a provider-stated cost. That total is returned as completion
    cost because xAI does not break it down by direction. Without one, falls back to
    the generic cost calculator for all pricing logic, with XAI-specific reasoning
    token handling.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing XAI-specific usage information

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    reported_cost: Final = _cost_reported_by_xai(usage)
    if reported_cost is not None:
        return 0.0, reported_cost

    # XAI-specific completion cost: completion is billed as visible + reasoning
    # tokens. Detect when the transformation layer already folded them so we
    # don't double-count; fall back to raw xAI shape for callers that bypass
    # the transformation (e.g. proxy logs replayed into cost calc).
    prompt_tokens: Final = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens: Final = int(getattr(usage, "completion_tokens", 0) or 0)
    total_tokens: Final = int(getattr(usage, "total_tokens", 0) or 0)
    reasoning_tokens: Final = (
        int(getattr(usage.completion_tokens_details, "reasoning_tokens", 0) or 0)
        if hasattr(usage, "completion_tokens_details") and usage.completion_tokens_details
        else 0
    )

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


def _web_search_cost_per_call_from_model_info(model_info: "ModelInfo") -> float:
    """
    Per-invocation web_search price from model_info when configured.

    Prefer ``search_context_cost_per_query`` (same shape as Gemini/Anthropic web
    search pricing in the model cost map). Fall back to current xAI list pricing.
    """
    search_costs: Final = model_info.get("search_context_cost_per_query")
    if not isinstance(search_costs, Mapping):
        return _DEFAULT_WEB_SEARCH_COST_PER_CALL
    for key in (
        "search_context_size_medium",
        "search_context_size_low",
        "search_context_size_high",
    ):
        value = search_costs.get(key)
        if value is None:
            continue
        try:
            cost = float(value)
        except (TypeError, ValueError):
            continue
        if cost > 0:
            return cost
    return _DEFAULT_WEB_SEARCH_COST_PER_CALL


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    Calculate the cost of web search requests for X.AI models.

    When xAI reports what it billed, that figure already covers the server-side
    search calls and ``cost_per_token`` has returned it, so there is nothing to add
    here. Otherwise price the invocations from
    usage.server_side_tool_usage_details.web_search_calls at the per-call rate
    (model_info.search_context_cost_per_query when set, else the default $5 / 1k).
    """
    if _cost_reported_by_xai(usage) is not None:
        return 0.0

    details: Final = getattr(usage, "server_side_tool_usage_details", None)
    if not isinstance(details, Mapping):
        return 0.0
    try:
        web_search_calls: Final = int(details.get("web_search_calls") or 0)
    except (TypeError, ValueError):
        return 0.0
    if web_search_calls <= 0:
        return 0.0
    return _web_search_cost_per_call_from_model_info(model_info) * web_search_calls
