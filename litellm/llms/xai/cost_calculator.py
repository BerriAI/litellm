"""
Helper util for handling XAI-specific cost calculation
- Prefers the cost xAI reports on the response over recomputing it locally
- Handles XAI-specific reasoning token billing (billed as part of completion tokens)
"""

import math
from collections.abc import Mapping
from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import BranchFacts, ReportedCost, branch_cost
from litellm.types.utils import ModelInfo, PromptTokensDetailsWrapper, Usage
from litellm.utils import get_model_info

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


def stated_cost(usage: Usage) -> float | None:
    reported_cost: Final[object] = getattr(usage, "cost", None)
    if not isinstance(reported_cost, (int, float)) or isinstance(reported_cost, bool):
        return None
    if not math.isfinite(reported_cost):
        return None
    if reported_cost < 0:
        return None
    return float(reported_cost)


def pricing_row(model: str, stated: bool) -> ModelInfo | None:
    """The row xAI's token math reads, ``None`` for an unmapped model whose reported cost bills it whole."""
    try:
        return get_model_info(model=model, custom_llm_provider="xai")
    except Exception:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
        if not stated:
            raise
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
    reported_cost: Final = stated_cost(usage)
    return branch_cost(
        model,
        "xai",
        usage,
        pricing_row(model, stated=reported_cost is not None),
        branch=BranchFacts(
            reported=None if reported_cost is None else ReportedCost(source="usage.cost", usd=reported_cost)
        ),
    )


def _web_search_cost_per_call_from_model_info(model_info: ModelInfo) -> float:
    """
    Per-invocation web_search price from model_info when configured.

    Prefer ``search_context_cost_per_query`` (same shape as Gemini/Anthropic web
    search pricing in the model cost map). Fall back to current xAI list pricing.
    """
    search_costs: Final = model_info.get("search_context_cost_per_query")
    if not isinstance(search_costs, Mapping):
        return _DEFAULT_WEB_SEARCH_COST_PER_CALL
    for value in (
        search_costs.get("search_context_size_medium"),
        search_costs.get("search_context_size_low"),
        search_costs.get("search_context_size_high"),
    ):
        if value is None:
            continue
        try:
            cost = float(value)
        except (TypeError, ValueError):
            continue
        if cost > 0:
            return cost
    return _DEFAULT_WEB_SEARCH_COST_PER_CALL


def cost_per_web_search_request(usage: Usage, model_info: ModelInfo) -> float:
    """
    Calculate the cost of web search requests for X.AI models.

    Counts invocations from usage.server_side_tool_usage_details.web_search_calls.
    Per-call rate comes from model_info.search_context_cost_per_query when set,
    otherwise the default xAI tools rate ($5 / 1k calls).
    """
    if stated_cost(usage) is not None:
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
