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


def web_search_cost_per_call_from_model_info(model_info: ModelInfo) -> float:
    """
    The price per xAI web search call, priced in Rust: the first positive ``search_context_cost_per_query`` price of
    medium, low and high, else xAI's list price.
    """
    from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import price_web_search_dispatch

    one_call: Final = Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0)
    one_call.server_side_tool_usage_details = {"web_search_calls": 1}  # pyright: ignore[reportAttributeAccessIssue]  # extra
    return price_web_search_dispatch("xai", one_call, model_info) or 0.0


def cost_per_web_search_request(usage: Usage, model_info: ModelInfo) -> float:
    """
    The cost of xAI web search calls, priced in Rust: each call in usage.server_side_tool_usage_details.web_search_calls
    at the row's price per call or xAI's list price, nothing when xAI states the call's cost.
    """
    from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import price_web_search_dispatch

    return price_web_search_dispatch("xai", usage, model_info) or 0.0


_web_search_cost_per_call_from_model_info: Final = web_search_cost_per_call_from_model_info
