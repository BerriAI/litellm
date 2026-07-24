"""
Helper util for handling anthropic-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Optional, Tuple

from pydantic import BaseModel, ValidationError

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _get_web_search_requests,
    generic_cost_per_token,
)

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage
import litellm

_UNPRICED_INFERENCE_GEOS = frozenset({"global", "not_available"})


def _pricing_modifier_multiplier(model: str, usage: "Usage") -> float:
    """
    Resolve the combined geo and speed multiplier for a request.

    Anthropic stacks these modifiers on top of the standard rates, and prompt
    caching multipliers apply on top of the modified rates rather than the
    unmodified ones, so a fast-mode cache read on Claude Opus 5 bills at
    0.1 x $10/MTok, not 0.1 x $5/MTok. The caller therefore scales the whole
    prompt cost, cache tokens included; see
    https://platform.claude.com/docs/en/build-with-claude/fast-mode#pricing

    Returns 1.0 when the model is unknown or carries no modifier entry.
    """
    try:
        model_info = litellm.get_model_info(model=model, custom_llm_provider="anthropic")
    except Exception:
        return 1.0

    modifiers = model_info.get("provider_specific_entry") or {}
    if not isinstance(modifiers, dict):
        return 1.0

    inference_geo = getattr(usage, "inference_geo", None)
    geo_multiplier = (
        modifiers.get(inference_geo.lower(), 1.0)
        if isinstance(inference_geo, str) and inference_geo.lower() not in _UNPRICED_INFERENCE_GEOS
        else 1.0
    )
    speed_multiplier = modifiers.get("fast", 1.0) if getattr(usage, "speed", None) == "fast" else 1.0

    return float(geo_multiplier) * float(speed_multiplier)


def cost_per_token(model: str, usage: "Usage", service_tier: str | None = None) -> Tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information
        - service_tier: the service tier the request was served at (e.g. "priority"),
          read from the Anthropic response usage and used to select tier-specific pricing

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="anthropic",
        service_tier=service_tier,
    )

    multiplier = _pricing_modifier_multiplier(model=model, usage=usage)
    if multiplier == 1.0:
        return prompt_cost, completion_cost

    return prompt_cost * multiplier, completion_cost * multiplier


class _AnthropicServerToolUseProbe(BaseModel):
    web_search_requests: int | None = None


class _AnthropicUsageProbe(BaseModel):
    server_tool_use: _AnthropicServerToolUseProbe | None = None


class _AnthropicResponseProbe(BaseModel):
    usage: _AnthropicUsageProbe | None = None


def get_anthropic_web_search_requests_from_response(
    response_object: object,
) -> int | None:
    """Read usage.server_tool_use.web_search_requests from a raw Anthropic
    /v1/messages response dict, returning None when absent."""
    if not isinstance(response_object, dict):
        return None
    try:
        probe = _AnthropicResponseProbe.model_validate(response_object)
    except ValidationError:
        return None
    if probe.usage is None or probe.usage.server_tool_use is None:
        return None
    return probe.usage.server_tool_use.web_search_requests


def get_cost_for_anthropic_web_search(
    model_info: Optional["ModelInfo"] = None,
    usage: Optional["Usage"] = None,
) -> float:
    """
    Get the cost of using a web search tool for Anthropic.
    """
    from litellm.types.utils import SearchContextCostPerQuery

    ## Check if web search requests are in the usage object
    if model_info is None:
        return 0.0

    if usage is None:
        return 0.0
    web_search_requests = _get_web_search_requests(getattr(usage, "server_tool_use", None))
    if web_search_requests is None:
        return 0.0

    ## Get the cost per web search request
    search_context_pricing: SearchContextCostPerQuery = (
        model_info.get("search_context_cost_per_query") or SearchContextCostPerQuery()
    )
    cost_per_web_search_request = search_context_pricing.get("search_context_size_medium", 0.0)
    if cost_per_web_search_request is None or cost_per_web_search_request == 0.0:
        return 0.0

    ## Calculate the total cost
    total_cost = cost_per_web_search_request * web_search_requests
    return total_cost
