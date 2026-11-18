"""
Helper util for handling anthropic-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Final, Optional

from pydantic import ValidationError

import litellm
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import BranchFacts, Elapsed, branch_cost
from litellm.litellm_core_utils.llm_cost_calc.utils import get_web_search_requests_from_usage
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage


def cost_per_token(
    model: str,
    usage: "Usage",
    service_tier: str | None = None,
    model_info: "ModelInfo | None" = None,
    elapsed: Elapsed | None = None,
) -> tuple[float, float]:
    """
    Prompt and completion cost of ``usage`` on an Anthropic model, with the speed and geo uplifts its row carries.

    ``model`` is the model name without provider prefix, ``service_tier`` the tier the response was served at and
    ``model_info`` the deployment's effective prices when they override the public rates. ``elapsed`` prices a row
    billed by the second.
    """
    return branch_cost(
        model,
        "anthropic",
        usage,
        model_info if model_info is not None else litellm.get_model_info(model=model, custom_llm_provider="anthropic"),
        service_tier=service_tier,
        branch=BranchFacts(elapsed=elapsed),
    )


class _AnthropicServerToolUseProbe(LiteLLMBaseModel):
    web_search_requests: int | None = None


class _AnthropicUsageProbe(LiteLLMBaseModel):
    server_tool_use: _AnthropicServerToolUseProbe | None = None


class _AnthropicResponseProbe(LiteLLMBaseModel):
    usage: _AnthropicUsageProbe | None = None


def get_anthropic_web_search_requests_from_response(
    response_object: object,
) -> int | None:
    """Read usage.server_tool_use.web_search_requests from a raw Anthropic
    /v1/messages response dict, returning None when absent."""
    if not isinstance(response_object, dict):
        return None
    try:
        probe: Final = _AnthropicResponseProbe.model_validate(response_object)
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
    web_search_requests: Final = get_web_search_requests_from_usage(usage)
    if web_search_requests is None:
        return 0.0

    ## Get the cost per web search request
    search_context_pricing: Final[SearchContextCostPerQuery] = (
        model_info.get("search_context_cost_per_query") or SearchContextCostPerQuery()
    )
    cost_per_web_search_request: Final = search_context_pricing.get("search_context_size_medium", 0.0)
    if cost_per_web_search_request is None or cost_per_web_search_request == 0.0:
        return 0.0

    ## Calculate the total cost
    total_cost: Final = cost_per_web_search_request * web_search_requests
    return total_cost
