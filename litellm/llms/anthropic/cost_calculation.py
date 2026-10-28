"""
Helper util for handling anthropic-specific cost calculation
- e.g.: prompt caching
"""

from typing import TYPE_CHECKING, Final, Optional

from pydantic import ValidationError

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    cost_per_text,
    get_web_search_requests_from_usage,
)
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage


def cost_per_token(
    model: str, usage: "Usage", service_tier: str | None = None, model_info: "ModelInfo | None" = None
) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Input:
        - model: str, the model name without provider prefix
        - usage: LiteLLM Usage block, containing anthropic caching information
        - service_tier: the service tier the request was served at (e.g. "priority"),
          read from the Anthropic response usage and used to select tier-specific pricing
        - model_info: effective deployment prices, when they override public rates

    Returns:
        Tuple[float, float] - prompt_cost_in_usd, completion_cost_in_usd
    """
    return cost_per_text(
        model=model,
        usage=usage,
        custom_llm_provider="anthropic",
        service_tier=service_tier,
        model_info=model_info,
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
    """``server_tool_use.web_search_requests`` at the row's medium search context price"""
    from litellm._cost import Usage as TokenUsage
    from litellm.litellm_core_utils.llm_cost_calc.utils import search_cost

    if model_info is None or usage is None:
        return 0.0
    web_search_requests: Final = get_web_search_requests_from_usage(usage)
    if web_search_requests is None:
        return 0.0
    return search_cost(
        model_info, TokenUsage(prompt_tokens=0, text=0, output_text=0, web_search_requests=web_search_requests)
    )
