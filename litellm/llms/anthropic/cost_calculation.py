"""
Helper util for handling anthropic-specific cost calculation
- e.g.: prompt caching
"""

from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, Optional

from pydantic import BaseModel, ValidationError

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    generic_cost_per_token,
    get_provider_specific_geo_multiplier,
    get_web_search_requests_from_usage,
)

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage
import litellm


def cost_per_token(model: str, usage: "Usage", service_tier: str | None = None) -> tuple[float, float]:
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
    iteration_costs = _cost_per_iteration(model=model, usage=usage, service_tier=service_tier)
    if iteration_costs is not None:
        prompt_cost, completion_cost = iteration_costs
    else:
        prompt_cost, completion_cost = generic_cost_per_token(
            model=model,
            usage=usage,
            custom_llm_provider="anthropic",
            service_tier=service_tier,
        )

    # Apply provider_specific_entry multipliers for geo/speed routing
    try:
        model_info: Final = litellm.get_model_info(model=model, custom_llm_provider="anthropic")
        provider_specific_entry: Final[dict] = model_info.get("provider_specific_entry") or {}

        geo_multiplier: Final = get_provider_specific_geo_multiplier(model_info=model_info, usage=usage)
        speed_multiplier: Final = (
            provider_specific_entry.get("fast", 1.0) if getattr(usage, "speed", None) == "fast" else 1.0
        )

        if speed_multiplier != 1.0:
            prompt_cost *= speed_multiplier
            completion_cost *= speed_multiplier

        if geo_multiplier != 1.0:
            prompt_cost *= geo_multiplier
            completion_cost *= geo_multiplier
    except Exception:
        pass

    return prompt_cost, completion_cost


def _attempt_is_billed(attempt: Mapping[str, object]) -> bool:
    """Anthropic does not bill an attempt the classifier blocked before any output; every other attempt is billed."""
    return attempt.get("type") != "message" or bool(attempt.get("output_tokens"))


def _attempt_cost(
    attempt: Mapping[str, object],
    fallback_model: str,
    usage: "Usage",
    service_tier: str | None,
) -> tuple[float, float]:
    from litellm.llms.anthropic.chat.transformation import AnthropicConfig

    model: Final = str(attempt.get("model") or fallback_model)
    attempt_usage: Final = AnthropicConfig().calculate_usage(
        usage_object={**attempt, "inference_geo": getattr(usage, "inference_geo", None)},
        reasoning_content=None,
        speed=getattr(usage, "speed", None),
    )
    return generic_cost_per_token(
        model=model,
        usage=attempt_usage,
        custom_llm_provider="anthropic",
        service_tier=service_tier,
    )


def _cost_per_iteration(model: str, usage: "Usage", service_tier: str | None) -> tuple[float, float] | None:
    """
    Price a server-side fallback response one attempt at a time.

    ``usage.iterations`` holds one entry per attempt with its own ``model``; the top-level ``model`` is only the
    attempt that produced the returned message. Pricing the summed tokens at that one model charges a declined
    attempt at the fallback model's rates. Anthropic bills each attempt at its own model, and does not bill an
    attempt that was blocked before any output. Returns None for single-attempt responses.
    """
    iterations: Final = getattr(usage, "iterations", None)
    attempts: Final = tuple(it for it in (iterations or ()) if isinstance(it, Mapping))
    if len(attempts) < 2 or not any(it.get("model") for it in attempts):
        return None
    costs: Final = tuple(_attempt_cost(it, model, usage, service_tier) for it in attempts if _attempt_is_billed(it))
    return sum(p for p, _ in costs), sum(c for _, c in costs)


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
