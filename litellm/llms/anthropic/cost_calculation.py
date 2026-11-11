"""
Helper util for handling anthropic-specific cost calculation
- e.g.: prompt caching
"""

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Final, Optional

from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    generic_cost_per_token,
    get_provider_specific_geo_multiplier,
    get_provider_specific_speed_multiplier,
    get_web_search_requests_from_usage,
)
from litellm.types.llms.base import LiteLLMBaseModel

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage
import litellm

# A refusal before any output is billed only in these categories, per
# https://platform.claude.com/docs/en/build-with-claude/refusals-and-fallback#how-refusals-are-billed (2026-11-03)
BILLED_REFUSAL_CATEGORIES: Final = frozenset({"bio", "frontier_llm", "reasoning_extraction"})

_USAGE_ITERATIONS: Final = TypeAdapter(tuple[Mapping[str, object], ...])
_REFUSAL_CATEGORIES: Final = TypeAdapter(tuple[str | None, ...])
_PRICING_FIELDS: Final = ("inference_geo", "speed")


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
    iterations: Final = _USAGE_ITERATIONS.validate_python(getattr(usage, "iterations", None) or ())
    served_by: Final = tuple(attempt for attempt in iterations if attempt.get("type") == "fallback_message")
    if not served_by:
        return _single_model_cost_per_token(model, usage, service_tier, model_info)
    # Pricing by any model other than the one that answered means the deployment set its own prices or base_model
    prices_each_attempt_at_its_model: Final = _attempt_model(served_by[-1], model) == model
    costs: Final = tuple(
        _attempt_cost(
            attempt=attempt,
            attempt_model=_attempt_model(attempt, model) if prices_each_attempt_at_its_model else model,
            model=model,
            usage=usage,
            service_tier=service_tier,
            model_info=model_info,
        )
        for attempt in _billed_fallback_attempts(iterations, usage)
    )
    return sum(prompt_cost for prompt_cost, _ in costs), sum(completion_cost for _, completion_cost in costs)


def _billed_fallback_attempts(
    iterations: Sequence[Mapping[str, object]], usage: "Usage"
) -> Sequence[Mapping[str, object]]:
    """The billed entries of a server-side fallback chain's ``usage.iterations``. The first
    ``fallback_refusal_categories`` entries declined, and one that declined before any output is billed only in a
    billed category."""
    categories: Final = _REFUSAL_CATEGORIES.validate_python(getattr(usage, "fallback_refusal_categories", None) or ())
    return tuple(
        attempt
        for index, attempt in enumerate(iterations)
        if index >= len(categories) or _produced_output(attempt) or categories[index] in BILLED_REFUSAL_CATEGORIES
    )


def _produced_output(attempt: Mapping[str, object]) -> bool:
    match attempt:
        case {"output_tokens": int() as output_tokens}:
            return output_tokens > 0
        case _:
            return False


def _attempt_model(attempt: Mapping[str, object], default: str) -> str:
    match attempt:
        case {"model": str() as attempt_model}:
            return attempt_model
        case _:
            return default


def _attempt_cost(
    attempt: Mapping[str, object],
    attempt_model: str,
    model: str,
    usage: "Usage",
    service_tier: str | None,
    model_info: "ModelInfo | None",
) -> tuple[float, float]:
    return _single_model_cost_per_token(
        model=attempt_model,
        usage=_attempt_usage(attempt, usage),
        service_tier=service_tier,
        model_info=model_info if attempt_model == model else None,
    )


def _attempt_usage(attempt: Mapping[str, object], usage: "Usage") -> "Usage":
    pricing_fields: Final = {field: getattr(usage, field) for field in _PRICING_FIELDS if hasattr(usage, field)}
    return litellm.AnthropicConfig().calculate_usage(usage_object={**attempt, **pricing_fields}, reasoning_content=None)


def _single_model_cost_per_token(
    model: str, usage: "Usage", service_tier: str | None, model_info: "ModelInfo | None"
) -> tuple[float, float]:
    prompt_cost, completion_cost = generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="anthropic",
        service_tier=service_tier,
        model_info=model_info,
    )

    # Apply provider_specific_entry multipliers for geo/speed routing
    try:
        effective_info: Final = (
            model_info
            if model_info is not None
            else litellm.get_model_info(model=model, custom_llm_provider="anthropic")
        )
        geo_multiplier: Final = get_provider_specific_geo_multiplier(model_info=effective_info, usage=usage)
        speed_multiplier: Final = get_provider_specific_speed_multiplier(model_info=effective_info, usage=usage)

        if speed_multiplier != 1.0:
            prompt_cost *= speed_multiplier
            completion_cost *= speed_multiplier

        if geo_multiplier != 1.0:
            prompt_cost *= geo_multiplier
            completion_cost *= geo_multiplier
    except Exception:
        pass

    return prompt_cost, completion_cost


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
