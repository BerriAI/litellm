"""
This file is used to calculate the cost of the Gemini API.

Handles the context caching for Gemini API.
"""

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage


def cost_per_token(model: str, usage: "Usage", service_tier: str | None = None) -> tuple[float, float]:
    """
    Calculates the cost per token for a given model, prompt tokens, and completion tokens.

    Follows the same logic as Anthropic's cost per token calculation.
    """
    from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import branch_cost
    from litellm.utils import get_model_info

    return branch_cost(
        model, "gemini", usage, get_model_info(model=model, custom_llm_provider="gemini"), service_tier=service_tier
    )


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    The cost of web search (grounding with Google Search), priced in Rust: per query for a row whose
    ``web_search_billing_unit`` is ``per_query`` (Gemini 3.x), else once per grounded prompt (Gemini 2.x), at the
    row's ``search_context_cost_per_query`` medium price or Google's list price.
    """
    from litellm.litellm_core_utils.llm_cost_calc.tool_call_cost_tracking import price_web_search_dispatch

    return price_web_search_dispatch("gemini", usage, model_info) or 0.0


def google_maps_grounding_requests(usage: "Usage | None") -> int | None:
    from litellm.types.utils import PromptTokensDetailsWrapper

    details: Final = usage.prompt_tokens_details if usage is not None else None
    if not isinstance(details, PromptTokensDetailsWrapper) or not hasattr(details, "google_maps_grounding_requests"):
        return None
    return details.google_maps_grounding_requests


def maps_grounding_cost(usage: "Usage", model_info: "ModelInfo", custom_llm_provider: str | None) -> float | None:
    """Grounding with Google Maps priced in Rust, ``None`` when ``custom_llm_provider`` is neither Gemini nor Vertex
    AI. Without a provider the Gemini calculator prices it."""
    from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import (
        MapsGroundingTool,
        OneMapsGrounding,
        OneToolOnly,
        price_tools,
    )

    requests: Final = google_maps_grounding_requests(usage)
    maps: Final = MapsGroundingTool(
        provider=custom_llm_provider,
        row=0,
        requests=max(requests, 0) if isinstance(requests, int) else 0,
    )
    return price_tools(OneToolOnly(one=OneMapsGrounding(tool="maps_grounding", **maps)), (model_info,))


def cost_per_google_maps_grounding_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    The cost of Grounding with Google Maps, priced in Rust: per query for a ``per_query`` row (Gemini 3.x), else
    once per grounded prompt (Gemini 2.x), at the row's ``google_maps_grounding_cost_per_query`` or Google's list
    price for that billing unit.
    """
    return maps_grounding_cost(usage, model_info, None) or 0.0
