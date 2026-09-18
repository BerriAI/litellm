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
    from litellm.litellm_core_utils.llm_cost_calc.utils import generic_cost_per_token

    return generic_cost_per_token(
        model=model,
        usage=usage,
        custom_llm_provider="gemini",
        service_tier=service_tier,
    )


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    Calculates the cost of web search (grounding with Google Search).

    Billing mode is determined by ``web_search_billing_unit`` in model_info:
    - ``"per_query"``: charged per individual search query (Gemini 3.x).
    - ``"per_prompt"`` (default): charged per grounded prompt (Gemini 2.x),
      regardless of how many queries were executed internally.

    Reads the per-request cost from ``search_context_cost_per_query`` in
    ``model_info`` when available, falling back to $0.035 for models not
    yet updated in the pricing JSON.
    """
    from litellm.litellm_core_utils.llm_cost_calc.utils import (
        get_web_search_requests_from_usage,
    )
    from litellm.types.utils import PromptTokensDetailsWrapper

    _DEFAULT_COST: Final = 35e-3
    search_costs: Final = model_info.get("search_context_cost_per_query") or {}
    _cost: Final = search_costs.get("search_context_size_medium", _DEFAULT_COST)

    requests_from_prompt_details: Final = (
        usage.prompt_tokens_details.web_search_requests
        if (
            usage is not None
            and usage.prompt_tokens_details is not None
            and isinstance(usage.prompt_tokens_details, PromptTokensDetailsWrapper)
            and hasattr(usage.prompt_tokens_details, "web_search_requests")
            and usage.prompt_tokens_details.web_search_requests is not None
        )
        else None
    )
    requests_from_server_tool_use: Final = get_web_search_requests_from_usage(usage)
    number_of_web_search_requests: Final = requests_from_prompt_details or requests_from_server_tool_use or 0

    billing_mode: Final = model_info.get("web_search_billing_unit") or "per_prompt"
    billable_requests: Final = (
        1 if (number_of_web_search_requests > 0 and billing_mode == "per_prompt") else number_of_web_search_requests
    )

    return _cost * billable_requests


GOOGLE_MAPS_GROUNDING_DEFAULT_COST_PER_QUERY: Final = 14e-3
GOOGLE_MAPS_GROUNDING_DEFAULT_COST_PER_PROMPT: Final = 25e-3


def google_maps_grounding_requests(usage: "Usage | None") -> int | None:
    from litellm.types.utils import PromptTokensDetailsWrapper

    details: Final = usage.prompt_tokens_details if usage is not None else None
    if not isinstance(details, PromptTokensDetailsWrapper) or not hasattr(details, "google_maps_grounding_requests"):
        return None
    return details.google_maps_grounding_requests


def cost_per_google_maps_grounding_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    Calculates the cost of Grounding with Google Maps.

    Billing follows ``web_search_billing_unit`` in model_info the same way Google Search grounding
    does: ``"per_query"`` (Gemini 3.x) multiplies the executed Maps queries, ``"per_prompt"``
    (default, Gemini 2.x) charges one flat fee per grounded prompt.

    The rate comes from ``google_maps_grounding_cost_per_query`` in ``model_info``, falling back
    to Google's list price for that billing unit when the pricing JSON has no entry yet.
    """
    requests: Final = google_maps_grounding_requests(usage)
    if not requests or requests <= 0:
        return 0.0
    billing_mode: Final = model_info.get("web_search_billing_unit") or "per_prompt"
    default_cost: Final = (
        GOOGLE_MAPS_GROUNDING_DEFAULT_COST_PER_QUERY
        if billing_mode == "per_query"
        else GOOGLE_MAPS_GROUNDING_DEFAULT_COST_PER_PROMPT
    )
    configured_cost: Final = model_info.get("google_maps_grounding_cost_per_query")
    cost: Final = default_cost if configured_cost is None else configured_cost
    billed_requests: Final = requests if billing_mode == "per_query" else 1
    return cost * billed_requests
