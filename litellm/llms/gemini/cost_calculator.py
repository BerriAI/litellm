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
    from litellm.litellm_core_utils.llm_cost_calc.utils import cost_per_text

    return cost_per_text(
        model=model,
        usage=usage,
        custom_llm_provider="gemini",
        service_tier=service_tier,
    )


def web_search_requests(usage: "Usage | None") -> int:
    """Grounded searches, Gemini's ``prompt_tokens_details.web_search_requests`` else ``server_tool_use``"""
    from litellm.litellm_core_utils.llm_cost_calc.utils import get_web_search_requests_from_usage
    from litellm.types.utils import PromptTokensDetailsWrapper

    details: Final = usage.prompt_tokens_details if usage is not None else None
    from_details: Final = (
        details.web_search_requests
        if isinstance(details, PromptTokensDetailsWrapper) and getattr(details, "web_search_requests", None)
        else None
    )
    return from_details or (get_web_search_requests_from_usage(usage) if usage is not None else None) or 0


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """Grounding with Google Search, per query or once per prompt as the row's ``web_search_billing_unit`` says"""
    from litellm._cost import Usage as TokenUsage
    from litellm.litellm_core_utils.llm_cost_calc.utils import search_cost

    return search_cost(
        _as_google_row(model_info),
        TokenUsage(prompt_tokens=0, text=0, output_text=0, web_search_requests=web_search_requests(usage)),
    )


def _as_google_row(model_info: "ModelInfo") -> dict[str, object]:
    """The row with a Google provider, so its missing search and Maps prices take Gemini's defaults"""
    provider: Final = model_info.get("litellm_provider")
    google: Final = isinstance(provider, str) and (provider == "gemini" or provider.startswith("vertex_ai"))
    return {**model_info, "litellm_provider": provider if google else "gemini"}


def google_maps_grounding_requests(usage: "Usage | None") -> int | None:
    from litellm.types.utils import PromptTokensDetailsWrapper

    details: Final = usage.prompt_tokens_details if usage is not None else None
    if not isinstance(details, PromptTokensDetailsWrapper) or not hasattr(details, "google_maps_grounding_requests"):
        return None
    return details.google_maps_grounding_requests


def cost_per_google_maps_grounding_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """Grounding with Google Maps, under the same billing unit as Google Search grounding"""
    from litellm._cost import Usage as TokenUsage
    from litellm.litellm_core_utils.llm_cost_calc.utils import search_cost

    return search_cost(
        _as_google_row(model_info),
        TokenUsage(
            prompt_tokens=0,
            text=0,
            output_text=0,
            maps_grounding_requests=max(google_maps_grounding_requests(usage) or 0, 0),
        ),
    )
