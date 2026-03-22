"""
Cost calculator for Vertex AI Gemini.

Used because there are differences in how Google AI Studio and Vertex AI Gemini handle web search requests.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage


def cost_per_web_search_request(usage: "Usage", model_info: "ModelInfo") -> float:
    """
    Calculate the cost of a web search request for Vertex AI Gemini.

    Vertex AI charges per grounded prompt. The rate varies by model family
    (e.g. $35/1K for Gemini 2.x, $14/1K for Gemini 3.x). The per-call cost
    is read from model_info["web_search_cost_per_request"] and falls back to
    $35e-3 when the field is absent.

    Args:
        usage: The usage object for the web search request.
        model_info: The model info for the web search request.

    Returns:
        The cost of the web search request.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # Read per-call cost from model_info; fall back to legacy $35/1K default
    _cost_per_call = (
        model_info.get("web_search_cost_per_request") if model_info else None
    )
    if _cost_per_call is None:
        _cost_per_call = 35e-3

    makes_web_search_request = False
    if (
        usage is not None
        and usage.prompt_tokens_details is not None
        and isinstance(usage.prompt_tokens_details, PromptTokensDetailsWrapper)
    ):
        makes_web_search_request = True

    # Calculate total cost
    if makes_web_search_request:
        return _cost_per_call
    else:
        return 0.0
