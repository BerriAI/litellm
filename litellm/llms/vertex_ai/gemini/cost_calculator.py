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

    Vertex AI charges $35/1000 prompts, independent of the number of web search requests.

    For a single call, this is $35e-3 USD.

    Args:
        usage: The usage object for the web search request.
        model_info: The model info for the web search request.

    Returns:
        The cost of the web search request.
    """
    from litellm.types.utils import PromptTokensDetailsWrapper

    # check if usage object has web search requests
    cost_per_llm_call_with_web_search = 35e-3

    makes_web_search_request = False
    if (
        usage is not None
        and usage.prompt_tokens_details is not None
        and isinstance(usage.prompt_tokens_details, PromptTokensDetailsWrapper)
    ):
        makes_web_search_request = True

    # Calculate total cost
    if makes_web_search_request:
        return cost_per_llm_call_with_web_search
    else:
        return 0.0
