from typing import TYPE_CHECKING, Optional

from . import *

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage


def get_cost_for_web_search_request(
    custom_llm_provider: str, usage: "Usage", model_info: "ModelInfo"
) -> Optional[float]:
    """
    Get the cost for a web search request for a given model.

    Args:
        custom_llm_provider: The custom LLM provider.
        usage: The usage object.
        model_info: The model info.
    """
    if custom_llm_provider == "gemini":
        from .gemini.cost_calculator import cost_per_web_search_request

        return cost_per_web_search_request(usage=usage, model_info=model_info)
    elif custom_llm_provider == "anthropic":
        from .anthropic.cost_calculation import get_cost_for_anthropic_web_search

        return get_cost_for_anthropic_web_search(model_info=model_info, usage=usage)
    elif custom_llm_provider.startswith("vertex_ai"):
        from .vertex_ai.gemini.cost_calculator import (
            cost_per_web_search_request as cost_per_web_search_request_vertex_ai,
        )

        return cost_per_web_search_request_vertex_ai(usage=usage, model_info=model_info)
    else:
        return None
