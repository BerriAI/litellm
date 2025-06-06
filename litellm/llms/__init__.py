from typing import TYPE_CHECKING, Optional

from . import *

if TYPE_CHECKING:
    from litellm.types.utils import ModelInfo, Usage


def get_cost_for_web_search_request(
    custom_llm_provider: str, usage: "Usage", model_info: "ModelInfo"
) -> Optional[float]:
    from .anthropic.cost_calculation import get_cost_for_anthropic_web_search
    from .gemini.cost_calculator import cost_per_web_search_request

    if custom_llm_provider == "gemini":
        return cost_per_web_search_request(usage=usage, model_info=model_info)
    elif custom_llm_provider == "anthropic":
        return get_cost_for_anthropic_web_search(model_info=model_info, usage=usage)
    else:
        return None
