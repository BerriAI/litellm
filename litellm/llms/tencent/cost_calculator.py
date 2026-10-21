from typing import Final

from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_image_response_cost_from_usage,
    flat_image_cost,
    generic_cost_per_token,
    resolve_image_model_info,
)
from litellm.types.utils import ImageResponse, ModelInfo, Usage


def cost_per_token(model: str, usage: Usage) -> tuple[float, float]:
    return generic_cost_per_token(model=model, usage=usage, custom_llm_provider="tencent")


def image_cost_calculator(model: str, image_response: ImageResponse, model_info: ModelInfo | None = None) -> float:
    resolved_model_info: Final = resolve_image_model_info(
        model=model, custom_llm_provider="tencent", model_info=model_info
    )
    token_based_cost: Final = calculate_image_response_cost_from_usage(
        model=model,
        image_response=image_response,
        custom_llm_provider="tencent",
        model_info=resolved_model_info,
    )
    if token_based_cost is not None:
        return token_based_cost
    return flat_image_cost(resolved_model_info, image_response)
