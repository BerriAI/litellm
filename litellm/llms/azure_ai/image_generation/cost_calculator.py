from collections.abc import Mapping
from typing import Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_image_response_cost_from_usage,
    image_cost,
    resolve_image_model_info,
)
from litellm.types.utils import ImageResponse, ModelInfo


def _requested_dimensions(optional_params: Mapping[str, object] | None, size: object) -> tuple[int, int] | None:
    width: Final = optional_params.get("width") if optional_params else None
    height: Final = optional_params.get("height") if optional_params else None
    if type(width) is int and type(height) is int and width > 0 and height > 0:
        return width, height
    if not isinstance(size, str):
        return None
    parts: Final = size.replace("-x-", "x").split("x")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        return None
    return int(parts[0]), int(parts[1])


def cost_calculator(
    model: str,
    image_response: object,
    size: str | None = None,
    n: int | None = None,
    optional_params: Mapping[str, object] | None = None,
    model_info: ModelInfo | None = None,
) -> float:
    """
    Azure AI image generation cost calculator
    """
    _model_info: Final = resolve_image_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
        model_info=model_info,
    )

    if isinstance(image_response, ImageResponse):
        token_based_cost: Final = calculate_image_response_cost_from_usage(
            model=model,
            image_response=image_response,
            custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
            model_info=_model_info,
        )
        if token_based_cost is not None:
            return token_based_cost

        num_images: Final = n if n is not None else len(image_response.data or ())
        width, height = _requested_dimensions(optional_params, size or image_response.size) or (1024, 1024)
        key: Final = _model_info.get("key")
        catalog_row: Final = litellm.model_cost.get(key) if isinstance(key, str) else None
        row: Final = {**_model_info, **(catalog_row or {}), **(model_info or {})}
        return image_cost(row, num_images, width=width, height=height) or 0.0

    raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")
