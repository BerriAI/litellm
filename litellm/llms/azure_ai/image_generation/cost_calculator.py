import re
from collections.abc import Mapping
from typing import Any, Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _get_cost_per_unit,  # pyright: ignore[reportPrivateUsage]  # shared rate-resolution helper used by sibling calculators
    calculate_image_response_cost_from_usage,
    resolve_image_model_info,
)
from litellm.types.utils import ImageResponse, ModelInfo

_SIZE_PATTERN: Final = re.compile(r"(\d+)(?:x|-x-)(\d+)")


def _rate(table: ModelInfo, key: str) -> float | None:
    return _get_cost_per_unit(table, key, default_value=None)


def _generated_pixels(optional_params: Mapping[str, object] | None, size: str | None) -> int:
    width: Final = optional_params.get("width") if optional_params else None
    height: Final = optional_params.get("height") if optional_params else None
    if type(width) is int and type(height) is int and width > 0 and height > 0:
        return width * height
    match: Final = _SIZE_PATTERN.fullmatch(size or "")
    return int(match[1]) * int(match[2]) if match else 0


def cost_calculator(
    model: str,
    image_response: Any,
    size: str | None = None,
    n: int | None = None,
    optional_params: Mapping[str, object] | None = None,
    model_info: ModelInfo | None = None,
) -> float:
    """
    Azure AI image generation cost calculator
    """
    if not isinstance(image_response, ImageResponse):
        raise TypeError(f"image_response must be of type ImageResponse got type={type(image_response)}")

    resolved: Final = resolve_image_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
        model_info=model_info,
    )

    usage_cost: Final = calculate_image_response_cost_from_usage(
        model=model,
        image_response=image_response,
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
        model_info=resolved,
    )
    if usage_cost is not None:
        return usage_cost

    num_images: Final = n if n is not None else len(image_response.data or ())
    generated_cost: Final = _generated_cost(resolved, num_images, _generated_pixels(optional_params, size))
    per_pixel: Final = _rate(resolved, "input_cost_per_pixel") or 0.0
    reference_cost: Final = per_pixel * max(image_response.reference_pixels or 0, 0)
    return generated_cost + reference_cost


def _generated_cost(resolved: ModelInfo, num_images: int, pixels_per_image: int) -> float:
    per_image: Final = _rate(resolved, "output_cost_per_image")
    if per_image is not None:
        return per_image * num_images
    per_pixel: Final = _rate(resolved, "input_cost_per_pixel")
    if per_pixel is not None:
        return per_pixel * pixels_per_image * num_images
    return 0.0
