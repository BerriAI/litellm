from collections.abc import Mapping
from typing import Any, Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _get_cost_per_unit,  # pyright: ignore[reportPrivateUsage]  # shared rate-resolution helper used by sibling calculators
    calculate_image_response_cost_from_usage,
    resolve_image_model_info,
)
from litellm.types.utils import ImageResponse, ModelInfo


def _rate(table: ModelInfo, key: str) -> float | None:
    return _get_cost_per_unit(table, key, default_value=None)


def _size_pixels(size: str | None) -> int:
    if size is None:
        return 0
    for separator in ("x", "-x-"):
        if separator in size:
            parts = size.split(separator)
            if len(parts) != 2:
                continue
            try:
                width = int(parts[0])
                height = int(parts[1])
            except ValueError:
                continue
            if width > 0 and height > 0:
                return width * height
            continue
    return 0


def _generated_pixels(
    optional_params: Mapping[str, object] | None, size: str | None, image_response: ImageResponse
) -> int:
    width: Final = optional_params.get("width") if optional_params else None
    height: Final = optional_params.get("height") if optional_params else None
    if type(width) is int and type(height) is int and width > 0 and height > 0:
        return width * height
    raw_size: Final = size or image_response.size
    return _size_pixels(raw_size if isinstance(raw_size, str) else None)


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
        raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")

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

    data: Final = image_response.data
    num_images: Final = n if n is not None else (len(data) if isinstance(data, list) else 0)
    generated_meters: Final = (
        ("output_cost_per_image", num_images),
        ("input_cost_per_pixel", _generated_pixels(optional_params, size, image_response) * num_images),
    )
    generated: Final = next(
        (rate * units for key, units in generated_meters if (rate := _rate(resolved, key)) is not None),
        0.0,
    )
    reference_pixels: Final = getattr(image_response, "_reference_pixels", None)
    reference: Final = (_rate(resolved, "input_cost_per_reference_pixel") or 0.0) * (
        reference_pixels if isinstance(reference_pixels, int) and not isinstance(reference_pixels, bool) else 0
    )
    return generated + reference
