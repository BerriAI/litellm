import re
from collections.abc import Mapping
from typing import Any, Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _get_cost_per_unit,
    calculate_image_response_cost_from_usage,
    resolve_image_model_info,
)
from litellm.types.utils import ImageResponse, ModelInfo

PIXELS_PER_MEGAPIXEL: Final[int] = 1_048_576
_IMAGE_SIZE: Final = re.compile(r"(\d+)(?:x|-x-)(\d+)")
_MEGAPIXEL_PRICE_FIELDS: Final = ("output_cost_per_first_megapixel", "output_cost_per_additional_megapixel")
_FLAT_PRICE_FIELDS: Final = ("output_cost_per_image", "input_cost_per_pixel")


def _input_cost_per_pixel(resolved: ModelInfo) -> float:
    deployment_price: Final = _get_cost_per_unit(resolved, "input_cost_per_pixel", default_value=None)
    if deployment_price is not None:
        return deployment_price
    model_cost_key: Final = resolved.get("key")
    shared_entry: Final = litellm.model_cost.get(model_cost_key) if model_cost_key is not None else None
    if shared_entry is None:
        return 0.0
    return shared_entry.get("input_cost_per_pixel") or 0.0


def _prices_any(model_info: ModelInfo | None, fields: tuple[str, ...]) -> bool:
    return model_info is not None and any(model_info.get(field) is not None for field in fields)


def _megapixel_price(resolved: ModelInfo, deployment: ModelInfo | None, field: str) -> float | None:
    deployment_price: Final = (
        _get_cost_per_unit(deployment, field, default_value=None) if deployment is not None else None
    )
    if deployment_price is not None:
        return deployment_price
    model_cost_key: Final = resolved.get("key")
    shared_entry: Final = litellm.model_cost.get(model_cost_key) if model_cost_key is not None else None
    shared_price: Final = shared_entry.get(field) if shared_entry is not None else None
    return float(shared_price) if isinstance(shared_price, (int, float)) else None


def _megapixel_tiers(resolved: ModelInfo, deployment: ModelInfo | None) -> tuple[float, float] | None:
    if _prices_any(deployment, _FLAT_PRICE_FIELDS) and not _prices_any(deployment, _MEGAPIXEL_PRICE_FIELDS):
        return None
    first: Final = _megapixel_price(resolved, deployment, "output_cost_per_first_megapixel")
    additional: Final = _megapixel_price(resolved, deployment, "output_cost_per_additional_megapixel")
    if first is None:
        return None if additional is None else (additional, additional)
    return first, first if additional is None else additional


def _image_dimensions(size: str | None, optional_params: Mapping[str, object] | None) -> tuple[int, int] | None:
    width: Final = optional_params.get("width") if optional_params else None
    height: Final = optional_params.get("height") if optional_params else None
    if type(width) is int and type(height) is int and width > 0 and height > 0:
        return width, height
    matched: Final = _IMAGE_SIZE.fullmatch(size) if size else None
    return (int(matched.group(1)), int(matched.group(2))) if matched else None


def _tiered_megapixel_cost(first: float, additional: float, width: int, height: int) -> float:
    megapixels: Final = width * height / PIXELS_PER_MEGAPIXEL
    return first * min(megapixels, 1.0) + additional * max(megapixels - 1.0, 0.0)


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
        tiers: Final = _megapixel_tiers(_model_info, model_info)
        dimensions: Final = _image_dimensions(size or image_response.size, optional_params)
        if tiers is not None and dimensions is not None:
            return _tiered_megapixel_cost(*tiers, *dimensions) * num_images

        output_cost_per_image: Final[float] = _model_info.get("output_cost_per_image") or 0.0
        if output_cost_per_image:
            return output_cost_per_image * num_images

        if _input_cost_per_pixel(_model_info):
            from litellm.cost_calculator import default_image_cost_calculator

            pixel_size: Final = f"{dimensions[0]}x{dimensions[1]}" if dimensions else size or image_response.size
            return default_image_cost_calculator(
                model=_model_info.get("key", model),
                custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
                size=pixel_size,
                n=num_images,
                model_info=model_info,
            )
        return 0.0

    raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")
