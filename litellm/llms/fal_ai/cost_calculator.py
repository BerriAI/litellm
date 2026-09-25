import os
from collections.abc import Mapping
from math import ceil
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import deployment_pricing, resolve_image_model_info
from litellm.types.utils import ImageObject, ImageResponse, ModelInfo

FAL_KEYED_PRICING_DEFAULT_QUALITY: Final[str] = "high"
_DEFAULT_KEYED_DIMENSIONS: Final[tuple[int, int]] = (1024, 768)
FAL_TEXT_TO_IMAGE_DEFAULT_SIZE: Final[str] = f"{_DEFAULT_KEYED_DIMENSIONS[0]}-x-{_DEFAULT_KEYED_DIMENSIONS[1]}"
FAL_PIXELS_PER_MEGAPIXEL: Final[int] = 1_048_576
FAL_NAMED_IMAGE_SIZES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "square_hd": "1024-x-1024",
        "square": "512-x-512",
        "portrait_4_3": "768-x-1024",
        "portrait_16_9": "576-x-1024",
        "landscape_4_3": "1024-x-768",
        "landscape_16_9": "1024-x-576",
    }
)

_OBJECT_MAP: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])

FAL_AI_QUEUE_DEFAULT_BASE: Final[str] = "https://queue.fal.run"


def fal_ai_queue_base() -> str:
    return os.getenv("FAL_AI_QUEUE_API_BASE") or FAL_AI_QUEUE_DEFAULT_BASE


def _keyed_size(optional_params: Mapping[str, object]) -> str | None:
    image_size: Final = optional_params.get("image_size")
    if image_size is None or image_size == "auto":
        return FAL_TEXT_TO_IMAGE_DEFAULT_SIZE
    if isinstance(image_size, Mapping):
        image_size_map: Final = _OBJECT_MAP.validate_python(image_size)
        width: Final = image_size_map.get("width")
        height: Final = image_size_map.get("height")
        if isinstance(width, int) and isinstance(height, int):
            return f"{width}-x-{height}"
        return None
    if isinstance(image_size, str):
        return FAL_NAMED_IMAGE_SIZES.get(image_size)
    return None


def _image_dimensions(image: object) -> tuple[int, int] | None:
    if not isinstance(image, ImageObject):
        return None
    raw_provider_specific_fields: Final = image.provider_specific_fields
    if not isinstance(raw_provider_specific_fields, Mapping):
        return None
    provider_specific_fields: Final = _OBJECT_MAP.validate_python(raw_provider_specific_fields)
    width: Final = provider_specific_fields.get("width")
    height: Final = provider_specific_fields.get("height")
    if type(width) is not int or width <= 0 or type(height) is not int or height <= 0:
        return None
    return width, height


def _keyed_quality(optional_params: Mapping[str, object]) -> str:
    raw_quality: Final = optional_params.get("quality")
    return raw_quality if isinstance(raw_quality, str) and raw_quality != "auto" else FAL_KEYED_PRICING_DEFAULT_QUALITY


def _parse_keyed_dimensions(size: str | None) -> tuple[int, int] | None:
    if size is None:
        return None
    parts: Final = tuple(size.split("-x-"))
    if len(parts) != 2:
        return None
    try:
        width, height = (int(part) for part in parts)
    except ValueError:
        return None
    return (width, height) if width > 0 and height > 0 else None


def _keyed_rows(model: str, quality: str) -> tuple[tuple[int, int, float], ...]:
    prefix: Final = f"fal_ai/{quality}/"
    suffix: Final = f"/{model}"
    return tuple(
        (width, height, float(raw_cost))
        for key in litellm.model_cost
        if isinstance(key, str) and key.startswith(prefix) and key.endswith(suffix)
        for size in (key[len(prefix) : -len(suffix)],)
        for dimensions in (_parse_keyed_dimensions(size),)
        if dimensions is not None
        for entry in (_entry(key),)
        if entry is not None
        for raw_cost in (entry.get("output_cost_per_image"),)
        if isinstance(raw_cost, (int, float))
        for width, height in (dimensions,)
    )


def _keyed_cost_per_image(
    model: str,
    image: object,
    optional_params: Mapping[str, object],
) -> float | None:
    quality: Final = _keyed_quality(optional_params)
    rows: Final = _keyed_rows(model, quality)
    if not rows:
        return None
    target_dimensions: Final = (
        _image_dimensions(image) or _parse_keyed_dimensions(_keyed_size(optional_params)) or _DEFAULT_KEYED_DIMENSIONS
    )
    target_pixels: Final = target_dimensions[0] * target_dimensions[1]
    return min(rows, key=lambda row: (abs(row[0] * row[1] - target_pixels), row[0] * row[1]))[2]


def _flat_cost_per_image(
    image: object,
    output_cost_per_image: float,
    output_cost_per_pixel: float | None,
) -> float:
    dimensions: Final = _image_dimensions(image)
    if dimensions is None or output_cost_per_pixel is None:
        return output_cost_per_image
    width, height = dimensions
    megapixels: Final = ceil(width * height / FAL_PIXELS_PER_MEGAPIXEL)
    return output_cost_per_pixel * FAL_PIXELS_PER_MEGAPIXEL * megapixels


def _entry(key: str) -> Mapping[str, object] | None:
    raw_entry: Final[object] = litellm.model_cost.get(key)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # global catalog is untyped
    if not isinstance(raw_entry, Mapping):
        return None
    return _OBJECT_MAP.validate_python(raw_entry)


def _resolution_key(resolution: object) -> str | None:
    if isinstance(resolution, bool) or not isinstance(resolution, (int, str)):
        return None
    return str(resolution)


def _resolution_cost_per_image(entry: Mapping[str, object] | None, resolution: object) -> float | None:
    resolution_key: Final = _resolution_key(resolution)
    if entry is None or resolution_key is None:
        return None
    cost: Final = entry.get(f"output_cost_per_image_{resolution_key}")
    return float(cost) if isinstance(cost, (int, float)) else None


def fal_ai_passthrough_cost(model: str, request_body: Mapping[str, object]) -> float | None:
    entry: Final = _entry(f"{litellm.LlmProviders.FAL_AI.value}/{model}")
    if entry is None:
        return None
    resolution_cost: Final = _resolution_cost_per_image(entry, request_body.get("resolution"))
    if resolution_cost is not None:
        return resolution_cost
    cost: Final = entry.get("output_cost_per_image")
    return float(cost) if isinstance(cost, (int, float)) else None


def cost_calculator(
    model: str,
    image_response: object,
    optional_params: Mapping[str, object] | None = None,
    model_info: ModelInfo | None = None,
) -> float:
    """
    fal.ai image generation cost calculator
    """
    if not isinstance(image_response, ImageResponse):
        raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")
    normalized_model: Final = model.removeprefix(f"{litellm.LlmProviders.FAL_AI.value}/")
    images: Final = tuple(image_response.data or ())
    deployment_prices: Final = deployment_pricing(model_info)
    deployment_cost_per_image: Final = (
        None if deployment_prices is None else deployment_prices.get("output_cost_per_image")
    )
    if deployment_cost_per_image is not None:
        return deployment_cost_per_image * len(images)
    params: Final[Mapping[str, object]] = optional_params or MappingProxyType({})
    resolution_cost_per_image: Final = _resolution_cost_per_image(
        _entry(f"{litellm.LlmProviders.FAL_AI.value}/{normalized_model}"), params.get("resolution")
    )
    if resolution_cost_per_image is not None:
        return resolution_cost_per_image * len(images)
    keyed_costs: Final = tuple(
        _keyed_cost_per_image(
            model=normalized_model,
            image=image,
            optional_params=params,
        )
        for image in images
    )
    if not any(cost is None for cost in keyed_costs):
        return sum(cost for cost in keyed_costs if cost is not None)
    resolved_model_info: Final = resolve_image_model_info(
        model=normalized_model,
        custom_llm_provider=litellm.LlmProviders.FAL_AI.value,
        model_info=deployment_prices,
    )
    raw_output_cost_per_image: Final = resolved_model_info.get("output_cost_per_image")
    output_cost_per_image: Final = (
        float(raw_output_cost_per_image) if isinstance(raw_output_cost_per_image, (int, float)) else 0.0
    )
    raw_output_cost_per_pixel: Final = resolved_model_info.get("output_cost_per_pixel")
    output_cost_per_pixel: Final = (
        float(raw_output_cost_per_pixel) if isinstance(raw_output_cost_per_pixel, (int, float)) else None
    )
    return sum(
        keyed_cost
        if keyed_cost is not None
        else _flat_cost_per_image(
            image=image,
            output_cost_per_image=output_cost_per_image,
            output_cost_per_pixel=output_cost_per_pixel,
        )
        for image, keyed_cost in zip(images, keyed_costs)
    )
