from collections.abc import Mapping
from math import ceil
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.types.utils import ImageObject, ImageResponse

FAL_KEYED_PRICING_DEFAULT_QUALITY: Final[str] = "high"
FAL_TEXT_TO_IMAGE_DEFAULT_SIZE: Final[str] = "1024-x-768"
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

_MODEL_COST_MAP: Final[TypeAdapter[Mapping[str, Mapping[str, object]]]] = TypeAdapter(
    Mapping[str, Mapping[str, object]]
)
_OBJECT_MAP: Final[TypeAdapter[Mapping[str, object]]] = TypeAdapter(Mapping[str, object])


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


def _response_size(image: object) -> str | None:
    if not isinstance(image, ImageObject):
        return None
    raw_provider_specific_fields: Final = image.provider_specific_fields
    if not isinstance(raw_provider_specific_fields, Mapping):
        return None
    provider_specific_fields: Final = _OBJECT_MAP.validate_python(raw_provider_specific_fields)
    width: Final = provider_specific_fields.get("width")
    height: Final = provider_specific_fields.get("height")
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    return f"{width}-x-{height}"


def _keyed_quality(optional_params: Mapping[str, object]) -> str:
    raw_quality: Final = optional_params.get("quality")
    return raw_quality if isinstance(raw_quality, str) and raw_quality != "auto" else FAL_KEYED_PRICING_DEFAULT_QUALITY


def _keyed_cost_per_image(
    model: str,
    image: object,
    optional_params: Mapping[str, object],
    model_cost_map: Mapping[str, Mapping[str, object]],
) -> float | None:
    quality: Final = _keyed_quality(optional_params)
    request_size: Final = _keyed_size(optional_params) or FAL_TEXT_TO_IMAGE_DEFAULT_SIZE
    sizes: Final = (_response_size(image), request_size, FAL_TEXT_TO_IMAGE_DEFAULT_SIZE)
    for size in sizes:
        if size is None:
            continue
        keyed_entry = model_cost_map.get(f"fal_ai/{quality}/{size}/{model}")
        if keyed_entry is None:
            continue
        keyed_cost = keyed_entry.get("output_cost_per_image")
        if isinstance(keyed_cost, (int, float)):
            return float(keyed_cost)
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
    if not isinstance(width, int) or not isinstance(height, int):
        return None
    return width, height


def _flat_cost_per_image(
    image: object,
    output_cost_per_image: float,
    output_cost_per_pixel: float | None,
) -> float:
    dimensions: Final = _image_dimensions(image)
    if dimensions is None or output_cost_per_pixel is None:
        return output_cost_per_image
    width, height = dimensions
    megapixels: Final = 1 if (width, height) == (1024, 1024) else ceil(width * height / 1_000_000)
    return output_cost_per_pixel * 1_000_000 * megapixels


def cost_calculator(
    model: str,
    image_response: object,
    optional_params: Mapping[str, object] | None = None,
) -> float:
    """
    fal.ai image generation cost calculator
    """
    if not isinstance(image_response, ImageResponse):
        raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")
    normalized_model: Final = model.removeprefix(f"{litellm.LlmProviders.FAL_AI.value}/")
    params: Final[Mapping[str, object]] = optional_params or MappingProxyType({})
    images: Final = tuple(image_response.data or ())
    raw_model_cost: Final[object] = litellm.model_cost  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # global catalog is untyped
    model_cost_map: Final = _MODEL_COST_MAP.validate_python(raw_model_cost)
    keyed_costs: Final = tuple(
        _keyed_cost_per_image(
            model=normalized_model,
            image=image,
            optional_params=params,
            model_cost_map=model_cost_map,
        )
        for image in images
    )
    if all(cost is not None for cost in keyed_costs):
        return sum(cost for cost in keyed_costs if cost is not None)
    model_info_entry: Final = next(
        (
            entry
            for key in (f"{litellm.LlmProviders.FAL_AI.value}/{normalized_model}", normalized_model)
            if (entry := model_cost_map.get(key)) is not None
        ),
        None,
    )
    model_info: Final = _OBJECT_MAP.validate_python(model_info_entry or MappingProxyType({}))
    raw_output_cost_per_image: Final = model_info.get("output_cost_per_image")
    output_cost_per_image: Final = (
        float(raw_output_cost_per_image) if isinstance(raw_output_cost_per_image, (int, float)) else 0.0
    )
    raw_output_cost_per_pixel: Final = model_info.get("output_cost_per_pixel")
    output_cost_per_pixel: Final = (
        float(raw_output_cost_per_pixel) if isinstance(raw_output_cost_per_pixel, (int, float)) else None
    )
    return sum(
        _flat_cost_per_image(
            image=image,
            output_cost_per_image=output_cost_per_image,
            output_cost_per_pixel=output_cost_per_pixel,
        )
        for image in images
    )
