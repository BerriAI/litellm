import os
from collections.abc import Mapping
from math import isfinite
from types import MappingProxyType
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.core_helpers import get_provider_response_headers_from_hidden_params
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import (
    FalImageDocument,
    FalImageFacts,
    FalSubmitDocument,
    price_fal,
    total_or_zero,
)
from litellm.litellm_core_utils.llm_cost_calc.utils import deployment_pricing, resolve_image_model_info
from litellm.llms.openai.cost_calculation import clip_seconds
from litellm.rust_bridge._native import NativeReceipt
from litellm.types.utils import ImageObject, ImageResponse, ModelInfo

FAL_KEYED_PRICING_DEFAULT_QUALITY: Final[str] = "high"
_DEFAULT_KEYED_DIMENSIONS: Final[tuple[int, int]] = (1024, 768)
FAL_TEXT_TO_IMAGE_DEFAULT_SIZE: Final[str] = f"{_DEFAULT_KEYED_DIMENSIONS[0]}-x-{_DEFAULT_KEYED_DIMENSIONS[1]}"
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
FAL_BILLABLE_UNITS_HEADER: Final[str] = "x-fal-billable-units"


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


def _entry(key: str) -> Mapping[str, object] | None:
    raw_entry: Final[object] = litellm.model_cost.get(key)  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # global catalog is untyped
    if not isinstance(raw_entry, Mapping):
        return None
    return _OBJECT_MAP.validate_python(raw_entry)


def _resolution_key(resolution: object) -> str | None:
    if isinstance(resolution, bool) or not isinstance(resolution, (int, str)):
        return None
    return str(resolution)


def _billable_units(response: object) -> float | None:
    headers: Final = get_provider_response_headers_from_hidden_params(response)
    raw_units: Final = None if headers is None else headers.get(FAL_BILLABLE_UNITS_HEADER)
    if raw_units is None:
        return None
    try:
        units: Final = float(raw_units)
    except ValueError:
        return None
    return units if isfinite(units) and units >= 0 else None


def _requested_image_count(request_body: Mapping[str, object]) -> int:
    num_images: Final = request_body.get("num_images")
    return num_images if type(num_images) is int and num_images > 0 else 1


def fal_ai_passthrough_cost(model: str, request_body: Mapping[str, object]) -> float | None:
    entry: Final = _entry(f"{litellm.LlmProviders.FAL_AI.value}/{model}")
    if entry is None:
        return None
    resolution: Final = request_body.get("resolution")
    outcome: Final = price_fal(
        FalSubmitDocument(
            kind="fal_submit",
            model=model,
            seconds=clip_seconds(request_body.get("duration")),
            clip_resolution=resolution if isinstance(resolution, str) else None,
            image_resolution=_resolution_key(resolution),
            images=_requested_image_count(request_body),
        ),
        catalog=entry,
    )
    return outcome.total if isinstance(outcome, NativeReceipt) else None


def _image_facts(image: object, optional_params: Mapping[str, object]) -> FalImageFacts:
    dimensions: Final = _image_dimensions(image)
    target: Final = dimensions or _parse_keyed_dimensions(_keyed_size(optional_params)) or _DEFAULT_KEYED_DIMENSIONS
    return FalImageFacts(target=target, pixels=dimensions[0] * dimensions[1] if dimensions else None)


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
    params: Final[Mapping[str, object]] = optional_params or MappingProxyType({})
    deployment_prices: Final = deployment_pricing(model_info)
    tables: Final = (deployment_prices,) if deployment_prices is not None else ()
    entry: Final = _entry(f"{litellm.LlmProviders.FAL_AI.value}/{normalized_model}")

    def document(fallback: bool) -> FalImageDocument:
        return FalImageDocument(
            kind="fal_image",
            model=normalized_model,
            quality=_keyed_quality(params),
            images=tuple(_image_facts(image, params) for image in image_response.data or ()),
            billable_units=_billable_units(image_response),
            resolution=_resolution_key(params.get("resolution")),
            deployment=deployment_prices is not None,
            fallback=fallback,
        )

    outcome: Final = price_fal(document(fallback=False), tables, entry)
    if isinstance(outcome, NativeReceipt):
        return outcome.total
    resolved_model_info: Final = resolve_image_model_info(
        model=normalized_model,
        custom_llm_provider=litellm.LlmProviders.FAL_AI.value,
        model_info=deployment_prices,
    )
    return total_or_zero(price_fal(document(fallback=True), (*tables, resolved_model_info), entry))
