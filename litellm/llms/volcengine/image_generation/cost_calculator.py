from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.llm_cost_calc.utils import resolve_image_model_info
from litellm.types.utils import ImageObject, ImageResponse, ModelInfo

_LOW_TIER_MAX_PIXELS: Final = 2_610_000
_PROVIDER_ENTRY_KEY: Final = "volcengine_image"
_TIER_FIELDS: Final = frozenset({"output_cost_per_image_1K", "output_cost_per_image_2K"})
_MAPPING: Final = TypeAdapter(Mapping[str, object])


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return _MAPPING.validate_python(value)


def _number(table: Mapping[str, object], key: str) -> float | None:
    value = table.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _pixels(size: object) -> int | None:
    if not isinstance(size, str):
        return None
    width, separator, height = size.partition("x")
    if not separator or not width.isdigit() or not height.isdigit():
        return None
    return int(width) * int(height)


def _tier(pixels: int | None) -> str | None:
    if pixels is None:
        return None
    return "1K" if pixels <= _LOW_TIER_MAX_PIXELS else "2K"


def _output_price(
    pricing: Mapping[str, object],
    entry: Mapping[str, object],
    image: object,
) -> float:
    fields: Final[Mapping[str, object]] = (
        _mapping(image.provider_specific_fields) if isinstance(image, ImageObject) else {}
    )
    tier: Final = _tier(_pixels(fields.get("size")))
    z_index: Final = fields.get("z_index")
    is_layer: Final = isinstance(z_index, int) and z_index > 0
    if is_layer and tier is not None:
        layer_price: Final = _number(entry, f"layer_output_cost_per_image_{tier}")
        if layer_price is not None:
            return layer_price
    if tier is not None:
        tier_price: Final = _number(pricing, f"output_cost_per_image_{tier}")
        if tier_price is not None:
            return tier_price
    return _number(pricing, "output_cost_per_image") or 0.0


def _input_cost(pricing: Mapping[str, object], entry: Mapping[str, object], image_response: ImageResponse) -> float:
    input_price: Final = _number(pricing, "input_cost_per_image")
    if input_price is None:
        return 0.0
    usage: Final = _mapping(image_response.usage.model_dump() if image_response.usage is not None else {})
    input_images: Final = _number(usage, "input_images") or 0.0
    free_input_images: Final = _number(entry, "free_input_images") or 0.0
    return input_price * max(input_images - free_input_images, 0.0)


def _pricing(model: str, model_info: ModelInfo | None) -> Mapping[str, object]:
    shared: Final = resolve_image_model_info(
        model=model,
        custom_llm_provider="volcengine",
        model_info=model_info,
    )
    if (
        model_info is not None
        and model_info.get("output_cost_per_image") is not None
        and not any(model_info.get(field) is not None for field in _TIER_FIELDS)
    ):
        return {key: value for key, value in shared.items() if key not in _TIER_FIELDS}
    return shared


def cost_calculator(
    model: str,
    image_response: ImageResponse,
    model_info: ModelInfo | None = None,
) -> float:
    pricing: Final = _pricing(model, model_info)
    entry: Final = _mapping(_mapping(pricing.get("provider_specific_entry")).get(_PROVIDER_ENTRY_KEY))
    output_cost: Final = sum(_output_price(pricing, entry, image) for image in image_response.data or [])
    return output_cost + _input_cost(pricing, entry, image_response)
