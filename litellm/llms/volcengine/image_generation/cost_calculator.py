from collections.abc import Mapping
from itertools import chain
from typing import Final

from openai.types.images_response import Image as OpenAIImage
from pydantic import TypeAdapter

from litellm.litellm_core_utils.llm_cost_calc.utils import resolve_image_model_info
from litellm.types.utils import ImageObject, ImageResponse, ModelInfo

_LOW_TIER_MAX_PIXELS: Final = 1536 * 1536 * 1.05 * 1.05
_PROVIDER_ENTRY_KEY: Final = "volcengine_image"
_MAPPING: Final = TypeAdapter(Mapping[str, object])


def _mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        return {}
    return _MAPPING.validate_python(value)


def _number(table: Mapping[str, object], key: str) -> float | None:
    value: Final = table.get(key)
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


def _image_fields(image: OpenAIImage) -> Mapping[str, object]:
    return _mapping(image.provider_specific_fields) if isinstance(image, ImageObject) else {}


def _generated_images(image_response: ImageResponse) -> tuple[Mapping[str, object], ...]:
    return tuple(_image_fields(image) for image in image_response.data or () if image.url or image.b64_json)


def _is_layer(fields: Mapping[str, object]) -> bool:
    z_index: Final = fields.get("z_index")
    return isinstance(z_index, int) and z_index > 0


def _provider_entry(prices: Mapping[str, object]) -> Mapping[str, object]:
    return _mapping(_mapping(prices.get("provider_specific_entry")).get(_PROVIDER_ENTRY_KEY))


def _price_lookups(
    prices: Mapping[str, object], tier_key: str | None, layer_key: str | None
) -> tuple[tuple[Mapping[str, object], str | None], ...]:
    return ((_provider_entry(prices), layer_key), (prices, tier_key), (prices, "output_cost_per_image"))


def _output_price(
    sources: tuple[Mapping[str, object], ...],
    fields: Mapping[str, object],
    layer_decomposition: bool,
) -> float:
    tier: Final = _tier(_pixels(fields.get("size")))
    tier_key: Final = f"output_cost_per_image_{tier}" if tier is not None else None
    layer_key: Final = f"layer_output_cost_per_image_{tier}" if tier is not None and layer_decomposition else None
    lookups: Final = chain.from_iterable(_price_lookups(prices, tier_key, layer_key) for prices in sources)
    prices: Final = (_number(table, key) for table, key in lookups if key is not None)
    return next((price for price in prices if price is not None), 0.0)


def _input_cost(pricing: Mapping[str, object], image_response: ImageResponse) -> float:
    input_price: Final = _number(pricing, "input_cost_per_image")
    if input_price is None:
        return 0.0
    usage: Final = _mapping(image_response.usage.model_dump() if image_response.usage is not None else {})
    input_images: Final = _number(usage, "input_images") or 0.0
    free_input_images: Final = _number(_provider_entry(pricing), "free_input_images") or 0.0
    return input_price * max(input_images - free_input_images, 0.0)


def cost_calculator(
    model: str,
    image_response: ImageResponse,
    model_info: ModelInfo | None = None,
) -> float:
    pricing: Final = resolve_image_model_info(model=model, custom_llm_provider="volcengine", model_info=model_info)
    sources: Final = (model_info or {}, pricing)
    images: Final = _generated_images(image_response)
    layer_decomposition: Final = any(_is_layer(fields) for fields in images)
    output_cost: Final = sum(_output_price(sources, fields, layer_decomposition) for fields in images)
    return output_cost + _input_cost(pricing, image_response)
