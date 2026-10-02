import base64
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Annotated, Final

from pydantic import Field, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _get_cost_per_unit,
    calculate_image_response_cost_from_usage,
    resolve_image_model_info,
)
from litellm.litellm_core_utils.token_counter import get_image_type, image_pixels_from_bytes
from litellm.llms.azure_ai.image_generation.flux_transformation import AzureFoundryFluxImageGenerationConfig
from litellm.types.utils import ImageResponse, ModelInfo

MEGAPIXEL: Final = 1024 * 1024
MAX_LONE_REFERENCE_MEGAPIXELS: Final = 4
IMAGE_HEADER_BASE64_PREFIX_CHARS: Final = 64 * 1024
JPEG_HEADER_BASE64_PREFIX_CHARS: Final = 512 * 1024
REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM: Final = "reference_image_pixels"
FLUX2_REFERENCE_IMAGE_FIELDS: Final = ("input_image", *(f"input_image_{index}" for index in range(2, 11)))
_MEGAPIXEL_PRICE_KEYS: Final = (
    "output_cost_per_image_first_megapixel",
    "output_cost_per_pixel",
    "input_cost_per_megapixel",
)
_BASE64_WHITESPACE: Final = MappingProxyType(dict.fromkeys(map(ord, " \t\r\n\v\f")))
_DATA_URL_HEADER_MAX_CHARS: Final = 256
_REFERENCE_PIXELS: Final = TypeAdapter(tuple[Annotated[int, Field(strict=True, gt=0)] | None, ...])


def registered_image_prices(
    model_key: str, merged: Mapping[str, object], overrides: Mapping[str, object]
) -> Mapping[str, object]:
    if not model_key.lower().startswith("azure_ai/flux.2-"):
        return merged
    legacy_fields: Final = ("output_cost_per_image", "input_cost_per_image", "input_cost_per_pixel")
    legacy_override: Final = any(overrides.get(key) is not None for key in legacy_fields)
    megapixel_override: Final = any(overrides.get(key) is not None for key in _MEGAPIXEL_PRICE_KEYS)
    if legacy_override and not megapixel_override:
        return MappingProxyType({**merged, **dict.fromkeys(_MEGAPIXEL_PRICE_KEYS)})
    return merged


@dataclass(frozen=True, slots=True, kw_only=True)
class _Flux2Prices:
    first_megapixel: float | None
    generated_image: float
    generated_megapixel: float
    reference_megapixel: float


def _price(resolved: ModelInfo, cost_key: str) -> float | None:
    deployment_price: Final = _get_cost_per_unit(resolved, cost_key, default_value=None)
    if deployment_price is not None:
        return deployment_price
    model_cost_key: Final = resolved.get("key")
    shared_entry: Final = litellm.model_cost.get(model_cost_key) if model_cost_key is not None else None
    if shared_entry is None:
        return None
    return _get_cost_per_unit(shared_entry, cost_key, default_value=None)


def _pixel_rate(resolved: ModelInfo) -> float:
    return _price(resolved, "input_cost_per_pixel") or 0.0


def _deployment_price(deployment: ModelInfo | None, cost_key: str) -> float | None:
    if deployment is None:
        return None
    return _get_cost_per_unit(deployment, cost_key, default_value=None)


def _flux2_prices(resolved: ModelInfo, deployment: ModelInfo | None) -> _Flux2Prices | None:
    price_keys: Final = (
        *_MEGAPIXEL_PRICE_KEYS,
        "output_cost_per_image",
        "input_cost_per_image",
        "input_cost_per_pixel",
    )
    deployment_prices: Final = MappingProxyType({key: _deployment_price(deployment, key) for key in price_keys})
    deployment_priced: Final = any(price is not None for price in deployment_prices.values())
    prices: Final = MappingProxyType(
        {key: deployment_prices[key] if deployment_priced else _price(resolved, key) for key in price_keys}
    )
    if all(prices[key] is None for key in _MEGAPIXEL_PRICE_KEYS):
        return None
    first_megapixel: Final = prices["output_cost_per_image_first_megapixel"]
    flat_price: Final = (
        prices["output_cost_per_image"]
        if prices["output_cost_per_image"] is not None
        else prices["input_cost_per_image"]
    )
    return _Flux2Prices(
        first_megapixel=first_megapixel,
        generated_image=flat_price or 0.0,
        generated_megapixel=(
            (prices["output_cost_per_pixel"] or 0.0) * MEGAPIXEL
            if first_megapixel is not None or flat_price is None
            else 0.0
        ),
        reference_megapixel=prices["input_cost_per_megapixel"] or 0.0,
    )


def _billable_megapixels(pixels: int) -> int:
    return (pixels + MEGAPIXEL - 1) // MEGAPIXEL


def record_reference_pixels(image_response: ImageResponse, reference_pixels: tuple[int | None, ...]) -> None:
    image_response._hidden_params[REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM] = reference_pixels


def record_request_reference_pixels(
    image_response: ImageResponse, model_or_endpoint: str, request_data: Mapping[str, object]
) -> ImageResponse:
    if not AzureFoundryFluxImageGenerationConfig.is_flux2_model(model_or_endpoint):
        return image_response
    references: Final = tuple(request_data.get(field) for field in FLUX2_REFERENCE_IMAGE_FIELDS)
    reference_pixels: Final = tuple(_reference_pixels(value) for value in references if value not in (None, ""))
    if reference_pixels:
        record_reference_pixels(image_response, reference_pixels)
    return image_response


def _reference_pixels(value: object) -> int | None:
    if not isinstance(value, str):
        return None
    data_url_header, separator, _ = value[:_DATA_URL_HEADER_MAX_CHARS].partition("base64,")
    return base64_image_pixels(
        value,
        start=len(data_url_header) + len(separator) if separator else 0,
        read_whole_jpeg=False,
    )


def _billable_reference_megapixels(model: str, reference_pixels: tuple[int | None, ...]) -> int:
    # Azure's FLUX.2 [pro] request_meta (2026-09-25) bills a lone reference at no more than 4 MP and each reference
    # of a multi-reference edit as exactly 1 MP whatever its size. FLUX.2 [flex] is assumed to match
    match reference_pixels:
        case ():
            return 0
        case (None,):
            verbose_logger.warning(
                "Could not read the dimensions of the %s reference image, billing the %d megapixel maximum",
                model,
                MAX_LONE_REFERENCE_MEGAPIXELS,
            )
            return MAX_LONE_REFERENCE_MEGAPIXELS
        case (int() as lone_reference,):
            return min(_billable_megapixels(lone_reference), MAX_LONE_REFERENCE_MEGAPIXELS)
        case _:
            return len(reference_pixels)


def _reference_cost(model: str, prices: _Flux2Prices, image_response: ImageResponse) -> float:
    reported_pixels: Final = image_response._hidden_params.get(REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM)
    if reported_pixels is None:
        return 0.0
    try:
        reference_pixels: Final = _REFERENCE_PIXELS.validate_python(reported_pixels)
    except ValidationError:
        verbose_logger.warning("Ignoring malformed FLUX.2 reference pixel counts: %r", reported_pixels)
        return 0.0
    return prices.reference_megapixel * _billable_reference_megapixels(model, reference_pixels)


def _flux2_generated_cost(
    model: str, prices: _Flux2Prices, image_response: ImageResponse, requested_pixels: int, n: int | None
) -> float:
    return sum(
        _generated_image_cost(prices, pixels)
        for pixels in _generated_pixels(model, image_response, requested_pixels, n)
    )


def _generated_image_cost(prices: _Flux2Prices, pixels: int) -> float:
    megapixels: Final = _billable_megapixels(pixels)
    if prices.first_megapixel is not None:
        return prices.first_megapixel + prices.generated_megapixel * max(megapixels - 1, 0)
    return prices.generated_image + prices.generated_megapixel * megapixels


def _generated_pixels(
    model: str, image_response: ImageResponse, requested_pixels: int, n: int | None
) -> tuple[int, ...]:
    images: Final = image_response.data or ()
    if not images:
        return (requested_pixels,) * (n or 0)
    return tuple(_billed_output_pixels(model, image.b64_json, requested_pixels) for image in images)


def _billed_output_pixels(model: str, b64_json: str | None, requested_pixels: int) -> int:
    if not b64_json:
        return requested_pixels
    pixels: Final = base64_image_pixels(b64_json, read_whole_jpeg=True)
    if pixels is None:
        verbose_logger.warning(
            "Could not read the dimensions of the image %s returned, billing %d pixels instead", model, requested_pixels
        )
        return requested_pixels
    return pixels


def base64_image_pixels(encoded_image: str, start: int = 0, *, read_whole_jpeg: bool) -> int | None:
    header_end: Final = start + IMAGE_HEADER_BASE64_PREFIX_CHARS
    header_bytes: Final = _decoded(encoded_image[start:header_end], clipped=len(encoded_image) > header_end)
    header_pixels: Final = image_pixels_from_bytes(header_bytes)
    if header_pixels is not None or len(encoded_image) <= header_end or get_image_type(header_bytes) != "jpeg":
        return header_pixels
    jpeg_prefix_end: Final = start + JPEG_HEADER_BASE64_PREFIX_CHARS
    prefix_pixels: Final = image_pixels_from_bytes(
        _decoded(encoded_image[start:jpeg_prefix_end], clipped=len(encoded_image) > jpeg_prefix_end)
    )
    if prefix_pixels is not None or not read_whole_jpeg or len(encoded_image) <= jpeg_prefix_end:
        return prefix_pixels
    return image_pixels_from_bytes(_decoded(encoded_image[start:]))


def _decoded(encoded_image: str, *, clipped: bool = False) -> bytes:
    prefix: Final = encoded_image.translate(_BASE64_WHITESPACE) if clipped else encoded_image
    decodable: Final = prefix[: len(prefix) // 4 * 4] if clipped else prefix
    try:
        return base64.b64decode(decodable)
    except ValueError:
        return b""


def cost_calculator(
    model: str,
    image_response: object,
    size: str | None = None,
    n: int | None = None,
    optional_params: Mapping[str, object] | None = None,
    model_info: ModelInfo | None = None,
) -> float:
    """
    Azure AI image generation and image edit cost calculator
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

        if AzureFoundryFluxImageGenerationConfig.is_flux2_model(model):
            prices: Final = _flux2_prices(_model_info, model_info)
            if prices is not None:
                requested_pixels: Final = _size_pixels(_output_size(size, optional_params, image_response)) or MEGAPIXEL
                return _flux2_generated_cost(model, prices, image_response, requested_pixels, n) + _reference_cost(
                    model, prices, image_response
                )

        num_images: Final = n if n is not None else len(image_response.data or ())
        output_cost_per_image: Final[float] = _model_info.get("output_cost_per_image") or 0.0
        if output_cost_per_image:
            return output_cost_per_image * num_images
        if not _pixel_rate(_model_info):
            return 0.0

        from litellm.cost_calculator import default_image_cost_calculator

        return default_image_cost_calculator(
            model=_model_info.get("key", model),
            custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
            size=_output_size(size, optional_params, image_response),
            n=num_images,
            model_info=model_info,
        )

    raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")


def _output_size(
    size: str | None, optional_params: Mapping[str, object] | None, image_response: ImageResponse
) -> str | None:
    width: Final = optional_params.get("width") if optional_params else None
    height: Final = optional_params.get("height") if optional_params else None
    if type(width) is int and type(height) is int and width > 0 and height > 0:
        return f"{width}x{height}"
    return size or image_response.size


def _size_pixels(size: str | None) -> int | None:
    dimensions: Final = (size or "").lower().replace("-x-", "x").split("x")
    if len(dimensions) != 2 or not all(dimension.isdigit() for dimension in dimensions):
        return None
    return int(dimensions[0]) * int(dimensions[1])
