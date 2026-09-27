import base64
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Final

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
MAX_LONE_REFERENCE_PIXELS: Final = MAX_LONE_REFERENCE_MEGAPIXELS * MEGAPIXEL
IMAGE_HEADER_BASE64_PREFIX_CHARS: Final = 64 * 1024
JPEG_HEADER_BASE64_PREFIX_CHARS: Final = 512 * 1024
JPEG_HEADER_PREFIX_BYTES: Final = JPEG_HEADER_BASE64_PREFIX_CHARS * 3 // 4
REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM: Final = "reference_image_pixels"
DEPLOYMENT_PER_IMAGE_PRICE_KEYS: Final = ("output_cost_per_image", "input_cost_per_image")
_REFERENCE_PIXELS: Final = TypeAdapter(tuple[Annotated[int, Field(strict=True, gt=0)] | None, ...])


@dataclass(frozen=True, slots=True, kw_only=True)
class _Flux2MegapixelPrices:
    first_megapixel: float
    additional_megapixel: float


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


def _flux2_prices(resolved: ModelInfo, deployment: ModelInfo | None) -> _Flux2MegapixelPrices:
    deployment_image_price: Final = next(
        (price for key in DEPLOYMENT_PER_IMAGE_PRICE_KEYS if (price := _deployment_price(deployment, key)) is not None),
        None,
    )
    deployment_pixel_rate: Final = _deployment_price(deployment, "input_cost_per_pixel")
    deployment_megapixel_rate: Final = None if deployment_pixel_rate is None else deployment_pixel_rate * MEGAPIXEL
    match (deployment_image_price, deployment_megapixel_rate):
        case (float() as image_price, float() as megapixel_rate):
            return _Flux2MegapixelPrices(first_megapixel=image_price, additional_megapixel=megapixel_rate)
        case (float() as image_price, None):
            return _Flux2MegapixelPrices(first_megapixel=image_price, additional_megapixel=0.0)
        case (None, float() as megapixel_rate):
            return _Flux2MegapixelPrices(first_megapixel=megapixel_rate, additional_megapixel=megapixel_rate)
        case _:
            return _catalog_flux2_prices(resolved)


def _catalog_flux2_prices(resolved: ModelInfo) -> _Flux2MegapixelPrices:
    catalog_megapixel_rate: Final = _pixel_rate(resolved) * MEGAPIXEL
    catalog_first_megapixel: Final = _price(resolved, "output_cost_per_image")
    return _Flux2MegapixelPrices(
        first_megapixel=catalog_megapixel_rate if catalog_first_megapixel is None else catalog_first_megapixel,
        additional_megapixel=catalog_megapixel_rate,
    )


def _billable_megapixels(pixels: int) -> int:
    return math.ceil(pixels / MEGAPIXEL)


def record_reference_pixels(image_response: ImageResponse, reference_pixels: tuple[int | None, ...]) -> None:
    image_response._hidden_params[REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM] = reference_pixels


def _billable_reference_megapixels(model: str, reference_pixels: tuple[int | None, ...]) -> int:
    # Azure's FLUX.2 [pro] request_meta (2026-09-25) bills a lone reference at no more than 4 MP and each reference
    # of a multi-reference edit as exactly 1 MP whatever its size. FLUX.2 [flex] is assumed to match
    match reference_pixels:
        case ():
            return 0
        case (None,):
            verbose_logger.warning(
                "Could not read the dimensions of the %s reference image, billing it as one megapixel", model
            )
            return 1
        case (int() as lone_reference,):
            return min(_billable_megapixels(lone_reference), MAX_LONE_REFERENCE_MEGAPIXELS)
        case _:
            return len(reference_pixels)


def _reference_cost(model: str, prices: _Flux2MegapixelPrices, image_response: ImageResponse) -> float:
    reported_pixels: Final = image_response._hidden_params.get(REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM)
    if reported_pixels is None:
        return 0.0
    try:
        reference_pixels: Final = _REFERENCE_PIXELS.validate_python(reported_pixels)
    except ValidationError:
        verbose_logger.warning("Ignoring malformed FLUX.2 reference pixel counts: %r", reported_pixels)
        return 0.0
    return prices.additional_megapixel * _billable_reference_megapixels(model, reference_pixels)


def _flux2_generated_cost(
    model: str, prices: _Flux2MegapixelPrices, image_response: ImageResponse, requested_pixels: int, n: int | None
) -> float:
    return sum(
        prices.first_megapixel + prices.additional_megapixel * (_billable_megapixels(pixels) - 1)
        for pixels in _generated_pixels(model, image_response, requested_pixels, n)
    )


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
    header_bytes: Final = _decoded(encoded_image[start:header_end])
    header_pixels: Final = image_pixels_from_bytes(header_bytes)
    if header_pixels is not None or len(encoded_image) <= header_end or get_image_type(header_bytes) != "jpeg":
        return header_pixels
    jpeg_prefix_end: Final = start + JPEG_HEADER_BASE64_PREFIX_CHARS
    prefix_pixels: Final = image_pixels_from_bytes(_decoded(encoded_image[start:jpeg_prefix_end]))
    if prefix_pixels is not None or not read_whole_jpeg or len(encoded_image) <= jpeg_prefix_end:
        return prefix_pixels
    return image_pixels_from_bytes(_decoded(encoded_image[start:]))


def _decoded(encoded_image: str) -> bytes:
    try:
        return base64.b64decode(encoded_image)
    except ValueError:
        return b""


def cost_calculator(
    model: str,
    image_response: Any,
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


def estimate_flux2_cost(
    model: str,
    model_info: ModelInfo | None,
    request_params: Mapping[str, object],
    reference_pixels: tuple[int | None, ...],
) -> float:
    """
    Upper bound on what a FLUX.2 generation or edit will cost, for admission checks before the call.
    A reference of unknown size counts as the largest a lone reference is billed at, and an edit without
    a size is assumed to return its largest reference, capped the same way
    """
    prices: Final = _flux2_prices(
        resolve_image_model_info(
            model=model, custom_llm_provider=litellm.LlmProviders.AZURE_AI.value, model_info=model_info
        ),
        model_info,
    )
    references: Final = tuple(MAX_LONE_REFERENCE_PIXELS if pixels is None else pixels for pixels in reference_pixels)
    output_megapixels: Final = _billable_megapixels(_estimated_output_pixels(request_params, references))
    image_cost: Final = prices.first_megapixel + prices.additional_megapixel * (output_megapixels - 1)
    image_count: Final = max(_positive_int(request_params.get(key)) or 1 for key in ("n", "num_images"))
    return image_cost * image_count + prices.additional_megapixel * _billable_reference_megapixels(model, references)


def _estimated_output_pixels(request_params: Mapping[str, object], references: tuple[int, ...]) -> int:
    size: Final = request_params.get("size")
    requested_pixels: Final = _size_pixels(_requested_size(size if isinstance(size, str) else None, request_params))
    return requested_pixels or max((min(pixels, MAX_LONE_REFERENCE_PIXELS) for pixels in references), default=MEGAPIXEL)


def _output_size(
    size: str | None, optional_params: Mapping[str, object] | None, image_response: ImageResponse
) -> str | None:
    return _requested_size(size, optional_params) or image_response.size


def _requested_size(size: str | None, params: Mapping[str, object] | None) -> str | None:
    width: Final = _positive_int(params.get("width")) if params else None
    height: Final = _positive_int(params.get("height")) if params else None
    if width is not None and height is not None:
        return f"{width}x{height}"
    return size


def _positive_int(value: object) -> int | None:
    match value:
        case bool():
            return None
        case int() if value > 0:
            return value
        case str() if value.isdecimal() and int(value) > 0:
            return int(value)
        case _:
            return None


def _size_pixels(size: str | None) -> int | None:
    dimensions: Final = (size or "").lower().replace("-x-", "x").split("x")
    if len(dimensions) != 2 or not all(dimension.isdecimal() for dimension in dimensions):
        return None
    return int(dimensions[0]) * int(dimensions[1])
