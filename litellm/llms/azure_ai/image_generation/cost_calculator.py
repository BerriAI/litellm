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
from litellm.litellm_core_utils.token_counter import image_dimensions_from_bytes
from litellm.llms.azure_ai.image_edit.flux2_transformation import REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM
from litellm.llms.azure_ai.image_generation.flux_transformation import AzureFoundryFluxImageGenerationConfig
from litellm.types.utils import ImageObject, ImageResponse, ModelInfo

MEGAPIXEL: Final = 1024 * 1024
MAX_LONE_REFERENCE_MEGAPIXELS: Final = 4
_REFERENCE_PIXELS: Final = TypeAdapter(tuple[Annotated[int, Field(strict=True, gt=0)], ...])


@dataclass(frozen=True, slots=True, kw_only=True)
class _Flux2MegapixelPrices:
    first: float
    additional: float
    reference: float


def _price(resolved: ModelInfo, cost_key: str) -> float | None:
    deployment_price: Final = _get_cost_per_unit(resolved, cost_key, default_value=None)
    if deployment_price is not None:
        return deployment_price
    model_cost_key: Final = resolved.get("key")
    shared_entry: Final = litellm.model_cost.get(model_cost_key) if model_cost_key is not None else None
    if shared_entry is None:
        return None
    return _get_cost_per_unit(shared_entry, cost_key, default_value=None)


def _pixel_rate(resolved: ModelInfo, cost_key: str) -> float:
    return _price(resolved, cost_key) or 0.0


def _deployment_price(deployment: ModelInfo | None, cost_key: str) -> float | None:
    if deployment is None:
        return None
    return _get_cost_per_unit(deployment, cost_key, default_value=None)


def _flux2_prices(resolved: ModelInfo, deployment: ModelInfo | None) -> _Flux2MegapixelPrices:
    # A deployment that prices the generated image itself also prices its references, unless it sets a reference
    # rate: a flat per-image price covers them, and a pixel rate bills them at that rate
    deployment_reference_rate: Final = _deployment_price(deployment, "input_cost_per_reference_pixel")
    deployment_image_price: Final = _deployment_price(deployment, "output_cost_per_image")
    if deployment_image_price is not None:
        return _Flux2MegapixelPrices(
            first=deployment_image_price,
            additional=0.0,
            reference=(deployment_reference_rate or 0.0) * MEGAPIXEL,
        )
    deployment_pixel_rate: Final = _deployment_price(deployment, "input_cost_per_pixel")
    if deployment_pixel_rate is not None:
        return _Flux2MegapixelPrices(
            first=deployment_pixel_rate * MEGAPIXEL,
            additional=deployment_pixel_rate * MEGAPIXEL,
            reference=(deployment_pixel_rate if deployment_reference_rate is None else deployment_reference_rate)
            * MEGAPIXEL,
        )
    catalog_megapixel_rate: Final = _pixel_rate(resolved, "input_cost_per_pixel") * MEGAPIXEL
    catalog_first_megapixel: Final = _price(resolved, "output_cost_per_image")
    return _Flux2MegapixelPrices(
        first=catalog_megapixel_rate if catalog_first_megapixel is None else catalog_first_megapixel,
        additional=catalog_megapixel_rate,
        reference=_pixel_rate(resolved, "input_cost_per_reference_pixel") * MEGAPIXEL,
    )


def _billable_megapixels(pixels: int) -> int:
    return math.ceil(pixels / MEGAPIXEL)


def _billable_reference_megapixels(reference_pixels: tuple[int, ...]) -> int:
    # Azure's FLUX.2 request_meta (2026-09-25) bills a lone reference at no more than 4 MP, and each reference of a
    # multi-reference edit as exactly 1 MP whatever its size
    match reference_pixels:
        case ():
            return 0
        case (lone_reference,):
            return min(_billable_megapixels(lone_reference), MAX_LONE_REFERENCE_MEGAPIXELS)
        case _:
            return len(reference_pixels)


def _reference_cost(prices: _Flux2MegapixelPrices, image_response: ImageResponse) -> float:
    reported_pixels: Final = image_response._hidden_params.get(REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM)
    if reported_pixels is None:
        return 0.0
    try:
        reference_pixels: Final = _REFERENCE_PIXELS.validate_python(reported_pixels)
    except ValidationError:
        verbose_logger.warning("Ignoring malformed FLUX.2 reference pixel counts: %r", reported_pixels)
        return 0.0
    return prices.reference * _billable_reference_megapixels(reference_pixels)


def _flux2_generated_cost(
    prices: _Flux2MegapixelPrices, image_response: ImageResponse, requested_pixels: int, n: int | None
) -> float:
    return sum(
        prices.first + prices.additional * (_billable_megapixels(pixels) - 1)
        for pixels in _generated_pixels(image_response, requested_pixels, n)
    )


def _generated_pixels(image_response: ImageResponse, requested_pixels: int, n: int | None) -> tuple[int, ...]:
    images: Final = image_response.data or ()
    if not images:
        return (requested_pixels,) * (n or 0)
    return tuple(_measured_pixels(image) or requested_pixels for image in images)


def _measured_pixels(image: ImageObject) -> int | None:
    if not image.b64_json:
        return None
    try:
        image_bytes: Final = base64.b64decode(image.b64_json)
    except ValueError:
        return None
    dimensions: Final = image_dimensions_from_bytes(image_bytes)
    if dimensions is None:
        return None
    return dimensions[0] * dimensions[1] or None


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
            return _flux2_generated_cost(prices, image_response, requested_pixels, n) + _reference_cost(
                prices, image_response
            )

        return _generated_cost(
            model=model,
            resolved=_model_info,
            image_response=image_response,
            size=size,
            n=n,
            optional_params=optional_params,
            model_info=model_info,
        )

    raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")


def _generated_cost(
    model: str,
    resolved: ModelInfo,
    image_response: ImageResponse,
    size: str | None,
    n: int | None,
    optional_params: Mapping[str, object] | None,
    model_info: ModelInfo | None,
) -> float:
    num_images: Final = n if n is not None else len(image_response.data or ())
    output_cost_per_image: Final[float] = resolved.get("output_cost_per_image") or 0.0
    if output_cost_per_image:
        return output_cost_per_image * num_images
    if not _pixel_rate(resolved, "input_cost_per_pixel"):
        return 0.0

    from litellm.cost_calculator import default_image_cost_calculator

    return default_image_cost_calculator(
        model=resolved.get("key", model),
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
        size=_output_size(size, optional_params, image_response),
        n=num_images,
        model_info=model_info,
    )


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
