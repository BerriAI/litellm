import math
from collections.abc import Mapping
from typing import Annotated, Any, Final

from pydantic import Field, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _get_cost_per_unit,
    calculate_image_response_cost_from_usage,
    resolve_image_model_info,
)
from litellm.llms.azure_ai.image_edit.flux2_transformation import REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM
from litellm.llms.azure_ai.image_generation.flux_transformation import AzureFoundryFluxImageGenerationConfig
from litellm.types.utils import ImageResponse, ModelInfo

MEGAPIXEL: Final = 1024 * 1024
MAX_LONE_REFERENCE_MEGAPIXELS: Final = 4
_REFERENCE_PIXELS: Final = TypeAdapter(tuple[Annotated[int, Field(strict=True, gt=0)], ...])


def _price(resolved: ModelInfo, cost_key: str) -> float | None:
    deployment_price: Final = _get_cost_per_unit(resolved, cost_key, default_value=None)
    if deployment_price is not None:
        return deployment_price
    model_cost_key: Final = resolved.get("key")
    shared_entry: Final = litellm.model_cost.get(model_cost_key) if model_cost_key is not None else None
    if shared_entry is None:
        return None
    return shared_entry.get(cost_key)


def _pixel_rate(resolved: ModelInfo, cost_key: str) -> float:
    return _price(resolved, cost_key) or 0.0


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


def _reference_cost(resolved: ModelInfo, image_response: ImageResponse) -> float:
    reported_pixels: Final = image_response._hidden_params.get(REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM)
    if reported_pixels is None:
        return 0.0
    try:
        reference_pixels: Final = _REFERENCE_PIXELS.validate_python(reported_pixels)
    except ValidationError:
        return 0.0
    return (
        _pixel_rate(resolved, "input_cost_per_reference_pixel")
        * MEGAPIXEL
        * _billable_reference_megapixels(reference_pixels)
    )


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

        return _generated_cost(
            model=model,
            resolved=_model_info,
            image_response=image_response,
            size=size,
            n=n,
            optional_params=optional_params,
            model_info=model_info,
        ) + _reference_cost(_model_info, image_response)

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
    pixel_size: Final = _output_size(size, optional_params, image_response)
    if AzureFoundryFluxImageGenerationConfig.is_flux2_model(model):
        return num_images * _flux2_image_cost(resolved, _size_pixels(pixel_size))
    output_cost_per_image: Final[float] = resolved.get("output_cost_per_image") or 0.0
    if output_cost_per_image:
        return output_cost_per_image * num_images
    if not _pixel_rate(resolved, "input_cost_per_pixel"):
        return 0.0

    from litellm.cost_calculator import default_image_cost_calculator

    return default_image_cost_calculator(
        model=resolved.get("key", model),
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
        size=pixel_size,
        n=num_images,
        model_info=model_info,
    )


def _flux2_image_cost(resolved: ModelInfo, pixels: int) -> float:
    megapixel_rate: Final = _pixel_rate(resolved, "input_cost_per_pixel") * MEGAPIXEL
    first_megapixel_price: Final = _price(resolved, "output_cost_per_image")
    first_megapixel: Final = megapixel_rate if first_megapixel_price is None else first_megapixel_price
    return first_megapixel + megapixel_rate * (_billable_megapixels(pixels) - 1)


def _output_size(
    size: str | None, optional_params: Mapping[str, object] | None, image_response: ImageResponse
) -> str | None:
    width: Final = optional_params.get("width") if optional_params else None
    height: Final = optional_params.get("height") if optional_params else None
    if type(width) is int and type(height) is int and width > 0 and height > 0:
        return f"{width}x{height}"
    return size or image_response.size


def _size_pixels(size: str | None) -> int:
    width, height = (int(dimension) for dimension in (size or "1024x1024").replace("-x-", "x").split("x"))
    return width * height
