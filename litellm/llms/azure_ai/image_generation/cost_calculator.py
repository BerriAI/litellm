import math
from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, ValidationError

import litellm
from litellm.litellm_core_utils.hidden_params import set_hidden_param
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_image_response_cost_from_usage,
    get_cost_per_unit,
    resolve_image_model_info,
)
from litellm.types.utils import ImageResponse, ModelInfo

_MEGAPIXEL_USAGE_HIDDEN_PARAM: Final = "megapixel_usage"
_MEGAPIXEL_PRICE_KEYS: Final = (
    "output_cost_per_first_megapixel",
    "output_cost_per_megapixel",
    "input_cost_per_megapixel",
)


class _MegapixelUsage(BaseModel):
    model_config = ConfigDict(frozen=True, allow_inf_nan=False)

    input_mp: float | None = Field(default=None, ge=0)
    output_mp: float = Field(gt=0)


class _MeteredImageResponse(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_meta: _MegapixelUsage


def record_megapixel_usage(response: object, response_body: bytes | Mapping[str, object]) -> None:
    try:
        usage: Final = (
            _MeteredImageResponse.model_validate_json(response_body)
            if isinstance(response_body, bytes)
            else _MeteredImageResponse.model_validate(response_body)
        ).request_meta
    except ValidationError:
        return
    set_hidden_param(response, _MEGAPIXEL_USAGE_HIDDEN_PARAM, usage.model_dump())


def _megapixel_cost(prices: ModelInfo, image_response: ImageResponse) -> float | None:
    output_rate: Final = get_cost_per_unit(prices, "output_cost_per_megapixel", default_value=None)
    if output_rate is None:
        return None
    try:
        usage: Final = _MegapixelUsage.model_validate(image_response.hidden_params.get(_MEGAPIXEL_USAGE_HIDDEN_PARAM))
    except ValidationError:
        return None
    first_rate: Final = get_cost_per_unit(prices, "output_cost_per_first_megapixel", default_value=None)
    input_rate: Final = get_cost_per_unit(prices, "input_cost_per_megapixel") or 0.0
    images: Final = max(len(image_response.data or ()), 1)
    output_mp_per_image: Final = usage.output_mp / images
    return images * (
        (output_rate if first_rate is None else first_rate) + output_rate * (math.ceil(output_mp_per_image) - 1)
    ) + input_rate * math.ceil(usage.input_mp or 0)


def _input_cost_per_pixel(resolved: ModelInfo) -> float:
    deployment_price: Final = get_cost_per_unit(resolved, "input_cost_per_pixel", default_value=None)
    if deployment_price is not None:
        return deployment_price
    model_cost_key: Final = resolved.get("key")
    shared_entry: Final = litellm.model_cost.get(model_cost_key) if model_cost_key is not None else None
    if shared_entry is None:
        return 0.0
    return shared_entry.get("input_cost_per_pixel") or 0.0


def cost_calculator(
    model: str,
    image_response: object,
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

        bills_megapixels: Final = model_info is None or any(
            model_info.get(key) is not None for key in _MEGAPIXEL_PRICE_KEYS
        )
        megapixel_cost: Final = _megapixel_cost(_model_info, image_response) if bills_megapixels else None
        if megapixel_cost is not None:
            return megapixel_cost

        num_images: Final = n if n is not None else len(image_response.data or ())
        output_cost_per_image: Final[float] = _model_info.get("output_cost_per_image") or 0.0
        if output_cost_per_image:
            return output_cost_per_image * num_images

        if _input_cost_per_pixel(_model_info):
            from litellm.cost_calculator import default_image_cost_calculator

            width: Final = optional_params.get("width") if optional_params else None
            height: Final = optional_params.get("height") if optional_params else None
            pixel_size: Final = (
                f"{width}x{height}"
                if type(width) is int and type(height) is int and width > 0 and height > 0
                else size or image_response.size
            )
            return default_image_cost_calculator(
                model=_model_info.get("key", model),
                custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
                size=pixel_size,
                n=num_images,
                model_info=model_info,
            )
        return 0.0

    raise ValueError(f"image_response must be of type ImageResponse got type={type(image_response)}")
