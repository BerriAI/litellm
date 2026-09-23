from collections.abc import Mapping
from typing import Any, Final

import litellm
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    _get_cost_per_unit,
    calculate_image_response_cost_from_usage,
    resolve_image_model_info,
)
from litellm.llms.azure_ai.image_edit.flux2_transformation import REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM
from litellm.types.utils import ImageResponse, ModelInfo


def _pixel_rate(resolved: ModelInfo, cost_key: str) -> float:
    deployment_price: Final = _get_cost_per_unit(resolved, cost_key, default_value=None)
    if deployment_price is not None:
        return deployment_price
    model_cost_key: Final = resolved.get("key")
    shared_entry: Final = litellm.model_cost.get(model_cost_key) if model_cost_key is not None else None
    if shared_entry is None:
        return 0.0
    return shared_entry.get(cost_key) or 0.0


def _reference_cost(resolved: ModelInfo, image_response: ImageResponse) -> float:
    pixels: Final = image_response._hidden_params.get(REFERENCE_IMAGE_PIXELS_HIDDEN_PARAM)
    if type(pixels) is not int or pixels <= 0:
        return 0.0
    return _pixel_rate(resolved, "input_cost_per_reference_pixel") * pixels


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
    output_cost_per_image: Final[float] = resolved.get("output_cost_per_image") or 0.0
    if output_cost_per_image:
        return output_cost_per_image * num_images
    if not _pixel_rate(resolved, "input_cost_per_pixel"):
        return 0.0

    from litellm.cost_calculator import default_image_cost_calculator

    width: Final = optional_params.get("width") if optional_params else None
    height: Final = optional_params.get("height") if optional_params else None
    pixel_size: Final = (
        f"{width}x{height}"
        if type(width) is int and type(height) is int and width > 0 and height > 0
        else size or image_response.size
    )
    return default_image_cost_calculator(
        model=resolved.get("key", model),
        custom_llm_provider=litellm.LlmProviders.AZURE_AI.value,
        size=pixel_size,
        n=num_images,
        model_info=model_info,
    )
