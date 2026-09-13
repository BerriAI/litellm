from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

import litellm
from litellm.types.utils import ImageResponse

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


def _keyed_size(model: str, optional_params: Mapping[str, object]) -> str | None:
    image_size: Final = optional_params.get("image_size")
    if image_size is None:
        return None if model.endswith("/edit") else FAL_TEXT_TO_IMAGE_DEFAULT_SIZE
    if isinstance(image_size, Mapping):
        width: Final = image_size.get("width")
        height: Final = image_size.get("height")
        if isinstance(width, int) and isinstance(height, int):
            return f"{width}-x-{height}"
        return None
    if isinstance(image_size, str):
        return FAL_NAMED_IMAGE_SIZES.get(image_size)
    return None


def _keyed_cost_per_image(model: str, optional_params: Mapping[str, object] | None) -> float | None:
    if optional_params is None:
        return None
    size: Final = _keyed_size(model=model, optional_params=optional_params)
    if size is None:
        return None
    raw_quality: Final = optional_params.get("quality")
    quality: Final = (
        raw_quality if isinstance(raw_quality, str) and raw_quality != "auto" else FAL_KEYED_PRICING_DEFAULT_QUALITY
    )
    keyed_entry: Final = litellm.model_cost.get(f"fal_ai/{quality}/{size}/{model}")
    if keyed_entry is None:
        return None
    keyed_cost: Final = keyed_entry.get("output_cost_per_image")
    return float(keyed_cost) if isinstance(keyed_cost, (int, float)) else None


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
    # the proxy cost path passes the provider-prefixed model name
    model = model.removeprefix(f"{litellm.LlmProviders.FAL_AI.value}/")
    num_images: Final[int] = len(image_response.data) if image_response.data else 0
    keyed_cost_per_image: Final = _keyed_cost_per_image(model=model, optional_params=optional_params)
    if keyed_cost_per_image is not None:
        return keyed_cost_per_image * num_images
    _model_info: Final = litellm.get_model_info(
        model=model,
        custom_llm_provider=litellm.LlmProviders.FAL_AI.value,
    )
    output_cost_per_image: Final[float] = _model_info.get("output_cost_per_image") or 0.0
    return output_cost_per_image * num_images
