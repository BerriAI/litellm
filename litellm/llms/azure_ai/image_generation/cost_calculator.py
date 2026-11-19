from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import ImageDocument, price_units
from litellm.litellm_core_utils.llm_cost_calc.utils import (
    calculate_image_response_cost_from_usage,
    resolve_image_model_info,
)
from litellm.rust_bridge._native import NativeReceipt
from litellm.types.utils import ImageResponse, ModelInfo

_ROW: Final = TypeAdapter(Mapping[str, object])


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

        num_images: Final = n if n is not None else len(image_response.data or ())
        model_cost_key: Final = _model_info.get("key")
        raw_shared_entry: Final[object] = litellm.model_cost.get(model_cost_key) if model_cost_key is not None else None
        shared_entry: Final = _ROW.validate_python(raw_shared_entry) if isinstance(raw_shared_entry, Mapping) else None
        outcome: Final = price_units(
            ImageDocument(
                kind="image",
                model=model,
                provider=litellm.LlmProviders.AZURE_AI.value,
                calculator="azure_ai",
                images=num_images,
                pixels=0,
                input_images=0,
            ),
            (shared_entry,) if shared_entry is not None else (),
            _model_info,
        )
        if isinstance(outcome, NativeReceipt):
            return outcome.total
        if outcome.reason == "not_ported":
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
