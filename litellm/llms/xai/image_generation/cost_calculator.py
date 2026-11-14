from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.llm_cost_calc.utils import get_cost_per_unit, resolve_image_model_info
from litellm.llms.xai.common_utils import xai_reported_cost_in_usd
from litellm.types.utils import ImageResponse, ModelInfo

_PARAMS: Final = TypeAdapter(Mapping[str, object])


def _request_param(optional_params: Mapping[str, object] | None, key: str) -> object:
    if optional_params is None:
        return None
    if key in optional_params:
        return optional_params[key]
    extra_body: Final = optional_params.get("extra_body")
    return _PARAMS.validate_python(extra_body).get(key) if isinstance(extra_body, Mapping) else None


def cost_calculator(
    model: str,
    image_response: ImageResponse,
    n: int,
    quality: str | None = None,
    optional_params: Mapping[str, object] | None = None,
    call_type: str | None = None,
    model_info: ModelInfo | None = None,
) -> float | None:
    if model_info is not None and model_info.get("input_cost_per_pixel") is not None:
        return None

    caller_set_prices: Final = model_info is not None and any(
        model_info.get(key) is not None
        for key in (
            "input_cost_per_image",
            "output_cost_per_image",
            "output_cost_per_image_1K",
            "output_cost_per_image_2K",
        )
    )
    billed_cost: Final = xai_reported_cost_in_usd(getattr(image_response.usage, "cost_in_usd_ticks", None))
    if billed_cost is not None and not caller_set_prices:
        return billed_cost

    requested_quality: Final = quality or image_response.quality or _request_param(optional_params, "quality") or "auto"
    resolution: Final = _request_param(optional_params, "resolution") or "1k"
    is_edit: Final = call_type in ("image_edit", "aimage_edit")
    low_quality: Final = requested_quality == "low" or (requested_quality == "auto" and not is_edit)
    if low_quality and resolution in ("1k", "2k"):
        try:
            pricing: Final = resolve_image_model_info(model, "xai", model_info)
        except Exception:  # noqa: BLE001  # model lookup raises bare Exception for unmapped models
            return None
        resolution_price: Final = get_cost_per_unit(pricing, f"output_cost_per_image_{str(resolution).upper()}", None)
        deployment_flat_price: Final = next(
            (
                get_cost_per_unit(model_info, key, None)
                for key in ("input_cost_per_image", "output_cost_per_image")
                if model_info is not None and model_info.get(key) is not None
            ),
            None,
        )
        if deployment_flat_price is not None:
            return deployment_flat_price * n
        if resolution_price is not None:
            return resolution_price * n

    return None
