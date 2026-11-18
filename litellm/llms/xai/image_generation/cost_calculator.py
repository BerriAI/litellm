from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter

from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import ImageDocument, price_units
from litellm.litellm_core_utils.llm_cost_calc.utils import resolve_image_model_info
from litellm.llms.xai.common_utils import xai_reported_cost_in_usd
from litellm.rust_bridge._native import NativeReceipt
from litellm.types.utils import ImageResponse, ModelInfo

_PARAMS: Final = TypeAdapter(Mapping[str, object])


def _request_param(optional_params: Mapping[str, object] | None, key: str) -> object:
    if optional_params is None:
        return None
    if key in optional_params:
        return optional_params[key]
    extra_body: Final = optional_params.get("extra_body")
    return _PARAMS.validate_python(extra_body).get(key) if isinstance(extra_body, Mapping) else None


def _pricing(model: str, model_info: ModelInfo | None) -> ModelInfo | None:
    try:
        return resolve_image_model_info(model, "xai", model_info)
    except Exception:  # noqa: BLE001  # model lookup raises bare Exception for unmapped models
        return None


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
    reported_cost: Final = xai_reported_cost_in_usd(getattr(image_response.usage, "cost_in_usd_ticks", None))
    billed_cost: Final = None if caller_set_prices else reported_cost

    requested_quality: Final = quality or image_response.quality or _request_param(optional_params, "quality") or "auto"
    resolution: Final = _request_param(optional_params, "resolution") or "1k"
    is_edit: Final = call_type in ("image_edit", "aimage_edit")
    low_quality: Final = requested_quality == "low" or (requested_quality == "auto" and not is_edit)
    if billed_cost is None and not (low_quality and resolution in ("1k", "2k")):
        return None
    outcome: Final = price_units(
        ImageDocument(
            kind="image",
            model=model,
            provider="xai",
            calculator="xai",
            images=n,
            pixels=0,
            input_images=0,
            size=str(resolution).upper(),
            reported_usd=billed_cost,
        ),
        (model_info,) if model_info is not None else (),
        _pricing(model, model_info) if billed_cost is None else None,
    )
    return outcome.total if isinstance(outcome, NativeReceipt) else None
