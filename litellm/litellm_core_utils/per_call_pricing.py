"""The price entry for a single call that sets its own custom pricing, built without writing ``litellm.model_cost``."""

from collections.abc import Mapping
from typing import Final, Literal

from pydantic import TypeAdapter

import litellm
from litellm.litellm_core_utils.ptu_pricing import zeroed_ptu_pricing
from litellm.types.utils import CustomPricingLiteLLMParams, ModelInfo

_STR_MAPPING: Final = TypeAdapter(Mapping[str, object])
_MODEL_INFO: Final = TypeAdapter(ModelInfo)


def _router_deployment_model_info(kwargs: Mapping[str, object]) -> Mapping[str, object] | None:
    """A Router stamps the deployment's ``model_info`` (with its ``id``) into the call's metadata and registers that
    deployment's full custom pricing under that id when the deployment is added."""
    for metadata_key in ("litellm_metadata", "metadata"):
        metadata = kwargs.get(metadata_key)
        if not isinstance(metadata, Mapping):
            continue
        model_info = _STR_MAPPING.validate_python(metadata).get("model_info")
        if isinstance(model_info, Mapping) and isinstance(model_info.get("id"), str):
            return _STR_MAPPING.validate_python(model_info)
    return None


def per_call_model_info(
    model: str, custom_llm_provider: str, kwargs: Mapping[str, object], mode: Literal["chat", "embedding"]
) -> ModelInfo | None:
    """The full price entry for one call, or None when the cost map (or the Router's registered deployment entry)
    already prices it. The cost map row for the model is overlaid with the registered deployment entry and then with
    the custom pricing fields the request itself set, so a Router call pays request overrides over deployment prices.
    The entry travels on the logging object and is never written into ``litellm.model_cost``, so the next call for the
    same model name is priced from the cost map again."""
    deployment: Final = _router_deployment_model_info(kwargs)
    registered: Final = litellm.model_cost.get(str(deployment["id"])) if deployment is not None else None
    custom: Final = {
        field: kwargs[field]
        for field in CustomPricingLiteLLMParams.model_fields
        if kwargs.get(field) is not None and (registered is None or registered.get(field) != kwargs[field])
    }
    if not custom:
        return None
    unmapped: Final[Mapping[str, object]] = {
        "key": model,
        "max_tokens": None,
        "max_input_tokens": None,
        "max_output_tokens": None,
        "input_cost_per_token": None,
        "output_cost_per_token": None,
        "litellm_provider": custom_llm_provider,
        "mode": mode,
        "supported_openai_params": None,
    }
    row: Final = _catalog_row(model, custom_llm_provider) or unmapped
    catalog: Final = (
        {**row, **{key: value for key, value in registered.items() if value is not None}} if registered else row
    )
    zeroed: Final = zeroed_ptu_pricing(deployment, custom) if deployment is not None else None
    return _MODEL_INFO.validate_python(
        {
            **catalog,
            **_token_rates_cleared_for_per_second_overrides(custom),
            **custom,
            **_tiers_with_flat_overrides(catalog, custom),
            **(zeroed or {}),
            "mode": catalog.get("mode") or mode,
        }
    )


_FLAT_TOKEN_RATE_FIELDS: Final = ("input_cost_per_token", "output_cost_per_token")
_PER_SECOND_RATE_FIELDS: Final = ("cost_per_second", "input_cost_per_second", "output_cost_per_second")
_TIERS: Final = TypeAdapter(tuple[Mapping[str, object], ...])


def _token_rates_cleared_for_per_second_overrides(custom: Mapping[str, object]) -> Mapping[str, object]:
    """Token rates in the entry make the cost calculator skip per-second pricing, so a caller who sets only a
    per-second rate gets the catalog's token rates and tier table cleared for this call."""
    if not any(key in custom for key in _PER_SECOND_RATE_FIELDS) or any(
        key in custom for key in _FLAT_TOKEN_RATE_FIELDS
    ):
        return {}
    return {"input_cost_per_token": None, "output_cost_per_token": None, "tiered_pricing": None}


def _tiers_with_flat_overrides(catalog: Mapping[str, object], custom: Mapping[str, object]) -> Mapping[str, object]:
    """A ``tiered_pricing`` table wins over flat rates in the cost calculator, so a flat rate the caller set is written
    into every tier of the table in force (the caller's own, else the inherited one) for that direction, and the other
    direction keeps each tier's own rate."""
    overrides: Final = {key: custom[key] for key in _FLAT_TOKEN_RATE_FIELDS if key in custom}
    tiers: Final = custom.get("tiered_pricing", catalog.get("tiered_pricing"))
    if not overrides or not isinstance(tiers, (list, tuple)) or not tiers:
        return {}
    return {"tiered_pricing": [{**tier, **overrides} for tier in _TIERS.validate_python(tiers)]}


def _catalog_row(model: str, custom_llm_provider: str) -> ModelInfo | None:
    try:
        return litellm.get_model_info(model=model, custom_llm_provider=custom_llm_provider)
    except Exception:  # noqa: BLE001  # get_model_info raises a bare Exception for an unmapped model
        return None
