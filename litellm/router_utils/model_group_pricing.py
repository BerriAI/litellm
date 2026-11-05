from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.litellm_logging import deployment_pricing_model_info
from litellm.types.router import DeploymentPricing, PriceTier

_RAW_PRICE_TIERS: Final = TypeAdapter(tuple[Mapping[str, object], ...])


def _cost_value(value: object) -> float | None:
    match value:
        case bool():
            return None
        case int() | float():
            return float(value)
        case _:
            return None


def _cost_fields(fields: Mapping[str, object]) -> Mapping[str, float]:
    return {name: cost for name, value in fields.items() if "cost" in name and (cost := _cost_value(value)) is not None}


def _price_tiers(tiered_pricing: object) -> tuple[PriceTier, ...]:
    try:
        return tuple(
            PriceTier.model_validate({"range": tier.get("range"), "costs": _cost_fields(tier)})
            for tier in _RAW_PRICE_TIERS.validate_python(tiered_pricing)
        )
    except ValidationError:
        return ()


def deployment_pricing(
    provider: str, deployment_id: str | None, deployment_model: str, resolved_model_info: Mapping[str, object]
) -> DeploymentPricing | None:
    """Prices from the entry billing charges this deployment at, so a cost-map variant its own pricing shadows
    is never shown. A deployment that declares no pricing is billed at its resolved cost-map entry."""
    billed_model_info: Final = deployment_pricing_model_info(model_id=deployment_id, deployment_model=deployment_model)
    model_info: Final = resolved_model_info if billed_model_info is None else billed_model_info
    costs: Final = _cost_fields(model_info)
    tiers: Final = _price_tiers(model_info.get("tiered_pricing"))
    if not costs and not tiers:
        return None
    return DeploymentPricing(provider=provider, costs=costs, tiered_pricing=tiers)
