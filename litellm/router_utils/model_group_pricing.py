import math
from collections.abc import Mapping
from typing import Final

from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.litellm_logging import deployment_pricing_model_info
from litellm.litellm_core_utils.llm_cost_calc.utils import get_vertex_regional_endpoint_uplift
from litellm.types.router import DeploymentPricing, PriceTier

_RAW_PRICE_TIERS: Final = TypeAdapter(tuple[Mapping[str, object], ...])


def _numeric_text(value: str) -> float | None:
    try:
        parsed: Final = float(value)
    except ValueError:
        return None
    return parsed if math.isfinite(parsed) else None


def _cost_value(value: object) -> float | None:
    match value:
        case bool():
            return None
        case int() | float():
            return float(value)
        case str():
            return _numeric_text(value)
        case _:
            return None


def _cost_fields(fields: Mapping[str, object], token_uplift: float) -> Mapping[str, float]:
    return {
        name: cost * token_uplift if "token" in name else cost
        for name, value in fields.items()
        if "cost" in name and (cost := _cost_value(value)) is not None
    }


def _price_tiers(tiered_pricing: object, token_uplift: float) -> tuple[PriceTier, ...]:
    try:
        return tuple(
            PriceTier.model_validate({"range": tier.get("range"), "costs": _cost_fields(tier, token_uplift)})
            for tier in _RAW_PRICE_TIERS.validate_python(tiered_pricing)
        )
    except ValidationError:
        return ()


def deployment_pricing(
    provider: str,
    deployment_id: str | None,
    deployment_model: str,
    resolved_model_info: Mapping[str, object],
    vertex_location: str | None,
) -> DeploymentPricing | None:
    """Prices from the entry billing charges this deployment at, so a cost-map variant its own pricing shadows
    is never shown. A deployment that declares no pricing is billed at its resolved cost-map entry. Token prices
    carry the Vertex regional-endpoint uplift billing applies at the deployment's location."""
    billed_model_info: Final = deployment_pricing_model_info(model_id=deployment_id, deployment_model=deployment_model)
    model_info: Final = resolved_model_info if billed_model_info is None else billed_model_info
    token_uplift: Final = get_vertex_regional_endpoint_uplift(model_info, vertex_location)
    costs: Final = _cost_fields(model_info, token_uplift)
    tiers: Final = _price_tiers(model_info.get("tiered_pricing"), token_uplift)
    if not costs and not tiers:
        return None
    return DeploymentPricing(provider=provider, costs=costs, tiered_pricing=tiers)
