import math
from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_logger
from litellm.types.utils import CostBreakdown

BEDROCK_GUARDRAIL_PRICING_KEY: Final = "bedrock/guardrails"


class GuardrailPricing(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    guardrail_cost_per_unit: Mapping[str, float]


class GuardrailCostEntry(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    guardrail_cost: float | None = None


GuardrailInformationShape = tuple[GuardrailCostEntry, ...] | GuardrailCostEntry | None

_GUARDRAIL_INFORMATION_ADAPTER: Final[TypeAdapter[GuardrailInformationShape]] = TypeAdapter(GuardrailInformationShape)


def _bedrock_guardrail_pricing(aws_region_name: str | None) -> GuardrailPricing | None:
    regional_key: Final = f"bedrock/{aws_region_name}/guardrails" if aws_region_name else None
    for key in (regional_key, BEDROCK_GUARDRAIL_PRICING_KEY):
        if key is None or key not in litellm.model_cost:
            continue
        try:
            return GuardrailPricing.model_validate(litellm.model_cost[key])
        except ValidationError as e:
            verbose_logger.warning("Ignoring malformed guardrail pricing entry %s: %s", key, e)
    return None


def bedrock_guardrail_cost(usage_units: Mapping[str, int], aws_region_name: str | None) -> float:
    pricing: Final = _bedrock_guardrail_pricing(aws_region_name)
    if pricing is None:
        return 0.0
    return sum(units * pricing.guardrail_cost_per_unit.get(counter, 0.0) for counter, units in usage_units.items())


def _billable_entry_cost(entry: GuardrailCostEntry) -> float:
    cost: Final = entry.guardrail_cost
    if cost is None or not math.isfinite(cost) or cost <= 0.0:
        return 0.0
    return cost


def guardrail_information_cost(guardrail_information: object) -> float:
    try:
        parsed: Final = _GUARDRAIL_INFORMATION_ADAPTER.validate_python(guardrail_information)
    except ValidationError:
        return 0.0
    if parsed is None:
        return 0.0
    if isinstance(parsed, GuardrailCostEntry):
        return _billable_entry_cost(parsed)
    return sum(_billable_entry_cost(entry) for entry in parsed)


def cost_breakdown_with_guardrail(cost_breakdown: CostBreakdown | None, guardrail_cost: float) -> CostBreakdown | None:
    if guardrail_cost <= 0.0:
        return cost_breakdown
    existing: Final[CostBreakdown] = cost_breakdown if cost_breakdown is not None else CostBreakdown()
    merged: Final[CostBreakdown] = {
        **existing,
        "guardrail_cost": guardrail_cost,
        "total_cost": existing.get("total_cost", 0.0) + guardrail_cost,
    }
    return merged
