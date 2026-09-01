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
    # ``bool | None`` because the TypedDict sanctions None; None means "not set"
    # and keeps the default billed behavior, so a None-carrying entry must not
    # fail union validation and silently zero a sibling entry's real cost.
    guardrail_cost_in_spend: bool | None = True


_GUARDRAIL_COST_ENTRY_ADAPTER: Final[TypeAdapter[GuardrailCostEntry]] = TypeAdapter(GuardrailCostEntry)


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


AZURE_PROMPT_SHIELD_TEXT_RECORD_UNIT: Final = "text_records"


def azure_prompt_shield_guardrail_cost(
    usage_units: Mapping[str, int],
    cost_tier: str | None,
    price_per_1000_text_records: float | None,
) -> float | None:
    """USD cost of an Azure Prompt Shield invocation from its text-record count.

    Returns 0.0 on the free tier, ``text_records * price / 1000`` when a price is
    configured, and None when pricing is not configured (usage-only tracking).
    """
    if cost_tier == "free":
        return 0.0
    if price_per_1000_text_records is None:
        return None
    return usage_units.get(AZURE_PROMPT_SHIELD_TEXT_RECORD_UNIT, 0) * price_per_1000_text_records / 1000.0


def _billable_entry_cost(entry: GuardrailCostEntry) -> float:
    if entry.guardrail_cost_in_spend is False:
        return 0.0
    cost: Final = entry.guardrail_cost
    if cost is None or not math.isfinite(cost) or cost <= 0.0:
        return 0.0
    return cost


def _validated_entry_cost(raw: object) -> float:
    """Billable cost of one raw ``guardrail_information`` entry.

    Validated per entry so one malformed entry (e.g. a custom hook stamping a
    non-boolean ``guardrail_cost_in_spend``) prices to 0.0 by itself instead of
    failing a whole-payload validation and silently zeroing a sibling entry's
    real billable cost."""
    try:
        return _billable_entry_cost(_GUARDRAIL_COST_ENTRY_ADAPTER.validate_python(raw))
    except ValidationError as e:
        verbose_logger.warning("Ignoring malformed guardrail_information entry for guardrail cost: %s", e)
        return 0.0


def guardrail_information_cost(guardrail_information: object) -> float:
    if guardrail_information is None:
        return 0.0
    if isinstance(guardrail_information, (list, tuple)):
        return sum(_validated_entry_cost(entry) for entry in guardrail_information)
    return _validated_entry_cost(guardrail_information)


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
