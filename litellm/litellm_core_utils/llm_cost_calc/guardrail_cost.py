import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Final

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import (
    object_items,
    object_mapping,
    redact_nested_match_and_regex_keys,  # pyright: ignore[reportUnknownVariableType]  # signature uses bare dict
)
from litellm.litellm_core_utils.usage.bedrock_guardrail_trace import (
    bedrock_guardrail_trace as bedrock_guardrail_trace,  # noqa: PLC0414  # re-export at the old path, tests import it
)
from litellm.litellm_core_utils.usage.bedrock_guardrail_trace import bedrock_guardrail_usage
from litellm.rust_bridge._native import NativeReceipt, NativeUnpriced
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import CostBreakdown, GuardrailStatus, StandardLoggingGuardrailInformation

if TYPE_CHECKING:
    from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import FeeDocument

BEDROCK_GUARDRAIL_PRICING_KEY: Final = "bedrock/guardrails"


class GuardrailCostEntry(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    guardrail_cost: float | None = None
    # ``bool | None`` because the TypedDict sanctions None; None means "not set"
    # and keeps the default billed behavior, so a None-carrying entry must not
    # fail union validation and silently zero a sibling entry's real cost.
    guardrail_cost_in_spend: bool | None = True


_GUARDRAIL_COST_ENTRY_ADAPTER: Final[TypeAdapter[GuardrailCostEntry]] = TypeAdapter(GuardrailCostEntry)


class GuardrailCostByUnitEntry(LiteLLMBaseModel):
    """The rollup-side view of a ``guardrail_information`` entry, validated apart from
    ``GuardrailCostEntry`` so a forged per-counter map can never zero the spend path."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    guardrail_cost_by_unit: Mapping[str, Annotated[float, Field(ge=0, allow_inf_nan=False)] | None] | None = None
    guardrail_cost_in_spend: bool | None = True


_GUARDRAIL_COST_BY_UNIT_ADAPTER: Final[TypeAdapter[GuardrailCostByUnitEntry]] = TypeAdapter(GuardrailCostByUnitEntry)


def billed_guardrail_cost_by_unit(raw: object) -> Mapping[str, float | None] | None:
    """Per-counter USD the daily rollup may record for one raw ``guardrail_information``
    entry; None when the entry is unpriced, report-only, or malformed, and None per
    counter the hook had no price for."""
    try:
        entry: Final = _GUARDRAIL_COST_BY_UNIT_ADAPTER.validate_python(raw)
    except ValidationError as e:
        verbose_logger.warning("Ignoring malformed guardrail_information entry for guardrail cost rollup: %s", e)
        return None
    return None if entry.guardrail_cost_in_spend is False else entry.guardrail_cost_by_unit


def _bedrock_guardrail_pricing_keys(aws_region_name: str | None) -> tuple[str, ...]:
    regional: Final = (f"bedrock/{aws_region_name}/guardrails",) if aws_region_name else ()
    return (*regional, BEDROCK_GUARDRAIL_PRICING_KEY)


def _priced(document: "FeeDocument") -> NativeReceipt | NativeUnpriced:
    from litellm.litellm_core_utils.llm_cost_calc.pricing_bridge import (  # noqa: PLC0415  # the bridge imports litellm.utils, which imports this module
        price_fee,
    )

    return price_fee(document)


def _bedrock_guardrail_outcome(
    usage_units: Mapping[str, int], aws_region_name: str | None
) -> NativeReceipt | NativeUnpriced:
    return _priced(
        {
            "kind": "guardrail",
            "keys": _bedrock_guardrail_pricing_keys(aws_region_name),
            "units": tuple({"counter": counter, "units": units} for counter, units in usage_units.items()),
        }
    )


def bedrock_guardrail_cost_by_unit(
    usage_units: Mapping[str, int], aws_region_name: str | None
) -> Mapping[str, float | None] | None:
    """USD per counter, keyed like ``usage_units``; None when no pricing entry exists,
    and None for a counter the entry has no price for, since only an explicit 0.0 means free."""
    outcome: Final = _bedrock_guardrail_outcome(usage_units, aws_region_name)
    return dict(outcome.parts) if isinstance(outcome, NativeReceipt) else None


def guardrail_cost_total(cost_by_unit: Mapping[str, float | None] | None) -> float:
    """The scalar the spend path bills: unknown-priced counters count as 0 here, the
    rollup keeps them unknown."""
    return sum(cost for cost in cost_by_unit.values() if cost is not None) if cost_by_unit is not None else 0.0


def bedrock_guardrail_cost(usage_units: Mapping[str, int], aws_region_name: str | None) -> float:
    outcome: Final = _bedrock_guardrail_outcome(usage_units, aws_region_name)
    return outcome.total if isinstance(outcome, NativeReceipt) else 0.0


def native_bedrock_guardrail_information(
    trace: Mapping[str, object], aws_region_name: str | None, guardrail_identifier: str | None
) -> StandardLoggingGuardrailInformation | None:
    """The guardrail entry for a trace Bedrock returned because the deployment's ``guardrailConfig`` ran with
    ``trace: enabled``: every assessment's ``invocationMetrics.usage`` summed and priced like the proxy-side Bedrock
    guardrail. None when the trace carries no usage, which is what ``trace: disabled`` leaves."""
    reported: Final = bedrock_guardrail_usage(trace)
    if reported is None:
        return None
    cost_by_unit: Final = bedrock_guardrail_cost_by_unit(reported.usage, aws_region_name)
    name: Final = guardrail_identifier or reported.first_guardrail_id
    status: Final[GuardrailStatus] = "guardrail_intervened" if reported.intervened else "success"
    priced: Final[StandardLoggingGuardrailInformation] = (
        {"guardrail_cost_by_unit": cost_by_unit} if cost_by_unit is not None else {}
    )
    return {
        "guardrail_name": name,
        "guardrail_id": name,
        "guardrail_provider": "bedrock",
        "guardrail_mode": GuardrailEventHooks.during_call,
        "guardrail_response": redact_nested_match_and_regex_keys(dict(reported.guardrail)),
        "guardrail_status": status,
        "guardrail_usage": reported.usage,
        "guardrail_cost": guardrail_cost_total(cost_by_unit),
        **priced,
    }


def guardrail_entries(raw: object) -> tuple[object, ...]:
    """The entries under ``standard_logging_guardrail_information``, which older writers stored as one dict."""
    single: Final = object_mapping(raw)
    return (single,) if single is not None else object_items(raw)


def with_guardrail_entry(
    metadata: dict[str, object],  # mutable-ok: the proxy's own request metadata, read later for headers and spend
    entry: StandardLoggingGuardrailInformation,
) -> None:
    """Append ``entry`` to the metadata's ``standard_logging_guardrail_information`` once, so the same trace recorded
    from the response headers path and again from the success handler bills a single time."""
    existing: Final = guardrail_entries(metadata.get("standard_logging_guardrail_information"))
    if entry in existing:
        return
    metadata["standard_logging_guardrail_information"] = [*existing, entry]  # rebind-ok: appending here is the job


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
    outcome: Final = _priced(
        {
            "kind": "prompt_shield",
            "units": tuple({"counter": counter, "units": units} for counter, units in usage_units.items()),
            "free_tier": cost_tier == "free",
            "per_thousand": price_per_1000_text_records,
        }
    )
    return outcome.total if isinstance(outcome, NativeReceipt) else None


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


def guardrail_share_of_spend(spend_log_metadata: Mapping[str, object], spend: float) -> float:
    """The part of a spend row's ``spend`` billed for its guardrails, priced the way success and failure rows add
    it to spend and capped at the row's spend, since an allowance discount or a repeat job poll can charge less."""
    return min(guardrail_information_cost(spend_log_metadata.get("guardrail_information")), max(spend, 0.0))


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
