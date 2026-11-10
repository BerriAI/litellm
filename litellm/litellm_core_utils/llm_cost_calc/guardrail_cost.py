import math
from collections.abc import Iterator, Mapping, Sequence
from functools import partial
from itertools import chain
from typing import Annotated, Final

from pydantic import BaseModel, ConfigDict, Field, JsonValue, TypeAdapter, ValidationError
from typing_extensions import TypeIs  # noqa: TID251  # TypeIs reaches typing only on 3.13

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.core_helpers import (
    redact_nested_match_and_regex_keys,  # pyright: ignore[reportUnknownVariableType]  # signature uses bare dict
)
from litellm.types.guardrails import GuardrailEventHooks
from litellm.types.llms.base import LiteLLMBaseModel
from litellm.types.utils import CostBreakdown, GuardrailStatus, StandardLoggingGuardrailInformation

BEDROCK_GUARDRAIL_PRICING_KEY: Final = "bedrock/guardrails"
BEDROCK_TRACE_FIELDS: Final = ("trace", "amazon-bedrock-trace")
_INTERVENING_ACTIONS: Final = frozenset({"BLOCKED", "ANONYMIZED"})


class GuardrailPricing(LiteLLMBaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    guardrail_cost_per_unit: Mapping[str, float]


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


def _priced_units(units: int, price_per_unit: float | None) -> float | None:
    return None if price_per_unit is None else units * price_per_unit


def bedrock_guardrail_cost_by_unit(
    usage_units: Mapping[str, int], aws_region_name: str | None
) -> Mapping[str, float | None] | None:
    """USD per counter, keyed like ``usage_units``; None when no pricing entry exists,
    and None for a counter the entry has no price for, since only an explicit 0.0 means free."""
    pricing: Final = _bedrock_guardrail_pricing(aws_region_name)
    if pricing is None:
        return None
    return {
        counter: _priced_units(units, pricing.guardrail_cost_per_unit.get(counter))
        for counter, units in usage_units.items()
    }


def guardrail_cost_total(cost_by_unit: Mapping[str, float | None] | None) -> float:
    """The scalar the spend path bills: unknown-priced counters count as 0 here, the
    rollup keeps them unknown."""
    return sum(cost for cost in cost_by_unit.values() if cost is not None) if cost_by_unit is not None else 0.0


def bedrock_guardrail_cost(usage_units: Mapping[str, int], aws_region_name: str | None) -> float:
    return guardrail_cost_total(bedrock_guardrail_cost_by_unit(usage_units, aws_region_name))


_OBJECT_MAPPING: Final = TypeAdapter(Mapping[str, object])
_OBJECT_ITEMS: Final = TypeAdapter(tuple[object, ...])


def object_mapping(value: object) -> Mapping[str, object] | None:
    """``value`` as a str-keyed mapping, None for anything else. Vendor JSON arrives untyped, so this is the one place
    that turns it into something the rest of the module can read without guessing."""
    try:
        return _OBJECT_MAPPING.validate_python(value)
    except ValidationError:
        return None


def object_items(value: object) -> tuple[object, ...]:
    if isinstance(value, (str, bytes, Mapping)):
        return ()
    try:
        return _OBJECT_ITEMS.validate_python(value)
    except ValidationError:
        return ()


def is_str_keyed_dict(
    value: object,
) -> TypeIs[dict[str, object]]:  # guard-ok: JSON object keys are str  # mutable-ok: the caller edits this dict in place
    return isinstance(value, dict)


def _trace_field(response: object, field: str) -> Mapping[str, object] | None:
    source: Final = object_mapping(response)
    trace: Final = object_mapping(source.get(field) if source is not None else getattr(response, field, None))
    if trace is None or object_mapping(trace.get("guardrail")) is None:
        return None
    return trace


def bedrock_guardrail_trace(response: object) -> Mapping[str, object] | None:
    """The guardrail trace a Bedrock response carries: Converse puts it under ``trace``, Invoke under
    ``amazon-bedrock-trace``. Returned as is, or None when the response has no guardrail section."""
    traces: Final = map(partial(_trace_field, response), BEDROCK_TRACE_FIELDS)
    return next((trace for trace in traces if trace is not None), None)


class _Assessment(BaseModel):
    model_config = ConfigDict(frozen=True, extra="allow")
    invocation_metrics: Mapping[str, JsonValue] = Field(default_factory=dict, alias="invocationMetrics")

    @property
    def billable_usage(self) -> Mapping[str, int]:
        usage: Final = self.invocation_metrics.get("usage")
        if not isinstance(usage, dict):
            return {}
        return {
            counter: units
            for counter, units in usage.items()
            if counter.endswith("Units") and isinstance(units, int) and not isinstance(units, bool)
        }


class _GuardrailTrace(BaseModel):
    """Both trace shapes: Converse ``inputAssessment`` {id: a} and ``outputAssessments`` {id: [a, ...]}, Invoke
    ``input`` {id: a} and ``outputs`` [{id: a}, ...]."""

    model_config = ConfigDict(frozen=True, extra="allow")
    input_assessment: Mapping[str, _Assessment] = Field(default_factory=dict, alias="inputAssessment")
    input: Mapping[str, _Assessment] = Field(default_factory=dict)
    output_assessments: Mapping[str, tuple[_Assessment, ...]] = Field(default_factory=dict, alias="outputAssessments")
    outputs: tuple[Mapping[str, _Assessment], ...] = ()

    @property
    def assessments(self) -> tuple[tuple[str, _Assessment], ...]:
        listed: Final = (
            (gid, a) for gid, items in self.output_assessments.items() for a in items
        )  # comprehension-ok: id then its list
        return (
            *self.input_assessment.items(),
            *self.input.items(),
            *listed,
            *chain.from_iterable(by_id.items() for by_id in self.outputs),
        )


_JSON_OBJECT: Final = TypeAdapter(dict[str, JsonValue])


def _summed_usage(usages: Sequence[Mapping[str, int]]) -> Mapping[str, int]:
    counters: Final = tuple(dict.fromkeys(chain.from_iterable(usages)))
    return {counter: sum(usage.get(counter, 0) for usage in usages) for counter in counters}


def _actions(node: JsonValue) -> Iterator[str]:
    if isinstance(node, list):
        yield from chain.from_iterable(map(_actions, node))
        return
    if not isinstance(node, dict):
        return
    action: Final = node.get("action")
    if isinstance(action, str):
        yield action
    yield from chain.from_iterable(map(_actions, node.values()))


def native_bedrock_guardrail_information(
    trace: Mapping[str, object], aws_region_name: str | None, guardrail_identifier: str | None
) -> StandardLoggingGuardrailInformation | None:
    """The guardrail entry for a trace Bedrock returned because the deployment's ``guardrailConfig`` ran with
    ``trace: enabled``: every assessment's ``invocationMetrics.usage`` summed and priced like the proxy-side Bedrock
    guardrail. None when the trace carries no usage, which is what ``trace: disabled`` leaves."""
    try:
        guardrail: Final = _JSON_OBJECT.validate_python(trace.get("guardrail"))
        pairs: Final = _GuardrailTrace.model_validate(guardrail).assessments
    except ValidationError:
        return None
    usage: Final = _summed_usage(tuple(assessment.billable_usage for _, assessment in pairs))
    if not usage:
        return None
    cost_by_unit: Final = bedrock_guardrail_cost_by_unit(usage, aws_region_name)
    name: Final = guardrail_identifier or pairs[0][0]
    status: Final[GuardrailStatus] = (
        "guardrail_intervened" if any(action in _INTERVENING_ACTIONS for action in _actions(guardrail)) else "success"
    )
    priced: Final[StandardLoggingGuardrailInformation] = (
        {"guardrail_cost_by_unit": cost_by_unit} if cost_by_unit is not None else {}
    )
    return {
        "guardrail_name": name,
        "guardrail_id": name,
        "guardrail_provider": "bedrock",
        "guardrail_mode": GuardrailEventHooks.during_call,
        "guardrail_response": redact_nested_match_and_regex_keys(guardrail),
        "guardrail_status": status,
        "guardrail_usage": usage,
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
