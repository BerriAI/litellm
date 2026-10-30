import re
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from enum import Enum
from pathlib import Path
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.accounting import (
    AdmissionDecision,
    AggregateQuery,
    AggregateRow,
    CostEvent,
    DeliveryState,
    DeliveryTarget,
    Dimension,
    DimensionFilter,
    DimensionName,
    EventKind,
    Granularity,
    InboundUsageEvent,
    Metric,
    MetricName,
    Multiplier,
    Outcome,
    OutboundUsageEvent,
    PricedQuantity,
    PricingSnapshot,
    Rate,
    validate_aggregate_rows,
)
from litellm.accounting.types import PricingUnit
from litellm.types.utils import CallTypes

_START: Final = datetime(2026, 1, 1, tzinfo=timezone.utc)
_END: Final = datetime(2026, 1, 2, tzinfo=timezone.utc)
_REQUEST: Final = Metric(name=MetricName.REQUESTS, value=Decimal(1))
_TOKEN: Final = Metric(name=MetricName.PROMPT_TOKENS, value=Decimal(10))
_SPEND: Final = Metric(name=MetricName.SPEND, value=Decimal("0.1234567890123456789"), unit="USD")
_TEAM: Final = Dimension(name=DimensionName.TEAM, value="team")


def _inbound(metrics=(_REQUEST,), outcome=Outcome.SUCCEEDED):
    return InboundUsageEvent(
        event_id="inbound:request",
        request_id="request",
        call_type=CallTypes.acompletion,
        started_at=_START,
        ended_at=_START,
        outcome=outcome,
        dimensions=(_TEAM,),
        metrics=metrics,
    )


def _outbound(metrics=(_TOKEN,), **changes):
    return OutboundUsageEvent.model_validate(
        {
            "event_id": "outbound:attempt",
            "attempt_id": "attempt",
            "request_id": "request",
            "retry_of_attempt_id": None,
            "call_type": CallTypes.acompletion,
            "started_at": _START,
            "ended_at": _END,
            "outcome": Outcome.SUCCEEDED,
            "dimensions": (_TEAM,),
            "metrics": metrics,
            **changes,
        }
    )


def _query(**changes):
    return AggregateQuery.model_validate(
        {
            "metric": MetricName.SPEND,
            "filters": (DimensionFilter(name=DimensionName.TEAM, values=("one", "two")),),
            "group_by": (DimensionName.MODEL,),
            "start": _START,
            "end": _END,
            "granularity": Granularity.DAY,
            **changes,
        }
    )


def test_wire_values_preserve_decimal_precision_and_normalize_aware_times():
    offset = timezone(timedelta(hours=-8))
    event = CostEvent(
        event_id="cost:attempt",
        usage_event_id="outbound:attempt",
        occurred_at=datetime(2025, 12, 31, 16, tzinfo=offset),
        dimensions=(_TEAM,),
        charges=(_SPEND,),
        snapshot=_snapshot(),
    )
    wire = event.model_dump(mode="json")
    assert wire["occurred_at"] == "2026-01-01T00:00:00Z"
    assert wire["charges"] == [{"name": "spend", "value": "0.1234567890123456789", "unit": "USD"}]
    assert event.occurred_at == _START
    assert CostEvent.model_validate_json(event.model_dump_json()).charges[0].value == _SPEND.value


@pytest.mark.parametrize(
    ("name", "value", "unit", "message"),
    [
        (MetricName.SPEND, Decimal(1), "count", "SPEND requires USD"),
        (MetricName.REQUESTS, Decimal(1), "USD", "SPEND requires USD"),
        (MetricName.PROMPT_TOKENS, Decimal("1.5"), "count", "integral"),
        (MetricName.SPEND, Decimal("NaN"), "USD", "finite"),
        (MetricName.SPEND, Decimal("Infinity"), "USD", "finite"),
        (MetricName.SPEND, Decimal(-1), "USD", "greater than or equal"),
        (MetricName.SPEND, 0.1, "USD", "instance of Decimal"),
    ],
)
def test_metrics_reject_wrong_units_nonfinite_values_and_binary_float_inputs(name, value, unit, message):
    with pytest.raises(ValidationError, match=message):
        Metric(name=name, value=value, unit=unit)


@pytest.mark.parametrize("metrics", [(_TOKEN,), (_SPEND,), ()])
def test_inbound_activity_cannot_own_usage_or_cost_or_omit_request_count(metrics):
    with pytest.raises(ValidationError, match="Inbound events"):
        _inbound(metrics)


@pytest.mark.parametrize(
    ("outcome", "metric"),
    [
        (Outcome.FAILED, Metric(name=MetricName.SUCCESSFUL_REQUESTS, value=Decimal(1))),
        (Outcome.SUCCEEDED, Metric(name=MetricName.FAILED_REQUESTS, value=Decimal(1))),
    ],
)
def test_inbound_outcome_counts_cannot_contradict_outcome(outcome, metric):
    with pytest.raises(ValidationError, match="count must match outcome"):
        _inbound((_REQUEST, metric), outcome)


@pytest.mark.parametrize("metrics", [(_REQUEST,), (_SPEND,)])
def test_outbound_attempts_cannot_own_caller_counts_or_charges(metrics):
    with pytest.raises(ValidationError, match="Outbound events own only"):
        _outbound(metrics)


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"started_at": datetime(2026, 1, 1)}, "timezone"),
        ({"ended_at": _START - timedelta(seconds=1)}, "End must follow start"),
        ({"retry_of_attempt_id": "attempt"}, "cannot retry itself"),
        ({"dimensions": (_TEAM, _TEAM)}, "Duplicate dimension values"),
        ({"dimensions": (_TEAM, Dimension(name=DimensionName.TEAM, value="other"))}, "Only tag and model access group"),
        (
            {
                "metrics": (
                    Metric(name=MetricName.COMPLETION_TOKENS, value=Decimal(5)),
                    Metric(name=MetricName.REASONING_TOKENS, value=Decimal(6)),
                )
            },
            "reasoning_tokens cannot exceed completion_tokens",
        ),
        (
            {
                "metrics": (
                    Metric(name=MetricName.CACHE_CREATION_INPUT_TOKENS, value=Decimal(5)),
                    Metric(name=MetricName.CACHE_CREATION_INPUT_TOKENS_ABOVE_1HR, value=Decimal(6)),
                )
            },
            "cache_creation_input_tokens_above_1hr cannot exceed cache_creation_input_tokens",
        ),
        ({"event_id": " "}, "pattern"),
        ({"metrics": (_TOKEN, _TOKEN)}, "Duplicate metric names"),
    ],
)
def test_outbound_events_reject_invalid_identity_time_and_duplicate_attribution(changes, message):
    with pytest.raises(ValidationError, match=message):
        _outbound(**changes)


def test_unknown_and_known_zero_usage_are_distinct_and_values_are_immutable():
    unknown = _outbound(None)
    zero = _outbound((Metric(name=MetricName.PROMPT_TOKENS, value=Decimal(0)),))
    assert unknown.metrics is None
    assert zero.metrics is not None and zero.metrics[0].value == 0
    assert _inbound().started_at == _inbound().ended_at
    with pytest.raises(ValidationError, match="frozen"):
        zero.attempt_id = "different"
    with pytest.raises(ValidationError, match="frozen"):
        zero.metrics[0].value = Decimal(10)


@pytest.mark.parametrize("charges", [(), (_TOKEN,), (_SPEND, _SPEND)])
def test_cost_events_require_one_authoritative_spend(charges):
    with pytest.raises(ValidationError, match=r"authoritative USD spend|Duplicate metric"):
        CostEvent(
            event_id="cost",
            usage_event_id="usage",
            occurred_at=_START,
            dimensions=(),
            charges=charges,
            snapshot=_snapshot(),
        )


def test_cost_event_cannot_reuse_its_usage_identity():
    with pytest.raises(ValidationError, match="distinct identities"):
        CostEvent(
            event_id="usage",
            usage_event_id="usage",
            occurred_at=_START,
            dimensions=(),
            charges=(_SPEND,),
            snapshot=_snapshot(),
        )


@pytest.mark.parametrize("granularity", [Granularity.DAY, None])
def test_daily_and_whole_interval_queries_preserve_filter_and_group_semantics(granularity):
    query = _query(granularity=granularity)
    assert query.granularity is granularity
    assert query.filters[0].values == ("one", "two")
    assert query.group_by == (DimensionName.MODEL,)
    assert query.start == _START and query.end == _END


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"end": _START}, "End must follow start"),
        ({"start": _START + timedelta(hours=1)}, "UTC day boundaries"),
        ({"end": _END + timedelta(microseconds=1)}, "UTC day boundaries"),
        ({"granularity": "hour"}, "day"),
        ({"group_by": (DimensionName.TEAM, DimensionName.TEAM)}, "Duplicate group-by"),
        ({"filters": (DimensionFilter(name=DimensionName.TEAM, values=("one",)),) * 2}, "one IN filter"),
        ({"filters": ({"name": "team", "values": ()},)}, "at least 1"),
    ],
)
def test_queries_reject_unsupported_or_ambiguous_ranges_and_filters(changes, message):
    with pytest.raises(ValidationError, match=message):
        _query(**changes)


def test_rows_allow_adjacent_buckets_and_overlapping_distinct_groups():
    first = AggregateRow(start=_START, end=_END, dimensions=(_TEAM,), value=Decimal(1))
    second = AggregateRow(start=_END, end=_END + timedelta(days=1), dimensions=(_TEAM,), value=Decimal(2))
    other = AggregateRow(start=_START, end=_END, dimensions=(), value=Decimal(3))
    rows = (second, other, first)
    assert validate_aggregate_rows(rows) is rows
    assert validate_aggregate_rows(()) == ()


def test_rows_reject_overlap_within_a_group_regardless_of_dimension_order():
    model = Dimension(name=DimensionName.MODEL, value="model")
    first = AggregateRow(start=_START, end=_END, dimensions=(_TEAM, model), value=Decimal(1))
    overlapping = AggregateRow(start=_START, end=_END, dimensions=(model, _TEAM), value=Decimal(2))
    with pytest.raises(ValueError, match="overlap within a dimension group"):
        validate_aggregate_rows((first, overlapping))


def test_rows_reject_empty_intervals():
    with pytest.raises(ValidationError, match="End must follow start"):
        AggregateRow(start=_START, end=_START, dimensions=(), value=Decimal(0))


def test_admission_decision_preserves_a_denial_reason():
    decision = AdmissionDecision(allowed=False, reason="Team budget exhausted")
    assert not decision.allowed
    assert decision.reason == "Team budget exhausted"


def _snapshot(**changes: object) -> PricingSnapshot:
    return PricingSnapshot.model_validate(
        {
            "calculator_version": "1.80.0",
            "model": "gpt-4o",
            "provider": "openai",
            "pricing_source": "cost_map",
            "priced_model_key": "openai/gpt-4o",
            "multipliers": (Multiplier(name="data_residency", value="eu", factor=Decimal("1.1")),),
            "rates": (Rate(measure="input", unit="token", usd_per_unit=Decimal("0.0000025")),),
            "usage": (PricedQuantity(measure="input", unit="token", value=_TOKEN.value),),
            "input_cost": Decimal("0.000025"),
            "total_cost": _SPEND.value,
            **changes,
        }
    )


def test_cost_event_carries_a_snapshot_whose_total_matches_the_spend():
    event = CostEvent(
        event_id="cost:attempt",
        usage_event_id="outbound:attempt",
        occurred_at=_START,
        dimensions=(_TEAM,),
        charges=(_SPEND,),
        snapshot=_snapshot(),
    )
    assert event.snapshot is not None
    assert event.snapshot.total_cost == _SPEND.value
    assert event.snapshot.multipliers[0].factor == Decimal("1.1")
    with pytest.raises(ValidationError, match="Snapshot total must equal the authoritative spend"):
        CostEvent(
            event_id="cost:attempt",
            usage_event_id="outbound:attempt",
            occurred_at=_START,
            dimensions=(_TEAM,),
            charges=(_SPEND,),
            snapshot=_snapshot(total_cost=Decimal("0.5")),
        )


@pytest.mark.parametrize(
    ("changes", "message"),
    (
        (
            {
                "rates": (
                    Rate(measure="input", unit="token", usd_per_unit=Decimal(1)),
                    Rate(measure="input", unit="token", usd_per_unit=Decimal(2)),
                )
            },
            "Duplicate rate measures and units",
        ),
        ({"usage": (_SPEND,)}, "PricedQuantity"),
        ({"usage": (_REQUEST,)}, "PricedQuantity"),
        ({"pricing_source": "guess"}, "pricing_source"),
        ({"multipliers": ({"name": "x", "value": "y", "factor": Decimal(0)},)}, "factor"),
        ({"schema_version": 2}, "schema_version"),
        (
            {"additional_costs": ({"name": "fee", "amount": Decimal(1)}, {"name": "fee", "amount": Decimal(2)})},
            "Duplicate additional cost names",
        ),
    ),
)
def test_snapshots_reject_ambiguous_rates_wrong_usage_and_unknown_sources(changes: Mapping[str, object], message: str):
    with pytest.raises(ValidationError, match=message):
        _snapshot(**changes)


@pytest.mark.parametrize(
    ("unit", "value"),
    [("token", "10"), ("second", "1.25"), ("character", "25"), ("image", "2"), ("request", "1")],
)
def test_snapshot_preserves_matching_rates_and_quantities_for_each_billing_unit(unit: PricingUnit, value: str) -> None:
    quantity: Final = PricedQuantity(measure="input", unit=unit, value=Decimal(value))
    rate: Final = Rate(measure="input", unit=unit, usd_per_unit=Decimal("0.003"))
    amount: Final = quantity.value * rate.usd_per_unit
    snapshot: Final = _snapshot(rates=(rate,), usage=(quantity,), total_cost=amount)
    restored: Final = PricingSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored == snapshot
    assert restored.usage[0].value * restored.rates[0].usd_per_unit == amount


@pytest.mark.parametrize(
    ("usage", "message"),
    [
        ((), "Every rate requires"),
        (({"measure": "output", "unit": "token", "value": Decimal(10)},), "Every rate requires"),
        (({"measure": "input", "unit": "second", "value": Decimal(10)},), "Every rate requires"),
        (
            ({"measure": "input", "unit": "token", "value": Decimal(10)},) * 2,
            "Duplicate priced quantity",
        ),
    ],
)
def test_snapshot_rejects_missing_mismatched_or_duplicate_priced_quantities(
    usage: tuple[Mapping[str, object], ...], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        _snapshot(usage=usage)


@pytest.mark.parametrize("unit", ["token", "character", "image", "request"])
def test_priced_counts_remain_integral(unit: PricingUnit) -> None:
    with pytest.raises(ValidationError, match="integral"):
        PricedQuantity(measure="input", unit=unit, value=Decimal("1.25"))


@pytest.mark.parametrize(
    ("value", "message"),
    [
        (Decimal(-1), "greater than or equal"),
        (Decimal("NaN"), "finite"),
        (Decimal("Infinity"), "finite"),
        (1.25, "instance of Decimal"),
    ],
)
def test_pricing_values_reject_invalid_amounts(value: Decimal | float, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        PricedQuantity.model_validate({"measure": "input", "unit": "second", "value": value})
    with pytest.raises(ValidationError, match=message):
        Rate.model_validate({"measure": "input", "unit": "second", "usd_per_unit": value})


@pytest.mark.parametrize("snapshot_fields", [{}, {"snapshot": None}])
def test_cost_event_cannot_omit_its_derivation(snapshot_fields: Mapping[str, object]) -> None:
    with pytest.raises(ValidationError, match="snapshot"):
        CostEvent.model_validate(
            {
                "event_id": "cost",
                "usage_event_id": "usage",
                "occurred_at": _START,
                "dimensions": (),
                "charges": (_SPEND,),
                **snapshot_fields,
            }
        )


def test_upstream_charge_snapshot_preserves_reported_cost_without_inventing_rates() -> None:
    snapshot: Final = _snapshot(
        pricing_source="upstream_header", rates=(), usage=(), upstream_reported_cost=_SPEND.value
    )
    restored: Final = PricingSnapshot.model_validate_json(snapshot.model_dump_json())
    assert restored.upstream_reported_cost == restored.total_cost == _SPEND.value
    assert restored.rates == restored.usage == ()


def test_subset_token_counts_round_trip_without_changing_totals() -> None:
    metrics: Final = (
        Metric(name=MetricName.COMPLETION_TOKENS, value=Decimal(10)),
        Metric(name=MetricName.REASONING_TOKENS, value=Decimal(3)),
        Metric(name=MetricName.CACHE_CREATION_INPUT_TOKENS, value=Decimal(8)),
        Metric(name=MetricName.CACHE_CREATION_INPUT_TOKENS_ABOVE_1HR, value=Decimal(2)),
    )
    event: Final = _outbound(metrics=metrics)
    assert OutboundUsageEvent.model_validate_json(event.model_dump_json()).metrics == metrics


@pytest.mark.parametrize(
    "schema_path",
    [
        "schema.prisma",
        "litellm/proxy/schema.prisma",
        "litellm-proxy-extras/litellm_proxy_extras/schema.prisma",
    ],
)
@pytest.mark.parametrize(
    ("enum_name", "python_enum"),
    [
        ("AccountingEventKind", EventKind),
        ("AccountingDeliveryState", DeliveryState),
        ("AccountingDeliveryTarget", DeliveryTarget),
    ],
)
def test_prisma_accepts_exactly_the_public_accounting_enum_values(
    schema_path: str, enum_name: str, python_enum: type[Enum]
) -> None:
    schema: Final = (Path(__file__).resolve().parents[3] / schema_path).read_text()
    declaration: Final = re.search(r"enum " + enum_name + r"\s*\{([^}]+)\}", schema)
    assert declaration is not None, f"Missing persisted enum {enum_name}"
    assert frozenset(declaration[1].split()) == frozenset(member.value for member in python_enum)


def test_one_event_carries_every_matched_model_access_group():
    groups = (
        Dimension(name=DimensionName.MODEL_ACCESS_GROUP, value="gpt-group"),
        Dimension(name=DimensionName.MODEL_ACCESS_GROUP, value="claude-group"),
    )
    assert _outbound(dimensions=(_TEAM, *groups)).dimensions == (_TEAM, *groups)
    with pytest.raises(ValidationError, match="Duplicate dimension values"):
        _outbound(dimensions=(groups[0], groups[0]))


def test_snapshots_reject_duplicate_multipliers_and_contradictory_token_counts():
    multiplier = Multiplier(name="data_residency", value="eu", factor=Decimal("1.1"))
    with pytest.raises(ValidationError, match="Duplicate multiplier names"):
        _snapshot(multipliers=(multiplier, multiplier))
    with pytest.raises(ValidationError, match="reasoning_tokens cannot exceed completion_tokens"):
        _snapshot(
            usage=(
                PricedQuantity(measure="input", unit="token", value=_TOKEN.value),
                PricedQuantity(measure="completion_tokens", unit="token", value=Decimal(5)),
                PricedQuantity(measure="reasoning_tokens", unit="token", value=Decimal(6)),
            )
        )


def test_agent_live_spend_is_a_delivery_target():
    assert DeliveryTarget.AGENT_SPEND.value == "LiteLLM_AgentsTable"
