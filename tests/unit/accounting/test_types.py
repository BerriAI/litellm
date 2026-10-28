from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Final

import pytest
from pydantic import ValidationError

from litellm.accounting import (
    AdmissionDecision,
    AggregateQuery,
    AggregateRow,
    CostEvent,
    Dimension,
    DimensionFilter,
    DimensionName,
    Granularity,
    InboundUsageEvent,
    Metric,
    MetricName,
    Outcome,
    OutboundUsageEvent,
    validate_aggregate_rows,
)
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
        ({"dimensions": (_TEAM, Dimension(name=DimensionName.TEAM, value="other"))}, "Only tag dimensions"),
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
        CostEvent(event_id="cost", usage_event_id="usage", occurred_at=_START, dimensions=(), charges=charges)


def test_cost_event_cannot_reuse_its_usage_identity():
    with pytest.raises(ValidationError, match="distinct identities"):
        CostEvent(event_id="usage", usage_event_id="usage", occurred_at=_START, dimensions=(), charges=(_SPEND,))


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
