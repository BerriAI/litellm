from datetime import datetime, timezone
from decimal import Decimal
from typing import Final

import pytest

from litellm.accounting import (
    CostEvent,
    Dimension,
    DimensionName,
    InboundUsageEvent,
    Measurement,
    Metric,
    MetricName,
    Outcome,
    OutboundUsageEvent,
    measurements_for,
)
from litellm.types.utils import CallTypes

_START: Final = datetime(2026, 1, 1, 23, 59, 59, tzinfo=timezone.utc)
_END: Final = datetime(2026, 1, 2, tzinfo=timezone.utc)
_DIMENSIONS: Final = (
    Dimension(name=DimensionName.TEAM, value="team"),
    Dimension(name=DimensionName.TAG, value="alpha"),
    Dimension(name=DimensionName.TAG, value="beta"),
)


@pytest.mark.parametrize(
    ("outcome", "outcome_metric"),
    [
        (Outcome.SUCCEEDED, MetricName.SUCCESSFUL_REQUESTS),
        (Outcome.FAILED, MetricName.FAILED_REQUESTS),
        (Outcome.CANCELLED, MetricName.FAILED_REQUESTS),
    ],
)
def test_caller_measurements_keep_both_request_and_outcome_counts(outcome, outcome_metric):
    event = InboundUsageEvent(
        event_id="caller-event",
        request_id="caller",
        call_type=CallTypes.acompletion,
        started_at=_START,
        ended_at=_END,
        outcome=outcome,
        dimensions=_DIMENSIONS,
        metrics=(Metric(name=MetricName.REQUESTS, value=Decimal(1)), Metric(name=outcome_metric, value=Decimal(1))),
    )
    assert measurements_for(event) == (
        Measurement(
            source_event_id="caller-event",
            occurred_at=_START,
            dimensions=_DIMENSIONS,
            metric=Metric(name=MetricName.REQUESTS, value=Decimal(1)),
        ),
        Measurement(
            source_event_id="caller-event",
            occurred_at=_START,
            dimensions=_DIMENSIONS,
            metric=Metric(name=outcome_metric, value=Decimal(1)),
        ),
    )


def test_usage_measurements_keep_every_metric_in_the_start_time_bucket():
    event = OutboundUsageEvent(
        event_id="usage-event",
        attempt_id="attempt",
        request_id="caller",
        retry_of_attempt_id=None,
        call_type=CallTypes.acompletion,
        started_at=_START,
        ended_at=_END,
        outcome=Outcome.SUCCEEDED,
        dimensions=_DIMENSIONS,
        metrics=(
            Metric(name=MetricName.PROMPT_TOKENS, value=Decimal(10)),
            Metric(name=MetricName.COMPLETION_TOKENS, value=Decimal(2)),
            Metric(name=MetricName.TOTAL_TOKENS, value=Decimal(12)),
            Metric(name=MetricName.CACHE_READ_INPUT_TOKENS, value=Decimal(3)),
            Metric(name=MetricName.CACHE_CREATION_INPUT_TOKENS, value=Decimal(1)),
        ),
    )
    assert measurements_for(event) == (
        Measurement(
            source_event_id="usage-event",
            occurred_at=_START,
            dimensions=_DIMENSIONS,
            metric=Metric(name=MetricName.PROMPT_TOKENS, value=Decimal(10)),
        ),
        Measurement(
            source_event_id="usage-event",
            occurred_at=_START,
            dimensions=_DIMENSIONS,
            metric=Metric(name=MetricName.COMPLETION_TOKENS, value=Decimal(2)),
        ),
        Measurement(
            source_event_id="usage-event",
            occurred_at=_START,
            dimensions=_DIMENSIONS,
            metric=Metric(name=MetricName.TOTAL_TOKENS, value=Decimal(12)),
        ),
        Measurement(
            source_event_id="usage-event",
            occurred_at=_START,
            dimensions=_DIMENSIONS,
            metric=Metric(name=MetricName.CACHE_READ_INPUT_TOKENS, value=Decimal(3)),
        ),
        Measurement(
            source_event_id="usage-event",
            occurred_at=_START,
            dimensions=_DIMENSIONS,
            metric=Metric(name=MetricName.CACHE_CREATION_INPUT_TOKENS, value=Decimal(1)),
        ),
    )


@pytest.mark.parametrize("metrics", [None, ()])
def test_unknown_or_empty_usage_does_not_fabricate_zero_measurements(metrics):
    event = OutboundUsageEvent(
        event_id="usage-event",
        attempt_id="attempt",
        request_id=None,
        retry_of_attempt_id=None,
        call_type=CallTypes.acompletion,
        started_at=_START,
        ended_at=_END,
        outcome=Outcome.SUCCEEDED,
        dimensions=(),
        metrics=metrics,
    )
    assert measurements_for(event) == ()


@pytest.mark.parametrize("amount", [Decimal(0), Decimal("0.1234567890123456789")])
def test_cost_measurements_preserve_authoritative_amount_identity_and_occurrence(amount):
    event = CostEvent(
        event_id="cost-event",
        usage_event_id="usage-event",
        occurred_at=_END,
        dimensions=_DIMENSIONS,
        charges=(Metric(name=MetricName.SPEND, value=amount, unit="USD"),),
    )
    assert measurements_for(event) == (
        Measurement(
            source_event_id="cost-event",
            occurred_at=_END,
            dimensions=_DIMENSIONS,
            metric=Metric(name=MetricName.SPEND, value=amount, unit="USD"),
        ),
    )


def test_billed_retry_and_success_contribute_one_caller_request_and_both_charges():
    caller = InboundUsageEvent(
        event_id="caller-event",
        request_id="caller",
        call_type=CallTypes.acompletion,
        started_at=_START,
        ended_at=_END,
        outcome=Outcome.SUCCEEDED,
        dimensions=_DIMENSIONS,
        metrics=(Metric(name=MetricName.REQUESTS, value=Decimal(1)),),
    )
    retry = OutboundUsageEvent(
        event_id="retry-usage",
        attempt_id="retry",
        request_id="caller",
        retry_of_attempt_id=None,
        call_type=CallTypes.acompletion,
        started_at=_START,
        ended_at=_END,
        outcome=Outcome.FAILED,
        dimensions=_DIMENSIONS,
        metrics=(Metric(name=MetricName.TOTAL_TOKENS, value=Decimal(4)),),
    )
    success = OutboundUsageEvent(
        event_id="success-usage",
        attempt_id="success",
        request_id="caller",
        retry_of_attempt_id="retry",
        call_type=CallTypes.acompletion,
        started_at=_START,
        ended_at=_END,
        outcome=Outcome.SUCCEEDED,
        dimensions=_DIMENSIONS,
        metrics=(Metric(name=MetricName.TOTAL_TOKENS, value=Decimal(12)),),
    )
    retry_cost = CostEvent(
        event_id="retry-cost",
        usage_event_id="retry-usage",
        occurred_at=_START,
        dimensions=_DIMENSIONS,
        charges=(Metric(name=MetricName.SPEND, value=Decimal("0.025"), unit="USD"),),
    )
    success_cost = CostEvent(
        event_id="success-cost",
        usage_event_id="success-usage",
        occurred_at=_START,
        dimensions=_DIMENSIONS,
        charges=(Metric(name=MetricName.SPEND, value=Decimal("0.125"), unit="USD"),),
    )
    measurements = (
        *measurements_for(caller),
        *measurements_for(retry),
        *measurements_for(success),
        *measurements_for(retry_cost),
        *measurements_for(success_cost),
    )
    assert sum(m.metric.value for m in measurements if m.metric.name is MetricName.REQUESTS) == Decimal(1)
    assert sum(m.metric.value for m in measurements if m.metric.name is MetricName.TOTAL_TOKENS) == Decimal(16)
    assert sum(m.metric.value for m in measurements if m.metric.name is MetricName.SPEND) == Decimal("0.150")
    assert {m.source_event_id for m in measurements} == {
        "caller-event",
        "retry-usage",
        "success-usage",
        "retry-cost",
        "success-cost",
    }
