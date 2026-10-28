"""Accounting values use UTC instants, decimal strings on the wire and USD charges."""

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Annotated, Final, Literal, TypeAlias

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict, Field, model_validator
from typing_extensions import Self

from litellm.types.utils import CallTypes

_CONFIG: Final = ConfigDict(frozen=True, extra="forbid", validate_default=True)
Identifier: TypeAlias = Annotated[str, Field(strict=True, min_length=1, pattern=r"\S")]
Quantity: TypeAlias = Annotated[Decimal, Field(strict=True, ge=0, allow_inf_nan=False)]


def _utc(value: datetime) -> datetime:
    return value.astimezone(timezone.utc)


Instant: TypeAlias = Annotated[AwareDatetime, AfterValidator(_utc)]


class MetricName(str, Enum):
    REQUESTS = "requests"
    SUCCESSFUL_REQUESTS = "successful_requests"
    FAILED_REQUESTS = "failed_requests"
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    TOTAL_TOKENS = "total_tokens"
    CACHE_READ_INPUT_TOKENS = "cache_read_input_tokens"
    CACHE_CREATION_INPUT_TOKENS = "cache_creation_input_tokens"
    SPEND = "spend"


class DimensionName(str, Enum):
    KEY = "key"
    USER = "user"
    TEAM = "team"
    ORGANIZATION = "organization"
    END_USER = "end_user"
    MODEL = "model"
    MODEL_GROUP = "model_group"
    DEPLOYMENT = "deployment"
    PROVIDER = "provider"
    TOOL = "tool"
    AGENT = "agent"
    TAG = "tag"


class Outcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Granularity(str, Enum):
    DAY = "day"


class Metric(BaseModel):
    model_config = _CONFIG

    name: MetricName
    value: Quantity
    unit: Literal["count", "USD"] = "count"

    @model_validator(mode="after")
    def validate_unit(self) -> Self:
        if (self.name is MetricName.SPEND) != (self.unit == "USD"):
            raise ValueError("SPEND requires USD; usage metrics require count")
        if self.unit == "count" and self.value != self.value.to_integral_value():
            raise ValueError("Count metrics require integral values")
        return self


class Dimension(BaseModel):
    model_config = _CONFIG

    name: DimensionName
    value: Identifier


def _dimensions(values: tuple[Dimension, ...]) -> tuple[Dimension, ...]:
    if len(frozenset(values)) != len(values):
        raise ValueError("Duplicate dimension values")
    single_names: Final = tuple(value.name for value in values if value.name is not DimensionName.TAG)
    if len(frozenset(single_names)) != len(single_names):
        raise ValueError("Only tag dimensions may have multiple values")
    return values


Dimensions: TypeAlias = Annotated[tuple[Dimension, ...], AfterValidator(_dimensions)]
_CALLER_METRICS: Final = frozenset((MetricName.REQUESTS, MetricName.SUCCESSFUL_REQUESTS, MetricName.FAILED_REQUESTS))


def _metrics(values: tuple[Metric, ...]) -> tuple[Metric, ...]:
    if len(frozenset(value.name for value in values)) != len(values):
        raise ValueError("Duplicate metric names")
    return values


Metrics: TypeAlias = Annotated[tuple[Metric, ...], AfterValidator(_metrics)]


def _interval(start: datetime, end: datetime, *, allow_empty: bool = False) -> None:
    if end < start or (end == start and not allow_empty):
        raise ValueError("End must follow start")


class InboundUsageEvent(BaseModel):
    model_config = _CONFIG

    event_id: Identifier
    request_id: Identifier
    call_type: CallTypes
    started_at: Instant
    ended_at: Instant
    outcome: Outcome
    dimensions: Dimensions
    metrics: Metrics

    @model_validator(mode="after")
    def validate_caller_activity(self) -> Self:
        _interval(self.started_at, self.ended_at, allow_empty=True)
        if any(metric.name not in _CALLER_METRICS for metric in self.metrics):
            raise ValueError("Inbound events own only caller request metrics")
        if not any(metric.name is MetricName.REQUESTS and metric.value == 1 for metric in self.metrics):
            raise ValueError("Inbound events count exactly one caller request")
        for metric in self.metrics:
            if metric.name is MetricName.SUCCESSFUL_REQUESTS and metric.value != int(self.outcome is Outcome.SUCCEEDED):
                raise ValueError("Successful request count must match outcome")
            if metric.name is MetricName.FAILED_REQUESTS and metric.value != int(self.outcome is not Outcome.SUCCEEDED):
                raise ValueError("Failed request count must match outcome")
        return self


class OutboundUsageEvent(BaseModel):
    model_config = _CONFIG

    event_id: Identifier
    attempt_id: Identifier
    request_id: Identifier | None
    retry_of_attempt_id: Identifier | None
    call_type: CallTypes
    started_at: Instant
    ended_at: Instant
    outcome: Outcome
    dimensions: Dimensions
    metrics: Metrics | None

    @model_validator(mode="after")
    def validate_attempt_usage(self) -> Self:
        _interval(self.started_at, self.ended_at, allow_empty=True)
        if self.retry_of_attempt_id == self.attempt_id:
            raise ValueError("An attempt cannot retry itself")
        if self.metrics is not None and any(
            metric.name in _CALLER_METRICS or metric.name is MetricName.SPEND for metric in self.metrics
        ):
            raise ValueError("Outbound events own only attempt usage metrics")
        return self


class CostEvent(BaseModel):
    model_config = _CONFIG

    event_id: Identifier
    usage_event_id: Identifier
    occurred_at: Instant
    dimensions: Dimensions
    charges: Metrics

    @model_validator(mode="after")
    def validate_charge(self) -> Self:
        if len(self.charges) != 1 or self.charges[0].name is not MetricName.SPEND:
            raise ValueError("A cost event requires one authoritative USD spend metric")
        if self.event_id == self.usage_event_id:
            raise ValueError("Cost and usage events require distinct identities")
        return self


class Unpriced(BaseModel):
    model_config = _CONFIG

    usage_event_id: Identifier
    reason: Identifier


class Measurement(BaseModel):
    model_config = _CONFIG

    source_event_id: Identifier
    occurred_at: Instant
    dimensions: Dimensions
    metric: Metric


class DimensionFilter(BaseModel):
    model_config = _CONFIG

    name: DimensionName
    values: Annotated[tuple[Identifier, ...], Field(min_length=1)]


class AggregateQuery(BaseModel):
    model_config = _CONFIG

    metric: MetricName
    filters: tuple[DimensionFilter, ...]
    group_by: tuple[DimensionName, ...]
    start: Instant
    end: Instant
    granularity: Granularity | None

    @model_validator(mode="after")
    def validate_query(self) -> Self:
        _interval(self.start, self.end)
        if len(frozenset(item.name for item in self.filters)) != len(self.filters):
            raise ValueError("Use one IN filter per dimension; filters combine with AND")
        if len(frozenset(self.group_by)) != len(self.group_by):
            raise ValueError("Duplicate group-by dimensions")
        if any(
            (instant.hour, instant.minute, instant.second, instant.microsecond) != (0, 0, 0, 0)
            for instant in (self.start, self.end)
        ):
            raise ValueError("Initial reporting queries require UTC day boundaries")
        return self


class AggregateRow(BaseModel):
    model_config = _CONFIG

    start: Instant
    end: Instant
    dimensions: Dimensions
    value: Quantity

    @model_validator(mode="after")
    def validate_bounds(self) -> Self:
        _interval(self.start, self.end)
        return self


def validate_aggregate_rows(rows: tuple[AggregateRow, ...]) -> tuple[AggregateRow, ...]:
    """Validate [start, end) buckets per group; overlapping tag groups are not additive."""
    groups: Final = frozenset(frozenset(row.dimensions) for row in rows)
    for group in groups:
        ordered = sorted((row for row in rows if frozenset(row.dimensions) == group), key=lambda row: row.start)
        if any(left.end > right.start for left, right in zip(ordered, ordered[1:])):
            raise ValueError("Aggregate buckets overlap within a dimension group")
    return rows


class AdmissionDecision(BaseModel):
    model_config = _CONFIG

    allowed: bool
    reason: Identifier | None = None
