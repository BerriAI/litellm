"""Accounting values use UTC instants, decimal strings on the wire and USD charges."""

from collections.abc import Mapping
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
    """Reasoning tokens are included in completion tokens; above-1hr writes are included in cache creation tokens."""

    REQUESTS = "requests"
    SUCCESSFUL_REQUESTS = "successful_requests"
    FAILED_REQUESTS = "failed_requests"
    PROMPT_TOKENS = "prompt_tokens"
    COMPLETION_TOKENS = "completion_tokens"
    REASONING_TOKENS = "reasoning_tokens"
    TOTAL_TOKENS = "total_tokens"
    CACHE_READ_INPUT_TOKENS = "cache_read_input_tokens"
    CACHE_CREATION_INPUT_TOKENS = "cache_creation_input_tokens"
    CACHE_CREATION_INPUT_TOKENS_ABOVE_1HR = "cache_creation_input_tokens_above_1hr"
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
    PROJECT = "project"
    MODEL_ACCESS_GROUP = "model_access_group"
    API_BASE = "api_base"
    SERVICE_TIER = "service_tier"
    TAG = "tag"


class Outcome(str, Enum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Granularity(str, Enum):
    DAY = "day"


class EventKind(str, Enum):
    INBOUND_USAGE = "inbound_usage"
    OUTBOUND_USAGE = "outbound_usage"
    COST = "cost"


class DeliveryState(str, Enum):
    PENDING = "pending"
    APPLIED = "applied"
    PARKED = "parked"


class DeliveryTarget(str, Enum):
    """One member per existing table the projector writes; each is applied and receipted independently."""

    KEY_SPEND = "LiteLLM_VerificationToken"
    USER_SPEND = "LiteLLM_UserTable"
    TEAM_SPEND = "LiteLLM_TeamTable"
    TEAM_MEMBER_SPEND = "LiteLLM_TeamMembership"
    ORGANIZATION_SPEND = "LiteLLM_OrganizationTable"
    ORGANIZATION_MEMBER_SPEND = "LiteLLM_OrganizationMembership"
    END_USER_SPEND = "LiteLLM_EndUserTable"
    AGENT_SPEND = "LiteLLM_AgentsTable"
    TAG_SPEND = "LiteLLM_TagTable"
    PROJECT_SPEND = "LiteLLM_ProjectTable"
    MODEL_ACCESS_GROUP_SPEND = "LiteLLM_ModelAccessGroupBudgetTable"
    BUDGET_SPEND = "LiteLLM_BudgetTable"
    BUDGET_WINDOW_SPEND = "LiteLLM_BudgetWindowSpend"
    DAILY_USER_SPEND = "LiteLLM_DailyUserSpend"
    DAILY_TEAM_SPEND = "LiteLLM_DailyTeamSpend"
    DAILY_ORGANIZATION_SPEND = "LiteLLM_DailyOrganizationSpend"
    DAILY_END_USER_SPEND = "LiteLLM_DailyEndUserSpend"
    DAILY_AGENT_SPEND = "LiteLLM_DailyAgentSpend"
    DAILY_TAG_SPEND = "LiteLLM_DailyTagSpend"
    DAILY_MODEL_USAGE = "LiteLLM_DailyModelUsage"
    DAILY_TOOL_SPEND = "LiteLLM_DailyToolSpend"
    AUTOROUTER_DAILY_SPEND = "LiteLLM_AutoRouterDailySpend"


_SUBSET_PAIRS: Final = (
    (MetricName.REASONING_TOKENS, MetricName.COMPLETION_TOKENS),
    (MetricName.CACHE_CREATION_INPUT_TOKENS_ABOVE_1HR, MetricName.CACHE_CREATION_INPUT_TOKENS),
)


def _check_subsets(values: Mapping[str, Decimal]) -> None:
    for part, whole in _SUBSET_PAIRS:
        if part.value in values and whole.value in values and values[part.value] > values[whole.value]:
            raise ValueError(f"{part.value} cannot exceed {whole.value}")


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


_REPEATABLE_DIMENSIONS: Final = frozenset({DimensionName.TAG, DimensionName.MODEL_ACCESS_GROUP})


def _dimensions(values: tuple[Dimension, ...]) -> tuple[Dimension, ...]:
    if len(frozenset(values)) != len(values):
        raise ValueError("Duplicate dimension values")
    single_names: Final = tuple(value.name for value in values if value.name not in _REPEATABLE_DIMENSIONS)
    if len(frozenset(single_names)) != len(single_names):
        raise ValueError("Only tag and model access group dimensions may have multiple values")
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
        if self.metrics is not None:
            _check_subsets({metric.name.value: metric.value for metric in self.metrics})
        return self


PricingUnit: TypeAlias = Literal["token", "second", "character", "image", "request"]


class Rate(BaseModel):
    model_config = _CONFIG

    measure: Identifier
    unit: PricingUnit
    usd_per_unit: Quantity


class PricedQuantity(BaseModel):
    """A pricing input in its billing unit. Only seconds may be fractional."""

    model_config = _CONFIG

    measure: Identifier
    unit: PricingUnit
    value: Quantity

    @model_validator(mode="after")
    def validate_discrete_quantity(self) -> Self:
        if self.unit != "second" and self.value != self.value.to_integral_value():
            raise ValueError("Discrete priced quantities require integral values")
        return self


class Multiplier(BaseModel):
    """A regional or tier factor applied on top of the priced key, e.g. data_residency=eu or vertex_location=us-east5."""

    model_config = _CONFIG

    name: Identifier
    value: Identifier
    factor: Annotated[Decimal, Field(strict=True, gt=0, allow_inf_nan=False)]


class NamedCost(BaseModel):
    """A free-form extra charge by name, e.g. azure_model_router_flat_cost."""

    model_config = _CONFIG

    name: Identifier
    amount: Quantity


class PricingSnapshot(BaseModel):
    """Retained inputs and result of authoritative compilation, reproducible after cost-map rates change.

    Request-scoped inputs, overrides and effective rates come from the usage row's versioned payload.
    Retries reuse the committed charge and snapshot.
    Component names match ``CostBreakdown``: reasoning_cost is part of output_cost, not an additional charge.
    Component sums and float tolerances belong to reconciliation, not this structural contract.
    """

    model_config = _CONFIG

    schema_version: Literal[1] = 1
    calculator_version: Identifier
    model: Identifier
    provider: Identifier | None
    pricing_source: Literal["cost_map", "deployment", "per_request", "upstream_header"]
    priced_model_key: Identifier
    multipliers: tuple[Multiplier, ...] = ()
    rates: tuple[Rate, ...] = ()
    usage: tuple[PricedQuantity, ...] = ()
    input_cost: Quantity | None = None
    output_cost: Quantity | None = None
    cache_read_cost: Quantity | None = None
    cache_creation_cost: Quantity | None = None
    reasoning_cost: Quantity | None = None
    tool_usage_cost: Quantity | None = None
    guardrail_cost: Quantity | None = None
    additional_costs: tuple[NamedCost, ...] = ()
    original_cost: Quantity | None = None
    discount_percent: Quantity | None = None
    discount_amount: Quantity | None = None
    margin_percent: Quantity | None = None
    margin_fixed_amount: Quantity | None = None
    margin_total_amount: Quantity | None = None
    upstream_reported_cost: Quantity | None = None
    total_cost: Quantity

    @model_validator(mode="after")
    def validate_snapshot(self) -> Self:
        rate_keys: Final = frozenset((rate.measure, rate.unit) for rate in self.rates)
        quantity_keys: Final = frozenset((quantity.measure, quantity.unit) for quantity in self.usage)
        if len(rate_keys) != len(self.rates):
            raise ValueError("Duplicate rate measures and units")
        if len(quantity_keys) != len(self.usage):
            raise ValueError("Duplicate priced quantity measures and units")
        if not rate_keys.issubset(quantity_keys):
            raise ValueError("Every rate requires one priced quantity with the same measure and unit")
        if len(frozenset(cost.name for cost in self.additional_costs)) != len(self.additional_costs):
            raise ValueError("Duplicate additional cost names")
        if len(frozenset(multiplier.name for multiplier in self.multipliers)) != len(self.multipliers):
            raise ValueError("Duplicate multiplier names")
        _check_subsets({quantity.measure: quantity.value for quantity in self.usage if quantity.unit == "token"})
        return self


class CostEvent(BaseModel):
    model_config = _CONFIG

    event_id: Identifier
    usage_event_id: Identifier
    occurred_at: Instant
    dimensions: Dimensions
    charges: Metrics
    snapshot: PricingSnapshot

    @model_validator(mode="after")
    def validate_charge(self) -> Self:
        if len(self.charges) != 1 or self.charges[0].name is not MetricName.SPEND:
            raise ValueError("A cost event requires one authoritative USD spend metric")
        if self.event_id == self.usage_event_id:
            raise ValueError("Cost and usage events require distinct identities")
        if self.snapshot.total_cost != self.charges[0].value:
            raise ValueError("Snapshot total must equal the authoritative spend")
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
