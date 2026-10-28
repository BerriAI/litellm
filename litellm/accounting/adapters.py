"""Pure legacy adapters. Callers supply facts that a spend-log row cannot establish."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from typing import TYPE_CHECKING, Annotated, Final, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, Json, JsonValue

from litellm.accounting.types import (
    CostEvent,
    Dimension,
    DimensionName,
    Identifier,
    InboundUsageEvent,
    Instant,
    Measurement,
    Metric,
    MetricName,
    OutboundUsageEvent,
    Outcome,
    Unpriced,
)
from litellm.constants import (
    BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX,
    BATCH_COST_REQUEST_ID_SUFFIX,
    BILLED_RETRY_ATTEMPT_METADATA_KEY,
    INTERNAL_CALL_ORIGIN_METADATA_KEY,
)
from litellm.litellm_core_utils.llm_cost_calc.cache_tokens import (
    extract_cache_creation_tokens,
    extract_cache_read_tokens,
)
from litellm.types.utils import CallTypes

if TYPE_CHECKING:
    from litellm.proxy._types import SpendLogsPayload


class SpendLogContext(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    usage_known: bool
    price_known: bool
    request_id: Identifier | None = None
    retry_of_attempt_id: Identifier | None = None
    outcome: Outcome | None = None


@dataclass(frozen=True, slots=True)
class SpendLogEvents:
    inbound: InboundUsageEvent | None
    outbound: OutboundUsageEvent
    cost: CostEvent | Unpriced


@dataclass(frozen=True, slots=True)
class SpendLogAdapterError:
    reason: str


class _Metadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    billed_retry_attempt: bool = Field(default=False, alias=BILLED_RETRY_ATTEMPT_METADATA_KEY)
    internal_call_origin: str | None = Field(default=None, alias=INTERNAL_CALL_ORIGIN_METADATA_KEY)
    usage_object: Mapping[str, JsonValue] | None = None


_Count: TypeAlias = Annotated[int, Field(strict=True, ge=0)]


class _SpendLog(BaseModel):
    model_config = ConfigDict(frozen=True)

    request_id: Identifier
    call_type: CallTypes
    started_at: Instant = Field(alias="startTime")
    ended_at: Instant = Field(alias="endTime")
    status: Literal["success", "failure"]
    metadata: Json[_Metadata]
    request_tags: Json[tuple[Identifier, ...]]
    spend: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None
    prompt_tokens: _Count
    completion_tokens: _Count
    total_tokens: _Count
    api_key: str
    user: str
    team_id: str | None
    organization_id: str | None
    end_user: str | None
    model: str
    model_group: str | None
    model_id: str | None
    custom_llm_provider: str | None
    mcp_namespaced_tool_name: str | None
    agent_id: str | None


def _dimensions(row: _SpendLog) -> tuple[Dimension, ...]:
    values: Final = (
        (DimensionName.KEY, row.api_key),
        (DimensionName.USER, row.user),
        (DimensionName.TEAM, row.team_id),
        (DimensionName.ORGANIZATION, row.organization_id),
        (DimensionName.END_USER, row.end_user),
        (DimensionName.MODEL, row.model),
        (DimensionName.MODEL_GROUP, row.model_group),
        (DimensionName.DEPLOYMENT, row.model_id),
        (DimensionName.PROVIDER, row.custom_llm_provider),
        (DimensionName.TOOL, row.mcp_namespaced_tool_name),
        (DimensionName.AGENT, row.agent_id),
    )
    dimensions: Final = tuple(Dimension(name=name, value=value) for name, value in values if value)
    tags: Final = tuple(Dimension(name=DimensionName.TAG, value=value) for value in sorted(frozenset(row.request_tags)))
    return dimensions + tags


def _usage(row: _SpendLog) -> tuple[Metric, ...]:
    values: Final = (
        (MetricName.PROMPT_TOKENS, row.prompt_tokens),
        (MetricName.COMPLETION_TOKENS, row.completion_tokens),
        (MetricName.TOTAL_TOKENS, row.total_tokens),
        (MetricName.CACHE_READ_INPUT_TOKENS, extract_cache_read_tokens(row.metadata.usage_object)),
        (MetricName.CACHE_CREATION_INPUT_TOKENS, extract_cache_creation_tokens(row.metadata.usage_object)),
    )
    return tuple(Metric(name=name, value=Decimal(value)) for name, value in values)


def _inbound(row: _SpendLog, context: SpendLogContext, outbound: OutboundUsageEvent) -> InboundUsageEvent | None:
    if (
        context.request_id is None
        or row.metadata.billed_retry_attempt
        or row.metadata.internal_call_origin
        or row.request_id.endswith((BATCH_COST_REQUEST_ID_SUFFIX, BACKGROUND_RESPONSE_COST_REQUEST_ID_SUFFIX))
    ):
        return None
    outcome_metric: Final = (
        MetricName.SUCCESSFUL_REQUESTS if outbound.outcome is Outcome.SUCCEEDED else MetricName.FAILED_REQUESTS
    )
    return InboundUsageEvent(
        event_id=f"inbound:{context.request_id}",
        request_id=context.request_id,
        call_type=outbound.call_type,
        started_at=outbound.started_at,
        ended_at=outbound.ended_at,
        outcome=outbound.outcome,
        dimensions=outbound.dimensions,
        metrics=(Metric(name=MetricName.REQUESTS, value=Decimal(1)), Metric(name=outcome_metric, value=Decimal(1))),
    )


def _cost(row: _SpendLog, context: SpendLogContext, outbound: OutboundUsageEvent) -> CostEvent | Unpriced:
    if not context.price_known or row.spend is None:
        return Unpriced(usage_event_id=outbound.event_id, reason="No authoritative price supplied")
    return CostEvent(
        event_id=f"cost:{row.request_id}",
        usage_event_id=outbound.event_id,
        occurred_at=outbound.started_at,
        dimensions=outbound.dimensions,
        charges=(Metric(name=MetricName.SPEND, value=Decimal(str(row.spend)), unit="USD"),),
    )


def adapt_spend_log(payload: SpendLogsPayload, *, context: SpendLogContext) -> SpendLogEvents | SpendLogAdapterError:
    """Preserve row IDs as attempt IDs, namespacing event IDs by kind.

    Only an explicitly supplied logical request ID can produce inbound activity.
    Retry counts do not establish parent links. Occurrence uses startTime to retain
    the existing daily bucket. Bodies and unrelated metadata are never copied.
    """
    try:
        row: Final = _SpendLog.model_validate(payload)
        outbound: Final = OutboundUsageEvent(
            event_id=f"outbound:{row.request_id}",
            attempt_id=row.request_id,
            request_id=context.request_id,
            retry_of_attempt_id=context.retry_of_attempt_id,
            call_type=row.call_type,
            started_at=row.started_at,
            ended_at=row.ended_at,
            outcome=context.outcome or (Outcome.SUCCEEDED if row.status == "success" else Outcome.FAILED),
            dimensions=_dimensions(row),
            metrics=_usage(row) if context.usage_known else None,
        )
        return SpendLogEvents(
            inbound=_inbound(row, context, outbound), outbound=outbound, cost=_cost(row, context, outbound)
        )
    except (ValueError, OverflowError):
        return SpendLogAdapterError(reason="Invalid spend-log accounting fields")


def measurements_for(event: InboundUsageEvent | OutboundUsageEvent | CostEvent) -> tuple[Measurement, ...]:
    """Return the complete contribution set for one event, ready for one apply call."""
    metrics: Final = event.charges if isinstance(event, CostEvent) else event.metrics
    occurred_at: Final = event.occurred_at if isinstance(event, CostEvent) else event.started_at
    return tuple(
        Measurement(source_event_id=event.event_id, occurred_at=occurred_at, dimensions=event.dimensions, metric=metric)
        for metric in metrics or ()
    )
