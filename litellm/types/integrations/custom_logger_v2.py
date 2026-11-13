import logging
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from functools import cache
from typing import Annotated, Final, TypeVar

from pydantic import BaseModel, BeforeValidator, ConfigDict, TypeAdapter, ValidationError
from pydantic import Field as Member

_LOGGER: Final = logging.getLogger("LiteLLM")

RecordT: Final = TypeVar("RecordT", bound=BaseModel)
ValueT: Final = TypeVar("ValueT")


I64 = Annotated[int, Member(ge=-(2**63), le=2**63 - 1)]
U16 = Annotated[int, Member(ge=0, le=2**16 - 1)]
F64 = Annotated[float, Member(allow_inf_nan=False)]


class Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Outcome(str, Enum):
    success = "success"
    failure = "failure"
    fallback_success = "fallback_success"
    fallback_failure = "fallback_failure"


def _moment_members(value: object) -> object:
    if not isinstance(value, datetime):
        return value
    offset: Final = value.utcoffset()
    return {
        "year": value.year,
        "month": value.month,
        "day": value.day,
        "hour": value.hour,
        "minute": value.minute,
        "second": value.second,
        "microsecond": value.microsecond,
        "utc_offset_seconds": None if offset is None else int(offset.total_seconds()),
    }


class Moment(Record):
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int
    utc_offset_seconds: int | None = None


MomentInput = Annotated[Moment, BeforeValidator(_moment_members)]


class ModelPreference(Record):
    model: str | None = None
    alternates: tuple[str, ...] | None = None


class KeyIdentity(Record):
    key_hash: str | None = None
    api_key: str | None = None
    key_alias: str | None = None
    key_name: str | None = None
    user_id: str | None = None
    user_email: str | None = None
    team_id: str | None = None
    team_alias: str | None = None
    org_id: str | None = None
    org_alias: str | None = None
    project_id: str | None = None
    project_alias: str | None = None
    end_user_id: str | None = None
    agent_id: str | None = None
    invoked_agent_id: str | None = None
    team_priority: str | None = None
    key_priority: str | None = None
    model_preferences: tuple[ModelPreference, ...] | None = None


class EstimatedOutputTokens(Record):
    key_per_model: I64 | None = None
    key_default: I64 | None = None
    team_per_model: I64 | None = None
    team_default: I64 | None = None


class KeyLimits(Record):
    rpm_limit: I64 | None = None
    tpm_limit: I64 | None = None
    max_parallel_requests: I64 | None = None
    budget_throttle_pct: F64 | None = None
    user_rpm_limit: I64 | None = None
    user_tpm_limit: I64 | None = None
    team_rpm_limit: I64 | None = None
    team_tpm_limit: I64 | None = None
    team_member_rpm_limit: I64 | None = None
    team_member_tpm_limit: I64 | None = None
    end_user_rpm_limit: I64 | None = None
    end_user_tpm_limit: I64 | None = None
    organization_rpm_limit: I64 | None = None
    organization_tpm_limit: I64 | None = None
    rpm_limit_type: str | None = None
    tpm_limit_type: str | None = None
    estimated_output_tokens: EstimatedOutputTokens | None = None


class BudgetPlanCounter(Record):
    counter_key: str | None = None
    max_budget: F64 | None = None
    fallback_spend: F64 | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    spend_log_entity_id: str | None = None


class BudgetPlan(Record):
    counters: tuple[BudgetPlanCounter, ...] | None = None
    reservation_cost: F64 | None = None
    input_cost: F64 | None = None
    input_tokens: I64 | None = None
    throttle_counter: str | None = None
    fail_closed: bool | None = None
    status_code: U16 | None = None
    reservation_id: str | None = None


class ReservationEntry(Record):
    counter_key: str | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    reserved_cost: F64 | None = None
    applied_adjustment: F64 | None = None


class BudgetReservation(Record):
    reserved_cost: F64 | None = None
    entries: tuple[ReservationEntry, ...] | None = None
    finalized: bool | None = None
    input_cost: F64 | None = None
    input_tokens: I64 | None = None


class RateLimitStatus(Record):
    code: str | None = None
    current_limit: I64 | None = None
    limit_remaining: I64 | None = None
    rate_limit_type: str | None = None
    descriptor_key: str | None = None
    descriptor_value: str | None = None


class RateLimitStatuses(Record):
    overall_code: str | None = None
    statuses: tuple[RateLimitStatus, ...] | None = None


class OutputCap(Record):
    explicit: I64 | None = None


class AttemptTiming(Record):
    start: MomentInput | None = None
    end: MomentInput | None = None
    api_call_start: MomentInput | None = None
    completion_start: MomentInput | None = None
    completion_start_present: bool | None = None
    queue_seconds: F64 | None = None
    overhead_ms: F64 | None = None


class UsageRecord(Record):
    prompt_tokens: I64 | None = None
    completion_tokens: I64 | None = None
    total_tokens: I64 | None = None
    cached_tokens: I64 | None = None
    prompt_audio_tokens: I64 | None = None
    cache_write_tokens: I64 | None = None
    cache_creation_tokens: I64 | None = None
    reasoning_tokens: I64 | None = None
    completion_audio_tokens: I64 | None = None
    cache_read_input_tokens: I64 | None = None
    cache_creation_input_tokens: I64 | None = None
    duration_seconds: F64 | None = None
    output_image_count: I64 | None = None


class Usage(Record):
    response: UsageRecord | None = None
    streamed: UsageRecord | None = None
    combined: UsageRecord | None = None
    upstream: UsageRecord | None = None
    socket_session: UsageRecord | None = None
    logged: UsageRecord | None = None


class ZeroCost(Record):
    reason: str | None = None


class Cost(Record):
    response_cost: F64 | None = None
    zero_cost: ZeroCost | None = None


class ModelInfo(Record):
    id: str | None = None
    tpm: I64 | None = None
    rpm: I64 | None = None


class Deployment(Record):
    model_info: ModelInfo | None = None
    provider: str | None = None
    tpm: I64 | None = None
    rpm: I64 | None = None
    original_model_group: str | None = None
    params_present: bool | None = None


class GuardrailEntry(Record):
    modes: tuple[str, ...] | None = None
    duration: F64 | None = None


class RateLimitState(Record):
    key_model_group: str | None = None
    key_remaining_requests: F64 | None = None
    key_remaining_tokens: F64 | None = None
    limiter_headers: Mapping[str, F64] | None = None
    provider_remaining_requests: F64 | None = None
    provider_remaining_tokens: F64 | None = None
    hidden_params_present: bool | None = None
    additional_headers_present: bool | None = None


class LoggedRequest(Record):
    client_ip: str | None = None
    user_agent: str | None = None
    service_tier: str | None = None
    stream: bool | None = None


class McpToolCall(Record):
    name: str | None = None
    server_name: str | None = None


_REPORTED: Final[set[tuple[str, str]]] = set()  # mutable-ok: each coercion failure is logged once per process


def _report(owner: str, member: str, error: ValidationError) -> None:
    if (owner, member) in _REPORTED:
        return
    _REPORTED.add((owner, member))
    _LOGGER.warning(
        "custom logger v2: %s.%s could not be coerced, so it is left absent: %s",
        owner,
        member,
        error.errors(include_url=False, include_input=False),
    )


def reset_coercion_reports_for_tests() -> None:
    _REPORTED.clear()


@cache
def _member_adapter(record: type[BaseModel], member: str) -> TypeAdapter[object]:
    return TypeAdapter(record.model_fields[member].annotation)


def coerce_record(record: type[RecordT], members: Mapping[str, object]) -> RecordT:
    kept: Final[dict[str, object]] = {}  # mutable-ok: filled member by member
    for member, raw in members.items():
        if raw is None:
            continue
        try:
            kept[member] = _member_adapter(record, member).validate_python(raw)
        except ValidationError as error:
            _report(record.__name__, member, error)
    return record.model_validate(kept)


def coerce_value(owner: str, adapter: TypeAdapter[ValueT], raw: object) -> ValueT | None:
    if raw is None:
        return None
    try:
        return adapter.validate_python(raw)
    except ValidationError as error:
        _report(owner, "value", error)
        return None


__all__ = [
    "AttemptTiming",
    "BudgetPlan",
    "BudgetPlanCounter",
    "BudgetReservation",
    "Cost",
    "Deployment",
    "EstimatedOutputTokens",
    "GuardrailEntry",
    "KeyIdentity",
    "KeyLimits",
    "LoggedRequest",
    "McpToolCall",
    "ModelInfo",
    "ModelPreference",
    "Moment",
    "Outcome",
    "OutputCap",
    "RateLimitState",
    "RateLimitStatus",
    "RateLimitStatuses",
    "Record",
    "ReservationEntry",
    "Usage",
    "UsageRecord",
    "ZeroCost",
    "coerce_record",
    "coerce_value",
    "reset_coercion_reports_for_tests",
]
