import logging
from collections.abc import Mapping
from datetime import datetime
from enum import Enum
from functools import cache
from typing import Annotated, Final, TypeAlias, TypeVar

from pydantic import (
    BaseModel,
    BeforeValidator,
    ConfigDict,
    PlainSerializer,
    PlainValidator,
    TypeAdapter,
    ValidationError,
)
from pydantic import Field as Member

_LOGGER: Final = logging.getLogger("LiteLLM")

RecordT: Final = TypeVar("RecordT", bound=BaseModel)
ValueT: Final = TypeVar("ValueT")


I64: TypeAlias = Annotated[int, Member(ge=-(2**63), le=2**63 - 1)]
U16: TypeAlias = Annotated[int, Member(ge=0, le=2**16 - 1)]
F64: TypeAlias = Annotated[float, Member(allow_inf_nan=True)]
Flag: TypeAlias = Annotated[bool, Member(strict=True)]


def _float_or_zero(value: object) -> float:
    try:
        return float(value or 0.0)  # pyright: ignore[reportArgumentType]  # float() takes what budget_reservation.py hands it
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"float() refuses {type(value).__name__}") from error


FloatOrZero: TypeAlias = Annotated[float, PlainValidator(_float_or_zero)]
Formatted: TypeAlias = Annotated[str, PlainValidator(str)]

_I128_BOUND: Final = 2**127


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class Stringified(Strict):
    text: str
    truthy: bool
    hashable: bool
    integer: bool
    type_name: str
    hash_error: str | None


def _hash_error(value: object) -> str | None:
    try:
        _ = hash(value)
    except TypeError as error:
        return str(error)
    return None


def _stringified(value: object) -> Stringified:
    try:
        hash_error: Final = _hash_error(value)
        return Stringified(
            text=str(value),
            truthy=bool(value),
            hashable=hash_error is None,
            integer=isinstance(value, int),
            type_name=type(value).__name__,
            hash_error=hash_error,
        )
    except Exception as error:
        raise ValueError(f"str(), bool() or hash() of a {type(value).__name__} raises") from error


def _scalar(value: object) -> bool | int | float | str | Stringified:
    if isinstance(value, bool | float | str | Stringified):
        return value
    if isinstance(value, int) and -_I128_BOUND <= value < _I128_BOUND:
        return value
    return _stringified(value)


def _dumped_scalar(value: object) -> object:
    return value.model_dump() if isinstance(value, Stringified) else value


Scalar: TypeAlias = Annotated[
    bool | int | float | str | Stringified, PlainValidator(_scalar), PlainSerializer(_dumped_scalar)
]
ScalarItem: TypeAlias = Scalar | None


class Record(Strict):
    rejected: tuple[str, ...] | None = None


class Outcome(str, Enum):
    success = "success"
    failure = "failure"
    fallback_success = "fallback_success"
    fallback_failure = "fallback_failure"


class MetadataKind(str, Enum):
    mapping = "mapping"
    missing = "missing"
    null = "null"
    other = "other"


def _datetime_members(value: datetime) -> Mapping[str, object]:
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


def _moment_members(value: object) -> object:
    if isinstance(value, Moment):
        return value
    if isinstance(value, datetime):
        return _datetime_members(value)
    raise ValueError(f"a {type(value).__name__} is not a datetime")


class Moment(Strict):
    year: int
    month: int
    day: int
    hour: int
    minute: int
    second: int
    microsecond: int
    utc_offset_seconds: int | None = None


MomentInput: TypeAlias = Annotated[Moment, BeforeValidator(_moment_members)]


class KeyIdentity(Record):
    key_hash: Scalar | None = None
    api_key: Scalar | None = None
    key_alias: Scalar | None = None
    key_name: Scalar | None = None
    user_id: Scalar | None = None
    user_email: Scalar | None = None
    team_id: Scalar | None = None
    team_alias: Scalar | None = None
    org_id: Scalar | None = None
    org_alias: Scalar | None = None
    project_id: Scalar | None = None
    project_alias: Scalar | None = None
    end_user_id: Scalar | None = None
    agent_id: Scalar | None = None
    invoked_agent_id: Scalar | None = None
    team_priority: Scalar | None = None
    key_priority: Scalar | None = None


class LoggedKeyIdentity(Record):
    metadata_kind: MetadataKind | None = None
    missing_keys: tuple[str, ...] | None = None
    key_hash: Scalar | None = None
    key_alias: Scalar | None = None
    key_spend: Scalar | None = None
    key_max_budget: Scalar | None = None
    key_budget_reset_at: Scalar | None = None
    user_id: Scalar | None = None
    user_email: Scalar | None = None
    user_spend: Scalar | None = None
    user_max_budget: Scalar | None = None
    team_id: Scalar | None = None
    team_alias: Scalar | None = None
    team_spend: Scalar | None = None
    team_max_budget: Scalar | None = None
    org_id: Scalar | None = None
    org_alias: Scalar | None = None
    project_id: Scalar | None = None
    project_alias: Scalar | None = None
    end_user_id: Scalar | None = None
    request_route: Scalar | None = None
    auth_metadata_priority: Scalar | None = None
    agent_id: Scalar | None = None
    request_user: Scalar | None = None
    cost_tracking_end_user_id: Scalar | None = None


class EstimatedOutputTokens(Record):
    key_per_model: I64 | None = None
    key_default: I64 | None = None
    team_per_model: I64 | None = None
    team_default: I64 | None = None


class AgentPolicy(Record):
    rpm_limit: Scalar | None = None
    tpm_limit: Scalar | None = None
    session_rpm_limit: Scalar | None = None
    session_tpm_limit: Scalar | None = None


class KeyLimits(Record):
    rpm_limit: Scalar | None = None
    tpm_limit: Scalar | None = None
    max_parallel_requests: Scalar | None = None
    budget_throttle_pct: Scalar | None = None
    user_rpm_limit: Scalar | None = None
    user_tpm_limit: Scalar | None = None
    team_rpm_limit: Scalar | None = None
    team_tpm_limit: Scalar | None = None
    team_member_rpm_limit: Scalar | None = None
    team_member_tpm_limit: Scalar | None = None
    end_user_rpm_limit: Scalar | None = None
    end_user_tpm_limit: Scalar | None = None
    organization_rpm_limit: Scalar | None = None
    organization_tpm_limit: Scalar | None = None
    rpm_limit_type: str | None = None
    tpm_limit_type: str | None = None
    estimated_output_tokens: EstimatedOutputTokens | None = None
    managed_agent_policy: AgentPolicy | None = None
    invoked_agent_policy: AgentPolicy | None = None


class BudgetPlanCounter(Strict):
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
    counter_key: Formatted | None = None
    entity_type: str | None = None
    entity_id: str | None = None
    reserved_cost: FloatOrZero | None = None
    applied_adjustment: FloatOrZero | None = None


class BudgetReservation(Record):
    reserved_cost: FloatOrZero | None = None
    entries: tuple[ReservationEntry | None, ...] | None = None
    entries_is_list: bool | None = None
    finalized: Flag | None = None
    callback_bound: Flag | None = None
    input_cost: FloatOrZero | None = None
    input_tokens: I64 | None = None
    not_a_mapping: Scalar | None = None


class RateLimitStatus(Strict):
    code: str | None = None
    current_limit: I64 | None = None
    limit_remaining: I64 | None = None
    rate_limit_type: str | None = None
    descriptor_key: str | None = None
    descriptor_value: str | None = None


class RateLimitStatuses(Record):
    overall_code: str | None = None
    statuses: tuple[RateLimitStatus, ...] | None = None


class LimiterRequest(Record):
    session_id: str | None = None
    session_id_error_type: str | None = None
    session_id_error_text: str | None = None
    tags: tuple[str, ...] | None = None
    resolved_agent_id: Scalar | None = None
    resolved_agent_id_error_type: str | None = None
    resolved_agent_id_error_text: str | None = None
    fallback_on_saturation: Flag | None = None
    fallback_on_saturation_invalid: str | None = None
    fallback_walk_armed: Flag | None = None


class OutputCap(Record):
    explicit: Scalar | None = None


class AttemptTiming(Record):
    start: MomentInput | None = None
    end: MomentInput | None = None
    api_call_start: MomentInput | None = None
    completion_start: MomentInput | None = None
    completion_start_present: bool | None = None
    queue_seconds: Scalar | None = None
    overhead_ms: Scalar | None = None
    logged_start: MomentInput | None = None
    logged_end: MomentInput | None = None


class UsageRecord(Record):
    prompt_tokens: Scalar | None = None
    completion_tokens: Scalar | None = None
    total_tokens: Scalar | None = None
    cached_tokens: Scalar | None = None
    prompt_audio_tokens: Scalar | None = None
    cache_write_tokens: Scalar | None = None
    cache_creation_tokens: Scalar | None = None
    reasoning_tokens: Scalar | None = None
    completion_audio_tokens: Scalar | None = None
    cache_read_input_tokens: Scalar | None = None
    cache_creation_input_tokens: Scalar | None = None
    duration_seconds: Scalar | None = None
    output_image_count: Scalar | None = None


class Usage(Record):
    response: UsageRecord | None = None
    streamed: UsageRecord | None = None
    combined: UsageRecord | None = None
    upstream: UsageRecord | None = None
    socket_session: UsageRecord | None = None
    logged: UsageRecord | None = None


class ZeroCost(Record):
    reason: Scalar | None = None
    reason_present: bool | None = None


class Cost(Record):
    response_cost: Scalar | None = None
    zero_cost: ZeroCost | None = None


class ModelInfo(Record):
    id: Scalar | None = None
    tpm: Scalar | None = None
    rpm: Scalar | None = None


class Deployment(Record):
    model_info: ModelInfo | None = None
    provider: Scalar | None = None
    tpm: Scalar | None = None
    rpm: Scalar | None = None
    original_model_group: str | None = None
    params_present: bool | None = None


class GuardrailEntry(Record):
    mapping: bool | None = None
    modes: tuple[ScalarItem, ...] | None = None
    duration: Scalar | None = None


class RateLimitState(Record):
    key_model_group: Scalar | None = None
    key_remaining_requests: Scalar | None = None
    key_remaining_tokens: Scalar | None = None
    limiter_headers: Mapping[str, Scalar] | None = None
    provider_remaining_requests: Scalar | None = None
    provider_remaining_tokens: Scalar | None = None
    hidden_params_present: bool | None = None
    additional_headers_present: bool | None = None


class LoggedModelGroup(Record):
    value: Scalar | None = None
    error_type: str | None = None
    error_text: str | None = None


class InputSequenceLength(Record):
    bucket: str | None = None
    error_type: str | None = None
    error_text: str | None = None


class LoggedRequest(Record):
    client_ip: Scalar | None = None
    user_agent: Scalar | None = None
    service_tier: str | None = None
    stream: Scalar | None = None


class MetadataLabel(Strict):
    name: str
    value: str


class McpToolCall(Record):
    name: Scalar | None = None
    server_name: Scalar | None = None


_REPORTED: Final[set[tuple[str, str]]] = set()  # mutable-ok: each rejected input is logged once per process


def _report(owner: str, member: str, error: ValidationError) -> None:
    if (owner, member) in _REPORTED:
        return
    _REPORTED.add((owner, member))
    _LOGGER.warning(
        "custom logger v2: %s.%s holds a value the contract cannot carry, so it is named in rejected: %s",
        owner,
        member,
        error.errors(include_url=False, include_input=False),
    )


def reset_coercion_reports_for_tests() -> None:
    _REPORTED.clear()


@cache
def _member_adapter(record: type[BaseModel], member: str) -> TypeAdapter[object]:
    return TypeAdapter(record.model_fields[member].annotation)


_REJECTED: Final = object()


def _carried(record: type[BaseModel], member: str, raw: object) -> object:
    try:
        return _member_adapter(record, member).validate_python(raw)
    except ValidationError as error:
        _report(record.__name__, member, error)
        return _REJECTED


def coerce_record(record: type[RecordT], members: Mapping[str, object], rejected: tuple[str, ...] = ()) -> RecordT:
    carried: Final = tuple(
        (member, _carried(record, member, raw))
        for member, raw in members.items()
        if raw is not None and member != "rejected" and member in record.model_fields
    )
    refused: Final = (*rejected, *(member for member, value in carried if value is _REJECTED))
    return record.model_validate(
        {
            **{member: value for member, value in carried if value is not _REJECTED},
            **({"rejected": refused} if refused else {}),
        }
    )


def coerce_value(owner: str, adapter: TypeAdapter[ValueT], raw: object) -> ValueT | None:
    if raw is None:
        return None
    try:
        return adapter.validate_python(raw)
    except ValidationError as error:
        _report(owner, "value", error)
        return None


__all__ = [
    "AgentPolicy",
    "AttemptTiming",
    "BudgetPlan",
    "BudgetPlanCounter",
    "BudgetReservation",
    "Cost",
    "Deployment",
    "EstimatedOutputTokens",
    "GuardrailEntry",
    "InputSequenceLength",
    "KeyIdentity",
    "KeyLimits",
    "LimiterRequest",
    "LoggedKeyIdentity",
    "LoggedModelGroup",
    "LoggedRequest",
    "McpToolCall",
    "MetadataKind",
    "MetadataLabel",
    "ModelInfo",
    "Moment",
    "Outcome",
    "OutputCap",
    "RateLimitState",
    "RateLimitStatus",
    "RateLimitStatuses",
    "Record",
    "ReservationEntry",
    "Scalar",
    "ScalarItem",
    "Strict",
    "Stringified",
    "Usage",
    "UsageRecord",
    "ZeroCost",
    "coerce_record",
    "coerce_value",
    "reset_coercion_reports_for_tests",
]
