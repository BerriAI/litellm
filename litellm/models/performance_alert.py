import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta, timezone
from typing import Final, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, Field

from litellm.models.alert import MAX_ALERT_ID, AlertDetection, AlertType, MetricEvidence

PerformanceScope: TypeAlias = Literal["gateway", "model", "team", "user", "key"]
SCOPES: Final[tuple[PerformanceScope, ...]] = ("gateway", "model", "team", "user", "key")
BASELINE_DAYS: Final = 7

MIN_ACTIVE_DAYS: Final = 4
SPEND_MULTIPLIER: Final = 3
SPEND_MIN_INCREASE_USD: Final = 50
TRAFFIC_SPIKE_MULTIPLIER: Final = 10
TRAFFIC_SPIKE_MIN_INCREASE: Final = 1000
TRAFFIC_DROP_RATIO: Final = 0.2
TRAFFIC_DROP_MIN_BASELINE: Final = 500
LATENCY_MULTIPLIER: Final = 3
LATENCY_MIN_INCREASE_MS: Final = 500
LATENCY_MIN_TIMED_REQUESTS: Final = 100
RELIABILITY_MULTIPLIER: Final = 3
RELIABILITY_MIN_INCREASE: Final = 0.02
RELIABILITY_MIN_REQUESTS: Final = 500


class ResourceWindow(BaseModel):
    """One resource's totals for the evaluated day, and summed over the seven days before it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    resource_id: str = Field(min_length=1, max_length=1024)
    spend: float = Field(allow_inf_nan=False)
    base_spend: float = Field(allow_inf_nan=False)
    requests: int = Field(ge=0, le=MAX_ALERT_ID)
    base_requests: int = Field(ge=0, le=MAX_ALERT_ID)
    failed: int = Field(ge=0, le=MAX_ALERT_ID)
    base_failed: int = Field(ge=0, le=MAX_ALERT_ID)
    response_ms: int = Field(ge=0, le=MAX_ALERT_ID)
    timed: int = Field(ge=0, le=MAX_ALERT_ID)
    base_response_ms: int = Field(ge=0, le=MAX_ALERT_ID)
    base_timed: int = Field(ge=0, le=MAX_ALERT_ID)
    active_days: int = Field(ge=0, le=BASELINE_DAYS)
    timed_days: int = Field(ge=0, le=BASELINE_DAYS)
    exists: bool


class SweepState(BaseModel):
    """Where the daily sweep stands. Each scope is walked from a day-derived start to the end of its IDs,
    then wraps to cover the IDs before that start, so an unfinished sweep leaves different resources out each day."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    day: date
    scope: PerformanceScope = "gateway"
    cursor: str = Field(default="", max_length=1024)
    wrapped: bool = False
    checked: int = Field(default=0, ge=0)
    skipped: int = Field(default=0, ge=0)
    completed: bool = False


@dataclass(frozen=True, slots=True)
class Claim:
    owner: str
    state: SweepState | None


class SweepOwnershipLostError(Exception):
    pass


class ResourceUnreadableError(Exception):
    """A page that timed out or held invalid data. Retried alone, then stepped over.
    Any other database error ends the run so the next run resumes from the same place."""


class PerformanceAlertStore(Protocol):
    async def claim(self, interval: timedelta) -> Claim | None: ...
    async def accounting_pending(self, day: date) -> bool: ...
    async def key_index_ready(self) -> bool: ...
    async def save(self, owner: str, state: SweepState) -> None: ...
    async def evaluate_page(self, owner: str, state: SweepState, limit: int) -> SweepState: ...
    async def skip_next(self, owner: str, state: SweepState) -> SweepState: ...


def target_day(now: datetime) -> date:
    """The last complete UTC day, switching to a new day at 01:00 UTC so late accounting has time to land."""
    return (now.astimezone(timezone.utc) - timedelta(hours=1)).date() - timedelta(days=1)


def rotation_start(day: date) -> str:
    return hashlib.sha256(day.isoformat().encode()).hexdigest()


def new_sweep(day: date) -> SweepState:
    return SweepState(day=day)


def next_scope(state: SweepState) -> SweepState:
    position: Final = SCOPES.index(state.scope)
    if position + 1 == len(SCOPES):
        return state.model_copy(update={"cursor": "", "wrapped": False, "completed": True})
    following: Final = SCOPES[position + 1]
    return state.model_copy(update={"scope": following, "cursor": rotation_start(state.day), "wrapped": False})


def advance(state: SweepState, resource_ids: Sequence[str], limit: int, skipped: int = 0) -> SweepState:
    counted: Final = state.model_copy(
        update={"checked": state.checked + len(resource_ids) - skipped, "skipped": state.skipped + skipped}
    )
    if state.scope == "gateway":
        return next_scope(counted)
    if len(resource_ids) == limit:
        return counted.model_copy(update={"cursor": resource_ids[-1]})
    if state.wrapped:
        return next_scope(counted)
    return counted.model_copy(update={"cursor": "", "wrapped": True})


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _alert(
    scope: PerformanceScope,
    window: ResourceWindow,
    day: date,
    kind: AlertType,
    value: float,
    baseline: float,
    sample_size: int,
) -> AlertDetection:
    start: Final = datetime.combine(day, time.min, timezone.utc)
    unit: Final = {"spend": "usd", "traffic": "requests", "latency": "milliseconds", "reliability": "ratio"}[kind]
    return AlertDetection.model_validate(
        {
            "type": kind,
            "resource_type": scope,
            "resource_id": "gateway" if scope == "gateway" else window.resource_id,
            "window_start": start,
            "window_end": start + timedelta(days=1),
            "evidence": MetricEvidence.model_validate(
                {"value": value, "baseline": baseline, "unit": unit, "sample_size": sample_size}
            ),
        }
    )


def _spend_spike(window: ResourceWindow) -> tuple[float, float] | None:
    baseline: Final = window.base_spend / BASELINE_DAYS
    triggered: Final = (
        window.active_days >= MIN_ACTIVE_DAYS
        and baseline > 0
        and window.spend >= SPEND_MULTIPLIER * baseline
        and window.spend - baseline >= SPEND_MIN_INCREASE_USD
    )
    return (window.spend, baseline) if triggered else None


def _traffic_change(window: ResourceWindow) -> tuple[float, float] | None:
    baseline: Final = window.base_requests / BASELINE_DAYS
    spike: Final = (
        window.active_days >= MIN_ACTIVE_DAYS
        and window.requests >= TRAFFIC_SPIKE_MULTIPLIER * baseline
        and window.requests - baseline >= TRAFFIC_SPIKE_MIN_INCREASE
    )
    drop: Final = (
        window.exists
        and window.active_days == BASELINE_DAYS
        and baseline >= TRAFFIC_DROP_MIN_BASELINE
        and window.requests < TRAFFIC_DROP_RATIO * baseline
    )
    return (window.requests, baseline) if spike or drop else None


def _slower_responses(window: ResourceWindow) -> tuple[float, float] | None:
    current: Final = _ratio(window.response_ms, window.timed)
    baseline: Final = _ratio(window.base_response_ms, window.base_timed)
    triggered: Final = (
        window.timed >= LATENCY_MIN_TIMED_REQUESTS
        and window.timed_days >= MIN_ACTIVE_DAYS
        and baseline > 0
        and current >= LATENCY_MULTIPLIER * baseline
        and current - baseline >= LATENCY_MIN_INCREASE_MS
    )
    return (current, baseline) if triggered else None


def _more_failures(window: ResourceWindow) -> tuple[float, float] | None:
    current: Final = min(1.0, _ratio(window.failed, window.requests))
    baseline: Final = min(1.0, _ratio(window.base_failed, window.base_requests))
    triggered: Final = (
        window.active_days >= MIN_ACTIVE_DAYS
        and window.requests >= RELIABILITY_MIN_REQUESTS
        and current >= RELIABILITY_MULTIPLIER * baseline
        and current - baseline >= RELIABILITY_MIN_INCREASE
    )
    return (current, baseline) if triggered else None


def detect(scope: PerformanceScope, window: ResourceWindow, day: date) -> tuple[AlertDetection, ...]:
    checks: Final[tuple[tuple[AlertType, tuple[float, float] | None, int], ...]] = (
        ("spend", _spend_spike(window), window.requests),
        ("traffic", _traffic_change(window), window.requests),
        ("latency", _slower_responses(window), window.timed),
        ("reliability", _more_failures(window), window.requests),
    )
    return tuple(
        _alert(scope, window, day, kind, found[0], found[1], sample_size)
        for kind, found, sample_size in checks
        if found is not None
    )
