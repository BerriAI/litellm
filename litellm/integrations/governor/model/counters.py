"""Counter value types and window descriptors."""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

from litellm.integrations.governor.model.subjects import Subject

CounterKind = Literal["spend", "rpm", "tpm", "inflight", "cost_throttle"]
WindowKind = Literal["daily", "weekly", "monthly", "rolling"]

_WINDOW_PERIODS: dict[WindowKind, timedelta] = {
    "daily": timedelta(days=1),
    "weekly": timedelta(weeks=1),
    "monthly": timedelta(days=30),
    "rolling": timedelta(0),
}


@dataclass(frozen=True)
class BudgetWindow:
    kind: WindowKind
    period: timedelta
    bucket_key: str

    @classmethod
    def current(cls, kind: WindowKind, now: datetime) -> "BudgetWindow":
        period = _WINDOW_PERIODS[kind]
        if kind == "daily":
            bucket = now.strftime("%Y-%m-%d")
        elif kind == "weekly":
            iso = now.isocalendar()
            bucket = f"{iso.year}-W{iso.week:02d}"
        elif kind == "monthly":
            bucket = now.strftime("%Y-%m")
        else:
            bucket = "rolling"
        return cls(kind=kind, period=period, bucket_key=bucket)


@dataclass(frozen=True)
class RateLimitWindow:
    period_seconds: int
    capacity: int
    algorithm: Literal["gcra", "sliding_window"]


@dataclass(frozen=True)
class GcraState:
    """Persisted theoretical arrival time for a GCRA token bucket."""

    tat_seconds: float


@dataclass(frozen=True)
class Counter:
    kind: CounterKind
    subject: Subject
    policy_id: str
    bucket_key: str
    value: float
    ttl_seconds: int


@dataclass(frozen=True)
class Outcome:
    success: bool
    actual_cost: float
    actual_input_tokens: int
    actual_output_tokens: int
    error_class: str | None
    upstream_status: int | None
