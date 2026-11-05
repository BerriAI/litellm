"""Daily comparisons over existing aggregated counters, outside the request path."""

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Final, Literal, Protocol

from litellm.repositories.usage_alert_metrics import DailyAlertMetrics


class ComparisonRule(Protocol):
    @property
    def metric(self) -> Literal["spend", "requests", "latency", "errors"]: ...

    @property
    def direction(self) -> Literal["increase", "decrease"]: ...

    @property
    def multiplier(self) -> float: ...

    @property
    def baseline_days(self) -> int: ...

    @property
    def min_history_days(self) -> int: ...

    @property
    def min_requests(self) -> int: ...

    @property
    def min_value(self) -> float: ...


@dataclass(frozen=True, slots=True)
class DailyComparison:
    baseline_sample_days: int
    observed: float
    baseline: float
    threshold: float
    breached: bool


def _value(metric: str, rows: Sequence[DailyAlertMetrics], days: int) -> float | None:
    if metric == "latency":
        timed: Final = sum(row.timed_requests for row in rows)
        return sum(row.response_time_ms for row in rows) / timed if timed else None
    if metric == "spend":
        return sum(row.spend for row in rows) / days
    if metric == "errors":
        return sum(row.errors for row in rows) / days
    return sum(row.requests for row in rows) / days


def compare_daily_usage(
    rule: ComparisonRule, rows: Sequence[DailyAlertMetrics], target: date
) -> DailyComparison | None:
    """None means insufficient evidence; it must not resolve an existing incident."""
    start: Final = target - timedelta(days=rule.baseline_days)
    history: Final = tuple(row for row in rows if start <= row.day < target)
    current: Final = tuple(row for row in rows if row.day == target)
    sample_days: Final = len({row.day for row in history})
    if sample_days < rule.min_history_days:
        return None
    baseline_requests: Final = sum(row.requests for row in history) / sample_days
    current_requests: Final = sum(row.requests for row in current)
    if baseline_requests < rule.min_requests:
        return None
    if rule.direction == "increase" and current_requests < rule.min_requests:
        return None
    if rule.metric == "latency":
        baseline_timed: Final = sum(row.timed_requests for row in history) / sample_days
        if baseline_timed < rule.min_requests or sum(row.timed_requests for row in current) < rule.min_requests:
            return None
    baseline: Final = _value(rule.metric, history, sample_days)
    observed: Final = _value(rule.metric, current, 1)
    if baseline is None or baseline < 0 or observed is None:
        return None
    if baseline == 0:
        if rule.direction == "decrease":
            return None
        return DailyComparison(sample_days, observed, 0, rule.min_value, observed > rule.min_value)
    threshold: Final = baseline * rule.multiplier if rule.direction == "increase" else baseline / rule.multiplier
    crossed: Final = observed >= threshold if rule.direction == "increase" else observed <= threshold
    return DailyComparison(
        sample_days, observed, baseline, threshold, crossed and abs(observed - baseline) >= rule.min_value
    )
