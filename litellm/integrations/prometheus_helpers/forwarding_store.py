from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final, Literal, TypeAlias

from prometheus_client import Counter, Gauge, Histogram
from pydantic import TypeAdapter

from litellm.constants import PROMETHEUS_OVERFLOW_SERIES_LABEL_VALUE
from litellm.integrations.prometheus_helpers.shared_prometheus_series_admissions import (
    SharedPrometheusSeriesAdmissions,
)

PrometheusKind: TypeAlias = Literal["counter", "gauge", "histogram"]
GaugeMode: TypeAlias = Literal[
    "all", "liveall", "min", "livemin", "max", "livemax", "sum", "livesum", "mostrecent", "livemostrecent"
]
ForwardedAction: TypeAlias = Literal["inc", "dec", "set", "observe", "remove", "clear"]
ForwardedOp: TypeAlias = tuple[str, list[str] | None, ForwardedAction, float]
ForwardedMetric: TypeAlias = Counter | Gauge | Histogram

_GAUGE_MODE: Final[TypeAdapter[GaugeMode]] = TypeAdapter(GaugeMode)


def forwarded_metric(
    kind: PrometheusKind,
    name: str,
    documentation: str,
    labelnames: Sequence[str],
    buckets: Sequence[float],
    multiprocess_mode: str,
) -> ForwardedMetric:
    match kind:
        case "counter":
            return Counter(name, documentation, labelnames=labelnames)
        case "gauge":
            return Gauge(
                name,
                documentation,
                labelnames=labelnames,
                multiprocess_mode=_GAUGE_MODE.validate_python(multiprocess_mode),
            )
        case "histogram":
            return Histogram(name, documentation, labelnames=labelnames, buckets=buckets)


@dataclass(frozen=True, slots=True)
class _Forwarded:
    metric: ForwardedMetric
    max_series: int | None
    shares_overflow: bool


def _apply(target: ForwardedMetric, action: ForwardedAction, amount: float) -> None:
    match target, action:
        case Counter(), "inc":
            target.inc(amount)
        case Gauge(), "inc":
            target.inc(amount)
        case Gauge(), "dec":
            target.dec(amount)
        case Gauge(), "set":
            target.set(amount)
        case Histogram(), "observe":
            target.observe(amount)
        case _:
            raise AttributeError(f"'{type(target).__name__}' object has no attribute '{action}'")


class ForwardingStore:
    __slots__ = ("_admissions", "_max_series", "_metrics")

    def __init__(self, admissions: SharedPrometheusSeriesAdmissions, max_series: int | None) -> None:
        self._admissions: Final = admissions
        self._max_series: Final = max_series
        self._metrics: Final[dict[str, _Forwarded]] = {}  # mutable-ok: one entry per metric, added at registration

    def add(self, name: str, metric: ForwardedMetric, wrapped: bool, shares_overflow: bool) -> None:
        self._metrics[name] = _Forwarded(metric, self._max_series if wrapped else None, shares_overflow)

    def __call__(self, ops: Sequence[ForwardedOp]) -> None:
        for name, values, action, amount in ops:
            forwarded = self._metrics[name]
            if action == "remove":
                forwarded.metric.remove(*(values or ()))
            elif action == "clear":
                forwarded.metric.clear()
            else:
                target = forwarded.metric if values is None else self._child(name, forwarded, tuple(values))
                if target is not None:
                    _apply(target, action, amount)

    def _child(self, name: str, forwarded: _Forwarded, values: tuple[str, ...]) -> ForwardedMetric | None:
        if forwarded.max_series is None or self._admissions.admit_series(name, values, forwarded.max_series):
            return forwarded.metric.labels(*values)
        if forwarded.shares_overflow:
            return forwarded.metric.labels(*(PROMETHEUS_OVERFLOW_SERIES_LABEL_VALUE,) * len(values))
        return None
