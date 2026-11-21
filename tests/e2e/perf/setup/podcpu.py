"""Estimate gateway CPU from PodMetrics source windows, clipped to the measured pass. Polling a stale
response adds no CPU. Reading results waits up to a minute for windows covering the pass's end. Gaps use
the mean of their neighbouring rates, or the one available rate at an interval boundary. Less than 90%
real-window coverage is missing, not zero. The estimate assumes uniform CPU within each source window, including at pass and
ramp boundaries; only cumulative counters could measure short bursts exactly. The runner's read-only
service account cannot exec into gateway pods or read kubelet counters."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
from itertools import chain, pairwise
from pathlib import Path
from typing import Final
from urllib.parse import quote

from e2e_http import AuthHeaders, Success, get_external
from pydantic import AwareDatetime, BaseModel, ConfigDict

SELECTOR_ENV: Final = "E2E_PERF_GATEWAY_POD_SELECTOR"
_SERVICE_ACCOUNT: Final = Path("/var/run/secrets/kubernetes.io/serviceaccount")
_INTERVAL_S: Final = 5.0
_MIN_COVERAGE: Final = 0.9
_SUFFIX_CORES: Final = {"n": 1e-9, "u": 1e-6, "m": 1e-3}
_DURATION_PART: Final = re.compile(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)")
_DURATION_UNITS: Final = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}


def _log(event: str, **fields: str | float | bool | None | tuple[str, ...]) -> None:
    print("gateway_cpu", json.dumps({"event": event, **fields}, sort_keys=True), flush=True)


class _Usage(BaseModel):
    model_config = ConfigDict(frozen=True)

    cpu: str


class _Container(BaseModel):
    model_config = ConfigDict(frozen=True)

    usage: _Usage


class _Metadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str


class PodMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: _Metadata
    timestamp: AwareDatetime
    window: str
    containers: list[_Container]


class PodMetricsList(BaseModel):
    model_config = ConfigDict(frozen=True)

    items: list[PodMetrics]


def cores(quantity: str) -> float:
    """A Kubernetes CPU quantity (`1500m`, `250000000n`, `2`) as cores."""
    if quantity and quantity[-1] in _SUFFIX_CORES:
        return float(quantity[:-1]) * _SUFFIX_CORES[quantity[-1]]
    return float(quantity)


@dataclass(frozen=True, slots=True)
class _Window:
    pod: str
    start: float
    end: float
    rate: float


def _window(pod: PodMetrics) -> _Window | None:
    parts: Final = tuple(_DURATION_PART.finditer(pod.window))
    if not parts or "".join(part.group() for part in parts) != pod.window:
        return None
    duration: Final = sum(float(part[1]) * _DURATION_UNITS[part[2]] for part in parts)
    try:
        rate: Final = sum(cores(container.usage.cpu) for container in pod.containers)
    except ValueError:
        return None
    if duration <= 0 or not math.isfinite(duration) or rate < 0 or not math.isfinite(rate) or not pod.containers:
        return None
    end: Final = pod.timestamp.timestamp()
    return _Window(pod.metadata.name, end - duration, end, rate)


@dataclass(frozen=True, slots=True)
class CpuSample:
    start: float
    end: float
    value: float

    @property
    def at(self) -> float:
        return (self.start + self.end) / 2


@dataclass(frozen=True, slots=True)
class CpuSeries:
    windows: tuple[_Window, ...]
    pods: frozenset[str]
    start: float
    stop: float

    def _real_samples(self, start: float, end: float) -> tuple[tuple[CpuSample, ...], ...]:
        return tuple(
            _coverage(
                tuple(window for window in self.windows if window.pod == pod),
                max(start, self.start),
                min(end, self.stop),
            ).samples
            for pod in sorted(self.pods)
        )

    def coverage_between(self, start: float, end: float) -> float | None:
        """Share of the interval with real source coverage for every selected pod."""
        if not self.pods or end <= start:
            return None
        real: Final = self._real_samples(start, end)
        boundaries: Final = sorted(
            {start, end, *(at for sample in chain.from_iterable(real) for at in (sample.start, sample.end))}
        )
        covered: Final = sum(
            right - left
            for left, right in pairwise(boundaries)
            if all(any(sample.start <= left and sample.end >= right for sample in pod) for pod in real)
        )
        return covered / (end - start)

    @property
    def samples(self) -> tuple[CpuSample, ...]:
        return self.samples_between(self.start, self.stop) or ()

    def samples_between(self, start: float, end: float) -> tuple[CpuSample, ...] | None:
        coverage: Final = self.coverage_between(start, end)
        if coverage is None or coverage < _MIN_COVERAGE:
            return None
        return tuple(
            sorted(
                chain.from_iterable(_bridge(pod, start, end) for pod in self._real_samples(start, end)),
                key=lambda s: s.at,
            )
        )

    def between(self, start: float, end: float) -> tuple[float, ...] | None:
        samples: Final = self.samples_between(start, end)
        return None if samples is None else tuple(sample.value for sample in samples)


@dataclass(frozen=True, slots=True)
class _Coverage:
    end: float
    samples: tuple[CpuSample, ...] = ()

    def extend(self, window: _Window, stop: float) -> _Coverage:
        end: Final = min(window.end, stop)
        start: Final = max(self.end, window.start)
        if end <= start:
            return self
        sample: Final = CpuSample(start=start, end=end, value=window.rate * (end - start))
        return _Coverage(end, (*self.samples, sample))


def _coverage(windows: tuple[_Window, ...], start: float, stop: float) -> _Coverage:
    return reduce(
        lambda acc, window: acc.extend(window, stop), sorted(windows, key=lambda w: (w.start, w.end)), _Coverage(start)
    )


def _gap(left: CpuSample | None, right: CpuSample | None, start: float, stop: float) -> CpuSample | None:
    begin: Final = start if left is None else left.end
    end: Final = stop if right is None else right.start
    neighbours: Final = tuple(
        sample.value / (sample.end - sample.start) for sample in (left, right) if sample is not None
    )
    if end <= begin or not neighbours:
        return None
    return CpuSample(begin, end, (end - begin) * sum(neighbours) / len(neighbours))


def _bridge(samples: tuple[CpuSample, ...], start: float, stop: float) -> tuple[CpuSample, ...]:
    gaps: Final = tuple(
        gap for left, right in pairwise((None, *samples, None)) if (gap := _gap(left, right, start, stop)) is not None
    )
    return (*samples, *gaps)


def _log_uncovered(pod: str, windows: tuple[_Window, ...], start: float, stop: float) -> None:
    coverage: Final = _coverage(windows, start, stop)
    gap: Final = (
        next(
            (
                uncovered
                for left, right in pairwise((None, *coverage.samples, None))
                if (uncovered := _gap(left, right, start, stop)) is not None
            ),
            None,
        )
        if coverage.samples
        else CpuSample(start, stop, 0.0)
    )
    if gap is None:
        return
    following: Final = next(
        (window for window in sorted(windows, key=lambda w: w.start) if window.start >= gap.end), None
    )
    _log(
        "missing",
        reason="tail_not_covered" if following is None else "source_gap",
        pod=pod,
        started_at=start,
        stopped_at=stop,
        uncovered_start=gap.start,
        uncovered_end=gap.end,
        gap_seconds=gap.end - gap.start,
        next_window_start=None if following is None else following.start,
        next_window_end=None if following is None else following.end,
    )


class GatewayCpuSampler:
    """Integrate nonoverlapping source windows for every selected pod, retaining interval boundaries
    so ramp steps receive only their share of each window's CPU."""

    def __init__(
        self,
        selector: str,
        namespace: str,
        token: str,
        ca_bundle: str,
        *,
        read_metrics: Callable[[], PodMetricsList | None] | None = None,
        clock: Callable[[], float] = time.time,
        pause: Callable[[float], None] = time.sleep,
        settle_timeout_s: float = 60.0,
    ) -> None:
        self._url: Final = (
            f"https://kubernetes.default.svc/apis/metrics.k8s.io/v1beta1/namespaces/{namespace}/pods"
            f"?labelSelector={quote(selector)}"
        )
        self._headers: Final = AuthHeaders(authorization=f"Bearer {token}")
        self._ca_bundle: Final = ca_bundle
        self._stop: Final = threading.Event()
        self._thread: Final = threading.Thread(target=self._run, name="gateway-cpu-sampler", daemon=True)
        self._read: Final = read_metrics or self._read_metrics
        self._clock: Final = clock
        self._pause: Final = pause
        self._settle_timeout_s: Final = settle_timeout_s
        self._windows: tuple[_Window, ...] = ()
        self._pods: frozenset[str] | None = None
        self._population_changed: bool = False
        self._started_at: float | None = None
        self._ended_at: float | None = None

    @property
    def seconds(self) -> float | None:
        samples: Final = self._finished_samples()
        return sum(sample.value for sample in samples) if samples is not None else None

    @property
    def series(self) -> CpuSeries:
        self._finished_samples()
        return self._series()

    @property
    def coverage(self) -> float | None:
        self._finished_samples()
        series: Final = self._series()
        return series.coverage_between(series.start, series.stop)

    def _series(self) -> CpuSeries:
        if self._started_at is None or self._ended_at is None or self._population_changed:
            return CpuSeries((), frozenset(), 0.0, 0.0)
        return CpuSeries(self._windows, self._pods or frozenset(), self._started_at, self._ended_at)

    def _finished_samples(self) -> tuple[CpuSample, ...] | None:
        if self._ended_at is None:
            _log("missing", reason="measurement_not_stopped")
            return None
        self._thread.join()
        samples: Final = self._covered_samples()
        if samples is None:
            self._log_missing()
        series: Final = self._series()
        _log(
            "reading",
            cpu_seconds=None if samples is None else sum(sample.value for sample in samples),
            coverage=series.coverage_between(series.start, series.stop),
        )
        return samples

    def _log_missing(self) -> None:
        start: Final = self._started_at
        stop: Final = self._ended_at
        if start is None or stop is None or stop <= start:
            _log("missing", reason="invalid_measurement_interval", started_at=start, stopped_at=stop)
        elif self._population_changed:
            _log("missing", reason="pod_population_changed", started_at=start, stopped_at=stop)
        elif not self._pods:
            _log("missing", reason="no_pod_metrics", started_at=start, stopped_at=stop)
        else:
            for pod in sorted(self._pods):
                _log_uncovered(pod, tuple(window for window in self._windows if window.pod == pod), start, stop)

    def _read_metrics(self) -> PodMetricsList | None:
        result: Final = get_external(
            self._url, response_type=PodMetricsList, headers=self._headers, timeout=10.0, ca_bundle=self._ca_bundle
        )
        return result.data if isinstance(result, Success) else None

    def _sample(self) -> None:
        metrics: Final = self._read()
        if metrics is None:
            _log("response", state="read_failed")
            return
        if not metrics.items:
            _log("response", state="empty")
            return
        for pod in metrics.items:
            _log(
                "response",
                pod=pod.metadata.name,
                timestamp=pod.timestamp.isoformat(),
                window=pod.window,
                cpu=tuple(container.usage.cpu for container in pod.containers),
                valid=_window(pod) is not None,
            )
        pods: Final = frozenset(pod.metadata.name for pod in metrics.items)
        if self._pods is None:
            self._pods = pods
        elif pods != self._pods:
            _log("pod_population_changed", expected=tuple(sorted(self._pods)), observed=tuple(sorted(pods)))
            self._population_changed = True
        windows: Final = tuple(window for pod in metrics.items if (window := _window(pod)) is not None)
        self._windows = tuple(dict.fromkeys((*self._windows, *windows)))

    def _covered_samples(self) -> tuple[CpuSample, ...] | None:
        series: Final = self._series()
        return series.samples_between(series.start, series.stop)

    def _tail_covered(self) -> bool:
        stop: Final = self._ended_at
        return (
            stop is not None
            and bool(self._pods)
            and all(
                any(window.pod == pod and window.end >= stop for window in self._windows) for pod in self._pods or ()
            )
        )

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(_INTERVAL_S):
            self._sample()
        if self._population_changed:
            return
        deadline: Final = time.monotonic() + self._settle_timeout_s
        self._sample()
        while not self._population_changed and not self._tail_covered() and time.monotonic() < deadline:
            self._pause(min(_INTERVAL_S, max(0.0, deadline - time.monotonic())))
            self._sample()

    def start(self) -> None:
        self._started_at = self._clock()
        _log("start", started_at=self._started_at)
        self._thread.start()

    def stop(self) -> None:
        self._ended_at = self._clock()
        _log("stop", stopped_at=self._ended_at)
        self._stop.set()


class NoSampler:
    seconds: Final[None] = None
    series: Final[None] = None
    coverage: Final[None] = None

    def start(self) -> None:
        return None

    def stop(self) -> None:
        return None


def in_cluster() -> tuple[str, str, str] | None:
    token: Final = _SERVICE_ACCOUNT / "token"
    ca: Final = _SERVICE_ACCOUNT / "ca.crt"
    namespace: Final = _SERVICE_ACCOUNT / "namespace"
    if not (token.is_file() and ca.is_file() and namespace.is_file()):
        return None
    return token.read_text().strip(), str(ca), namespace.read_text().strip()


@contextmanager
def gateway_cpu_sampler() -> Generator[GatewayCpuSampler | NoSampler]:
    selector: Final = os.environ.get(SELECTOR_ENV, "")
    cluster: Final = in_cluster()
    sampler: Final = (
        GatewayCpuSampler(selector, cluster[2], cluster[0], cluster[1]) if selector and cluster else NoSampler()
    )
    sampler.start()
    try:
        yield sampler
    finally:
        sampler.stop()
