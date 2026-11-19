"""Estimate gateway CPU from PodMetrics source windows, clipped to the measured pass. Polling a stale
response adds no CPU. Reading results waits up to a minute for windows covering the pass's end. Incomplete coverage
is missing, not zero. The estimate assumes uniform CPU within each source window, including at pass and
ramp boundaries; only cumulative counters could measure short bursts exactly. The runner's read-only
service account cannot exec into gateway pods or read kubelet counters. Whole-second source timestamps
allow apparent gaps below one second, estimated with the next window's average."""

from __future__ import annotations

import math
import os
import re
import threading
import time
from collections.abc import Callable, Generator
from contextlib import contextmanager
from dataclasses import dataclass
from functools import reduce
from itertools import chain
from pathlib import Path
from typing import Final
from urllib.parse import quote

from e2e_http import AuthHeaders, Success, get_external
from pydantic import AwareDatetime, BaseModel, ConfigDict

SELECTOR_ENV: Final = "E2E_PERF_GATEWAY_POD_SELECTOR"
_SERVICE_ACCOUNT: Final = Path("/var/run/secrets/kubernetes.io/serviceaccount")
_INTERVAL_S: Final = 5.0
_SOURCE_TIME_RESOLUTION_S: Final = 1.0
_SUFFIX_CORES: Final = {"n": 1e-9, "u": 1e-6, "m": 1e-3}
_DURATION_PART: Final = re.compile(r"(\d+(?:\.\d+)?)(ns|us|µs|ms|s|m|h)")
_DURATION_UNITS: Final = {"ns": 1e-9, "us": 1e-6, "µs": 1e-6, "ms": 1e-3, "s": 1.0, "m": 60.0, "h": 3600.0}


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
    samples: tuple[CpuSample, ...]

    def between(self, start: float, end: float) -> tuple[float, ...] | None:
        if not self.samples:
            return None
        return tuple(
            sample.value * (min(end, sample.end) - max(start, sample.start)) / (sample.end - sample.start)
            for sample in self.samples
            if min(end, sample.end) > max(start, sample.start)
        )


@dataclass(frozen=True, slots=True)
class _Coverage:
    end: float
    samples: tuple[CpuSample, ...] = ()

    def extend(self, window: _Window, stop: float) -> _Coverage:
        end: Final = min(window.end, stop)
        if window.start - self.end >= _SOURCE_TIME_RESOLUTION_S or end <= self.end:
            return self
        sample: Final = CpuSample(start=self.end, end=end, value=window.rate * (end - self.end))
        return _Coverage(end, (*self.samples, sample))


def _integrate(windows: tuple[_Window, ...], start: float, stop: float) -> tuple[CpuSample, ...] | None:
    coverage: Final = reduce(
        lambda acc, window: acc.extend(window, stop), sorted(windows, key=lambda w: w.end), _Coverage(start)
    )
    return coverage.samples if coverage.end >= stop else None


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
        return CpuSeries(samples=self._finished_samples() or ())

    def _finished_samples(self) -> tuple[CpuSample, ...] | None:
        if self._ended_at is None:
            return None
        self._thread.join()
        return self._covered_samples()

    def _read_metrics(self) -> PodMetricsList | None:
        result: Final = get_external(
            self._url, response_type=PodMetricsList, headers=self._headers, timeout=10.0, ca_bundle=self._ca_bundle
        )
        return result.data if isinstance(result, Success) else None

    def _sample(self) -> None:
        metrics: Final = self._read()
        if metrics is None or not metrics.items:
            return
        pods: Final = frozenset(pod.metadata.name for pod in metrics.items)
        if self._pods is None:
            self._pods = pods
        elif pods != self._pods:
            self._population_changed = True
        windows: Final = tuple(window for pod in metrics.items if (window := _window(pod)) is not None)
        self._windows = tuple(dict.fromkeys((*self._windows, *windows)))

    def _covered_samples(self) -> tuple[CpuSample, ...] | None:
        start: Final = self._started_at
        stop: Final = self._ended_at
        if start is None or stop is None or stop <= start or not self._pods or self._population_changed:
            return None
        coverages: Final = tuple(
            _integrate(tuple(window for window in self._windows if window.pod == pod), start, stop)
            for pod in self._pods
        )
        if any(coverage is None for coverage in coverages):
            return None
        return tuple(
            sorted(chain.from_iterable(coverage for coverage in coverages if coverage is not None), key=lambda s: s.at)
        )

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(_INTERVAL_S):
            self._sample()
        if self._population_changed:
            return
        deadline: Final = time.monotonic() + self._settle_timeout_s
        self._sample()
        while not self._population_changed and self._covered_samples() is None and time.monotonic() < deadline:
            self._pause(min(_INTERVAL_S, max(0.0, deadline - time.monotonic())))
            self._sample()

    def start(self) -> None:
        self._started_at = self._clock()
        self._thread.start()

    def stop(self) -> None:
        self._ended_at = self._clock()
        self._stop.set()


class NoSampler:
    seconds: Final[None] = None
    series: Final[None] = None

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
