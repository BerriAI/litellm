"""Resident memory and CPU of the proxy process tree, sampled on a background thread.

The proxy under load runs several worker processes, and `/metrics` cannot report their
memory: litellm sets PROMETHEUS_MULTIPROC_DIR when num_workers > 1, and the multiprocess
collector drops the process collector's `process_resident_memory_bytes` /
`process_cpu_seconds_total` entirely. So the test measures the tree itself through psutil,
which needs the proxy to run on the same host as the test.
"""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Final

import psutil
from pydantic import BaseModel, ConfigDict


class _MemoryInfo(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    rss: int


@dataclass(frozen=True, slots=True)
class UsageSample:
    elapsed_seconds: float
    rss_bytes: int
    cpu_seconds: float


@dataclass(frozen=True, slots=True)
class UsageWindow:
    """The samples taken across one phase, plus what they say about that phase."""

    samples: tuple[UsageSample, ...]

    def rss_percentile(self, fraction: float) -> int:
        if not self.samples:
            return 0
        ordered: Final = sorted(sample.rss_bytes for sample in self.samples)
        return ordered[_rank(len(ordered), fraction)]

    def cpu_seconds_consumed(self) -> float:
        """CPU seconds the tree burned across the window, from its monotonic counter."""
        if len(self.samples) < 2:
            return 0.0
        return self.samples[-1].cpu_seconds - self.samples[0].cpu_seconds

    def cpu_seconds_per_request(self, requests: int) -> float:
        """CPU seconds the tree spent per request served.

        The portable cost figure: cores-busy saturates at the worker count under enough load,
        so it reads the same whether a request costs 10 ms of CPU or 40 ms. This does not.
        """
        return self.cpu_seconds_consumed() / requests if requests else 0.0

    def cpu_utilization_percentiles(self) -> tuple[float, float, float]:
        """Per-interval CPU utilization (cores busy) at p50, p90 and p99.

        Derived from consecutive samples of the cumulative counter rather than
        psutil's own cpu_percent, so it covers every process in the tree including
        workers that came and went between samples.
        """
        rates: Final = sorted(
            (later.cpu_seconds - earlier.cpu_seconds) / (later.elapsed_seconds - earlier.elapsed_seconds)
            for earlier, later in zip(self.samples, self.samples[1:])
            if later.elapsed_seconds > earlier.elapsed_seconds
        )
        if not rates:
            return 0.0, 0.0, 0.0
        return (
            rates[_rank(len(rates), 0.5)],
            rates[_rank(len(rates), 0.9)],
            rates[_rank(len(rates), 0.99)],
        )

    def summary(self) -> str:
        p50_cpu, p90_cpu, p99_cpu = self.cpu_utilization_percentiles()
        return (
            f"RSS p50 {self.rss_percentile(0.5) / 2**20:.0f} MB, "
            f"p90 {self.rss_percentile(0.9) / 2**20:.0f} MB, "
            f"p99 {self.rss_percentile(0.99) / 2**20:.0f} MB; "
            f"CPU cores busy p50 {p50_cpu:.2f}, p90 {p90_cpu:.2f}, p99 {p99_cpu:.2f}; "
            f"{self.cpu_seconds_consumed():.1f} CPU seconds consumed"
        )


def _rank(count: int, fraction: float) -> int:
    """Index of the sample at `fraction`, the same lower-sample convention as locust's percentiles."""
    return min(count - 1, max(0, math.ceil(count * fraction) - 1))


def _read_process(process: psutil.Process) -> tuple[int, float] | None:
    try:
        with process.oneshot():
            memory: Final = _MemoryInfo.model_validate(process.memory_info())
            times: Final = process.cpu_times()
            return memory.rss, times.user + times.system
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None


class ProxyUsageSampler:
    """Samples the proxy process tree every `interval_seconds` until stopped.

    `split()` returns the samples taken so far and starts a new window, so one sampler
    covers a baseline phase and a chaos phase without a gap between them.
    """

    def __init__(self, pid: int, interval_seconds: float = 1.0) -> None:
        self._process: Final = psutil.Process(pid)
        self._interval: Final = interval_seconds
        self._stop: Final = threading.Event()
        self._lock: Final = threading.Lock()
        self._samples: list[UsageSample] = []  # mutable-ok: a sampling buffer the reader drains under a lock
        self._started: Final = time.monotonic()
        self._thread: Final = threading.Thread(target=self._run, name="proxy-usage-sampler", daemon=True)

    def __enter__(self) -> ProxyUsageSampler:
        self._thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self._thread.join(timeout=self._interval * 5)

    def _tree(self) -> tuple[psutil.Process, ...]:
        try:
            return (self._process, *self._process.children(recursive=True))
        except psutil.NoSuchProcess:
            return ()

    def _sample(self) -> UsageSample | None:
        readings: Final = tuple(reading for process in self._tree() if (reading := _read_process(process)) is not None)
        if not readings:
            return None
        return UsageSample(
            elapsed_seconds=time.monotonic() - self._started,
            rss_bytes=sum(rss for rss, _ in readings),
            cpu_seconds=sum(cpu for _, cpu in readings),
        )

    def _run(self) -> None:
        while not self._stop.is_set():
            sample = self._sample()
            if sample is not None:
                with self._lock:
                    self._samples.append(sample)
            self._stop.wait(self._interval)

    def split(self) -> UsageWindow:
        """The window that ends now; the next one starts from this window's last sample.

        The boundary sample is carried into the next window so its CPU counter has a
        starting point, which is what makes the two windows' utilization comparable.
        """
        with self._lock:
            taken = tuple(self._samples)
            self._samples = [taken[-1]] if taken else []  # rebind-ok: drains the buffer under the lock
        return UsageWindow(samples=taken)
