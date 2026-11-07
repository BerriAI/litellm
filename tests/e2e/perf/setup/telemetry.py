"""The gateway's own `/metrics`, read before and after a run (and every few seconds during a ramped run, see
`telemetry_sampler`) for the numbers nothing outside the gateway can see. Today that is `litellm_redis_latency`, the histogram customers' Grafana panels show when Redis
slows the gateway down. The gateway's latency and request counts are read per request from its response
headers instead (`setup/wire.py`), which gives percentiles where a histogram delta only gives a mean.

RSS and CPU are not here: a multi-worker gateway serves `/metrics` through the prometheus multiprocess
collector, which drops `process_resident_memory_bytes` and `process_cpu_seconds_total`, so those stay
on psutil (`proxy_usage.py`)."""

from __future__ import annotations

import math
import threading
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Final

from e2e_http import NoBody
from prometheus_client.parser import text_string_to_metric_families
from prometheus_client.samples import Sample
from proxy_client import ProxyClient

from setup.series import Recorder, Series

_SAMPLE_SECONDS: Final = 10.0


@dataclass(frozen=True, slots=True)
class HistogramTotals:
    """`_sum` and `_count` of one histogram across every label set, enough for a mean over a window."""

    total: float
    count: float

    def minus(self, earlier: HistogramTotals) -> HistogramTotals:
        return HistogramTotals(total=self.total - earlier.total, count=self.count - earlier.count)

    @property
    def mean(self) -> float | None:
        return self.total / self.count if self.count > 0 else None


@dataclass(frozen=True, slots=True)
class GatewayTelemetry:
    redis_latency: HistogramTotals
    """`litellm_redis_latency`, seconds per Redis call the gateway made."""

    def minus(self, earlier: GatewayTelemetry) -> GatewayTelemetry:
        return GatewayTelemetry(redis_latency=self.redis_latency.minus(earlier.redis_latency))


def _value(samples: Iterable[Sample], name: str) -> float:
    return sum(sample.value for sample in samples if sample.name == name and math.isfinite(sample.value))


def _histogram(samples: tuple[Sample, ...], family: str) -> HistogramTotals:
    return HistogramTotals(total=_value(samples, f"{family}_sum"), count=_value(samples, f"{family}_count"))


def parse_gateway_telemetry(text: str) -> GatewayTelemetry:
    samples: Final = tuple(chain.from_iterable(family.samples for family in text_string_to_metric_families(text)))
    return GatewayTelemetry(redis_latency=_histogram(samples, "litellm_redis_latency"))


def read_gateway_telemetry(proxy: ProxyClient) -> GatewayTelemetry | None:
    """One `/metrics` scrape, or None when the gateway does not serve it (no `prometheus` callback), in which
    case every limit on a gateway-reported metric fails as missing rather than reading a silent zero."""
    scrape: Final = proxy.probe("/metrics", params=NoBody())
    return parse_gateway_telemetry(scrape.body) if scrape.status_code == 200 else None


class TelemetrySampler:
    """`/metrics` every `_SAMPLE_SECONDS` on a background thread while traffic runs, so a ramped run can read
    the Redis latency of each step's window as the difference between the scrapes around it."""

    def __init__(self, proxy: ProxyClient) -> None:
        self._proxy: Final = proxy
        self._stop: Final = threading.Event()
        self._thread: Final = threading.Thread(target=self._run, name="gateway-telemetry", daemon=True)
        self._scrapes: Final[Recorder[GatewayTelemetry]] = Recorder()

    @property
    def series(self) -> Series[GatewayTelemetry]:
        return self._scrapes.series

    def _run(self) -> None:
        while not self._stop.is_set():
            scrape: GatewayTelemetry | None = read_gateway_telemetry(self._proxy)
            if scrape is not None:
                self._scrapes.record(scrape)
            _ = self._stop.wait(_SAMPLE_SECONDS)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=30)


@contextmanager
def telemetry_sampler(proxy: ProxyClient) -> Generator[TelemetrySampler]:
    sampler: Final = TelemetrySampler(proxy)
    sampler.start()
    try:
        yield sampler
    finally:
        sampler.stop()
