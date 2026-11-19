"""Per-worker RSS readings of the proxy through /debug/memory/summary.

One read goes to every configured replica (PROXY_REPLICA_URLS) under the master
key and answers from whichever worker behind that address took the connection;
the release stack runs one worker per gateway replica, so a read per replica is
a read per worker. A reading keys its worker by replica address, hostname, and
pid, since pods in their own pid namespaces report the same pids. A replica that
gives no reading (unreachable, a non-2xx, or a summary without ram_usage_mb) is
kept as a failure reason rather than dropped, so a test can fail on it by name
instead of passing on the replicas that did answer.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final

from e2e_config import MEMORY_RSS_SAMPLE_INTERVAL_SECONDS, MEMORY_RSS_SETTLE_SAMPLES
from e2e_http import Result, Success
from models import MemorySummaryResponse
from proxy_client import ProxyClient

WorkerKey = tuple[str, str | None, int]


@dataclass(frozen=True, slots=True)
class RssReading:
    replica: str
    hostname: str | None
    worker_pid: int
    ram_usage_mb: float

    @property
    def worker(self) -> WorkerKey:
        return (self.replica, self.hostname, self.worker_pid)

    @property
    def where(self) -> str:
        return f"worker pid {self.worker_pid} on {self.hostname or 'an unnamed host'} behind {self.replica}"


@dataclass(frozen=True, slots=True)
class RssCapture:
    readings: tuple[RssReading, ...]
    failures: tuple[str, ...]

    @property
    def heaviest(self) -> RssReading | None:
        return max(self.readings, key=lambda reading: reading.ram_usage_mb, default=None)

    @property
    def junit_properties(self) -> tuple[tuple[str, object], ...]:
        heaviest: Final = self.heaviest
        if heaviest is None:
            return ()
        return (("idle_rss_heaviest_mb", heaviest.ram_usage_mb), ("idle_rss_heaviest_worker", heaviest.where))


def _outcome(replica: str, result: Result[MemorySummaryResponse]) -> RssReading | str:
    match result:
        case Success(data=body) if body.memory.ram_usage_mb is not None:
            return RssReading(replica, body.hostname, body.worker_pid, body.memory.ram_usage_mb)
        case Success(data=body):
            return f"{replica} answered /debug/memory/summary without ram_usage_mb: {body.memory.error}"
        case _:
            return f"{replica} gave no /debug/memory/summary reading: {result}"


def rss_capture(summaries: Mapping[str, Result[MemorySummaryResponse]]) -> RssCapture:
    outcomes: Final = tuple(_outcome(replica, result) for replica, result in summaries.items())
    return RssCapture(
        readings=tuple(outcome for outcome in outcomes if isinstance(outcome, RssReading)),
        failures=tuple(outcome for outcome in outcomes if isinstance(outcome, str)),
    )


def read_rss_everywhere(proxy: ProxyClient, *, timeout: float | None = None) -> RssCapture:
    return rss_capture(proxy.memory_summary_everywhere(timeout=timeout))


RSS_SAMPLE_CAP: Final = 4 * MEMORY_RSS_SETTLE_SAMPLES


def _read_rss_everywhere_after_pause(proxy: ProxyClient) -> tuple[RssReading, ...]:
    time.sleep(MEMORY_RSS_SAMPLE_INTERVAL_SECONDS)
    capture: Final = read_rss_everywhere(proxy)
    assert not capture.failures, (
        f"{len(capture.failures)} replica(s) gave no RSS reading mid-checkpoint, so their workers cannot be "
        f"compared with themselves: {'; '.join(capture.failures)}"
    )
    return capture.readings


def _readings_until_no_new_worker(
    proxy: ProxyClient, readings: tuple[RssReading, ...], samples: int, samples_since_new_worker: int
) -> tuple[RssReading, ...]:
    if samples >= RSS_SAMPLE_CAP or samples_since_new_worker >= MEMORY_RSS_SETTLE_SAMPLES:
        return readings
    sample: Final = _read_rss_everywhere_after_pause(proxy)
    known: Final = frozenset(reading.worker for reading in readings)
    new_worker_answered: Final = any(reading.worker not in known for reading in sample)
    return _readings_until_no_new_worker(
        proxy, readings + sample, samples + 1, 0 if new_worker_answered else samples_since_new_worker + 1
    )


def settled_rss_per_worker(proxy: ProxyClient) -> Mapping[WorkerKey, RssReading]:
    """The lowest reading per worker, sampled until no new worker has answered for a settle window: a burst of
    traffic leaves a transient bulge of garbage that gc reclaims within seconds, and a load-balanced address
    reaches the workers behind it one answer at a time."""
    readings: Final = _readings_until_no_new_worker(proxy, (), 0, 0)
    assert readings, "no /debug/memory/summary read carried ram_usage_mb, so the proxy cannot report its RSS"
    return MappingProxyType(
        {
            worker: min((reading for reading in readings if reading.worker == worker), key=lambda r: r.ram_usage_mb)
            for worker in {reading.worker for reading in readings}
        }
    )
