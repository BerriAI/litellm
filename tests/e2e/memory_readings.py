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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

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
