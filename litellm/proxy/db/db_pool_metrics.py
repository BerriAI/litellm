"""Bridges the Prisma query engine's connection-pool counters into Prometheus.

The query engine tracks pool occupancy and, separately, how long a query spent
waiting for a pool slot versus executing against the database. That split is the
only way to tell "the database is slow" apart from "we ran out of connections",
so it is sampled here and re-published under ``litellm_db_pool_*`` names.

Sampling is driven from the DB call path rather than from a scheduled job: an
exporter that runs on a timer stops reporting exactly when the event loop is
saturated, which is the window an operator most needs. Requests are what stall
during pool exhaustion, and they keep arriving, so a throttled sample taken
alongside real DB work stays alive through the incident.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_proxy_logger
from litellm.constants import (
    DB_POOL_METRICS_MIN_SAMPLE_INTERVAL_SECONDS,
    DB_POOL_METRICS_SAMPLE_TIMEOUT_SECONDS,
)

if TYPE_CHECKING:
    from prisma import Metrics

_MILLISECONDS_PER_SECOND: Final = 1000.0

_POOL_BUSY_KEY: Final = "prisma_pool_connections_busy"
_POOL_IDLE_KEY: Final = "prisma_pool_connections_idle"
_POOL_OPEN_KEY: Final = "prisma_pool_connections_open"
_PENDING_ACQUIRERS_KEY: Final = "prisma_client_queries_wait"
_ACQUIRE_WAIT_HISTOGRAM_KEY: Final = "prisma_client_queries_wait_histogram_ms"
_QUERY_DURATION_HISTOGRAM_KEY: Final = "prisma_datasource_queries_duration_histogram_ms"


class SupportsPoolSample(Protocol):
    """The one method this module needs from a Prisma client wrapper.

    Deliberately not ``runtime_checkable``: ``PrismaWrapper`` resolves most of
    its surface through an instance-level ``__getattr__``, which an
    ``isinstance`` check against a protocol does not see, so a runtime check
    here would reject the very object it exists to accept.
    """

    async def get_pool_sample(self) -> DBPoolSample: ...


@dataclass(frozen=True, slots=True)
class DBPoolSample:
    """One reading of the engine's pool counters.

    ``max_connections`` is derived as ``busy + idle`` rather than parsed out of
    ``DATABASE_URL``. The engine reports ``idle`` as remaining capacity, so the
    sum is the configured ``connection_limit``, and deriving it here keeps the
    database credentials out of this path entirely.
    """

    busy_connections: float
    idle_connections: float
    open_connections: float
    pending_acquirers: float
    acquire_wait_seconds_total: float
    acquire_count_total: int
    query_duration_seconds_total: float
    query_count_total: int

    @property
    def max_connections(self) -> float:
        return self.busy_connections + self.idle_connections


@dataclass(frozen=True, slots=True)
class DBPoolMetricsUpdate:
    """A sample plus the cumulative movement since the previous sample.

    The engine reports totals; Prometheus counters take increments. The deltas
    are the difference between consecutive samples, and are omitted entirely
    when the engine's totals move backwards, which happens when the query engine
    restarts and its counters reset.
    """

    sample: DBPoolSample
    pending_acquirers: float
    acquire_wait_seconds_delta: float
    acquire_count_delta: int
    query_duration_seconds_delta: float
    query_count_delta: int


def _gauge_value(metrics: Metrics, key: str) -> float:
    for gauge in metrics.gauges:
        if gauge.key == key:
            return float(gauge.value)
    return 0.0


def _histogram_seconds_and_count(metrics: Metrics, key: str) -> tuple[float, int]:
    for histogram in metrics.histograms:
        if histogram.key == key:
            return (histogram.value.sum / _MILLISECONDS_PER_SECOND, histogram.value.count)
    return (0.0, 0)


def parse_pool_sample(metrics: Metrics) -> DBPoolSample:
    acquire_wait_seconds, acquire_count = _histogram_seconds_and_count(metrics, _ACQUIRE_WAIT_HISTOGRAM_KEY)
    query_duration_seconds, query_count = _histogram_seconds_and_count(metrics, _QUERY_DURATION_HISTOGRAM_KEY)
    return DBPoolSample(
        busy_connections=_gauge_value(metrics, _POOL_BUSY_KEY),
        idle_connections=_gauge_value(metrics, _POOL_IDLE_KEY),
        open_connections=_gauge_value(metrics, _POOL_OPEN_KEY),
        pending_acquirers=_gauge_value(metrics, _PENDING_ACQUIRERS_KEY),
        acquire_wait_seconds_total=acquire_wait_seconds,
        acquire_count_total=acquire_count,
        query_duration_seconds_total=query_duration_seconds,
        query_count_total=query_count,
    )


class DBPoolMetricsSampler:
    """Throttled reader of the engine's pool counters.

    One instance per process. ``maybe_sample`` is safe to call on every DB
    operation: it returns ``None`` without touching the engine until
    ``min_interval_seconds`` has elapsed since the last reading.
    """

    def __init__(
        self,
        *,
        min_interval_seconds: float = DB_POOL_METRICS_MIN_SAMPLE_INTERVAL_SECONDS,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._min_interval_seconds: Final = min_interval_seconds
        self._monotonic: Final = monotonic
        self._last_sampled_at: float | None = None
        self._previous: DBPoolSample | None = None
        self._pending_baseline: float = 0.0

    def is_due(self) -> bool:
        """Whether enough time has passed to justify reading the engine again."""
        if self._last_sampled_at is None:
            return True
        return (self._monotonic() - self._last_sampled_at) >= self._min_interval_seconds

    async def maybe_sample(self, resolve_client: Callable[[], SupportsPoolSample | None]) -> DBPoolMetricsUpdate | None:
        """Read the engine's counters, or return ``None`` if not yet due.

        The interval is consumed as soon as an attempt starts, before the client
        is resolved, so a deployment that has no client to sample throttles the
        same as one that does rather than retrying on every database call.

        Never raises, and never blocks for long: a wedged query engine costs a
        bounded wait. A pool sample is diagnostic, and an engine that cannot
        answer one is already the subject of a louder alarm.

        The client is resolved through a callable rather than passed in so that
        resolution, which reads proxy module state, happens under the same
        throttle and the same exception guard as the read itself.
        """
        if not self.is_due():
            return None
        self._last_sampled_at = self._monotonic()

        try:
            client: Final = resolve_client()
            if client is None:
                return None
            sample: Final = await asyncio.wait_for(
                client.get_pool_sample(), timeout=DB_POOL_METRICS_SAMPLE_TIMEOUT_SECONDS
            )
        except Exception as e:  # noqa: BLE001  # a diagnostic read must not fail the database call it rides on
            verbose_proxy_logger.debug("db pool metrics sample failed: %s", e)
            return None

        update: Final = self._to_update(sample)
        self._previous = sample
        return update

    def _corrected_pending_acquirers(self, sample: DBPoolSample) -> float:
        """The engine's waiter gauge, corrected for waiters that timed out.

        ``prisma_client_queries_wait`` is decremented when a waiter acquires a
        connection but not when it gives up, so every pool timeout latches the
        gauge one higher for the life of the process. Observed directly against
        a live engine at ``connection_limit=2``: after two bursts that produced
        seven P2024s, a fully drained pool still reported seven waiters.

        Free capacity is proof that nobody is queued, so any reading taken while
        a connection is idle is exactly the accumulated latch, and subtracting it
        recovers the real depth. The baseline re-arms on every idle sample, so
        repeated exhaustion stays accurate rather than drifting further each time.
        """
        if sample.idle_connections > 0:
            self._pending_baseline = sample.pending_acquirers
            return 0.0
        return max(0.0, sample.pending_acquirers - self._pending_baseline)

    def _to_update(self, sample: DBPoolSample) -> DBPoolMetricsUpdate:
        pending: Final = self._corrected_pending_acquirers(sample)
        previous: Final = self._previous
        engine_restarted: Final = previous is not None and (
            sample.acquire_count_total < previous.acquire_count_total
            or sample.query_count_total < previous.query_count_total
        )
        if previous is None or engine_restarted:
            return DBPoolMetricsUpdate(
                sample=sample,
                pending_acquirers=pending,
                acquire_wait_seconds_delta=0.0,
                acquire_count_delta=0,
                query_duration_seconds_delta=0.0,
                query_count_delta=0,
            )
        return DBPoolMetricsUpdate(
            sample=sample,
            pending_acquirers=pending,
            acquire_wait_seconds_delta=max(
                0.0, sample.acquire_wait_seconds_total - previous.acquire_wait_seconds_total
            ),
            acquire_count_delta=sample.acquire_count_total - previous.acquire_count_total,
            query_duration_seconds_delta=max(
                0.0, sample.query_duration_seconds_total - previous.query_duration_seconds_total
            ),
            query_count_delta=sample.query_count_total - previous.query_count_total,
        )
