"""
Collects Prisma/PostgreSQL connection pool and engine health metrics
and exposes them as Prometheus gauges/counters.
"""

import asyncio
import os
from typing import Optional

from prometheus_client import REGISTRY, Counter, Gauge

import litellm
from litellm._logging import verbose_proxy_logger


def _is_metric_registered(metric_name: str) -> bool:
    """Check if a Prometheus metric is already registered (O(1) lookup)."""
    names_to_collectors = getattr(REGISTRY, "_names_to_collectors", None)
    if names_to_collectors is not None:
        return metric_name in names_to_collectors
    for metric in REGISTRY.collect():
        if metric_name == metric.name:
            return True
    return False


def _get_or_create_gauge(
    name: str, description: str, labelnames: Optional[list] = None
) -> Gauge:
    if _is_metric_registered(name):
        names_to_collectors = getattr(REGISTRY, "_names_to_collectors", None)
        if names_to_collectors is not None and name in names_to_collectors:
            return names_to_collectors[name]
    if labelnames:
        return Gauge(name, description, labelnames=labelnames)
    return Gauge(name, description)


def _get_or_create_counter(name: str, description: str) -> Counter:
    if _is_metric_registered(name):
        names_to_collectors = getattr(REGISTRY, "_names_to_collectors", None)
        if names_to_collectors is not None and name in names_to_collectors:
            return names_to_collectors[name]
    return Counter(name, description)


_POOL_METRICS_SQL = """
SELECT state, count(*) as count
FROM pg_stat_activity
WHERE pid != pg_backend_pid() AND datname = current_database() AND usename = current_user
GROUP BY state
"""

_LOCK_WAITING_SQL = """
SELECT count(*) as waiting
FROM pg_stat_activity
WHERE pid != pg_backend_pid() AND datname = current_database() AND usename = current_user
  AND wait_event_type = 'Lock'
"""

_MIN_COLLECTION_INTERVAL = 5
_DEFAULT_COLLECTION_INTERVAL = 30


class PrismaMetricsCollector:
    """Periodically collects DB pool and engine health metrics for Prometheus."""

    def __init__(
        self,
        prisma_client: "litellm.proxy.utils.PrismaClient",  # type: ignore[name-defined]
        collection_interval: Optional[float] = None,
    ) -> None:
        self.prisma_client = prisma_client

        if collection_interval is not None:
            self._interval = max(collection_interval, _MIN_COLLECTION_INTERVAL)
        else:
            raw = os.environ.get(
                "PRISMA_METRICS_COLLECTION_INTERVAL_SECONDS",
                str(_DEFAULT_COLLECTION_INTERVAL),
            )
            self._interval = max(float(raw), _MIN_COLLECTION_INTERVAL)

        self._task: Optional[asyncio.Task] = None

        # Prometheus metrics
        self._pool_connections = _get_or_create_gauge(
            "litellm_db_pool_connections",
            "Number of DB connections by state",
            labelnames=["state"],
        )
        self._pool_waiting = _get_or_create_gauge(
            "litellm_db_pool_lock_waiting_connections",
            "Number of connections blocked on row/table locks in the DB pool",
        )
        self._engine_up = _get_or_create_gauge(
            "litellm_db_engine_up",
            "Whether the Prisma query engine process is alive (1=up, 0=down)",
        )
        self._engine_restarts = _get_or_create_counter(
            "litellm_db_engine_restarts_total",
            "Total number of Prisma query engine restarts",
        )

    def start(self) -> None:
        """Start the background collection loop. No-op if already running."""
        if self._task is not None:
            return
        self._task = asyncio.create_task(self._collection_loop())
        verbose_proxy_logger.info(
            "Started PrismaMetricsCollector (interval=%ss)", self._interval
        )

    async def stop(self) -> None:
        """Stop the background collection loop."""
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None
        verbose_proxy_logger.info("Stopped PrismaMetricsCollector")

    async def _collection_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(self._interval)
                await self._collect_pool_metrics()
                self._collect_engine_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                verbose_proxy_logger.warning("PrismaMetricsCollector loop error: %s", e)

    async def _collect_pool_metrics(self) -> None:
        try:
            rows = await self.prisma_client.db.query_raw(_POOL_METRICS_SQL)
            for row in rows:
                state = row.get("state") or "unknown"
                self._pool_connections.labels(state=state).set(row.get("count", 0))

            lock_rows = await self.prisma_client.db.query_raw(_LOCK_WAITING_SQL)
            if lock_rows:
                self._pool_waiting.set(lock_rows[0].get("waiting", 0))
        except Exception as e:
            verbose_proxy_logger.warning(
                "PrismaMetricsCollector failed to collect pool metrics: %s", e
            )

    def _collect_engine_health(self) -> None:
        alive = self.prisma_client._is_engine_alive()
        self._engine_up.set(1 if alive else 0)

    def increment_engine_restarts(self) -> None:
        """Increment the engine restart counter. Call from attempt_db_reconnect()."""
        self._engine_restarts.inc()

    @staticmethod
    def should_enable() -> bool:
        """Check if Prometheus system metrics are enabled."""
        return "prometheus_system" in litellm.service_callback
