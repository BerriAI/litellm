"""
Unit tests for PrismaMetricsCollector.

All Prometheus metrics are isolated per test using a custom CollectorRegistry
to avoid cross-test registration conflicts.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import CollectorRegistry

sys.path.insert(
    0, os.path.abspath("../../../..")
)  # Adds the parent directory to the system path

import litellm
from litellm.proxy.db.prisma_metrics_collector import (
    PrismaMetricsCollector,
    _DEFAULT_COLLECTION_INTERVAL,
    _MIN_COLLECTION_INTERVAL,
)


def _make_prisma_client():
    """Create a mock PrismaClient with the interface PrismaMetricsCollector uses."""
    client = MagicMock()
    client.db = MagicMock()
    client.db.query_raw = AsyncMock(return_value=[])
    client._is_engine_alive = MagicMock(return_value=True)
    return client


def _make_collector(prisma_client=None, collection_interval=None, registry=None):
    """Create a PrismaMetricsCollector with an isolated Prometheus registry.

    Patches the module-level helper functions to use the provided registry,
    so every test gets its own metric instances.
    """
    if prisma_client is None:
        prisma_client = _make_prisma_client()
    if registry is None:
        registry = CollectorRegistry()

    from prometheus_client import Counter, Gauge

    def _patched_get_or_create_gauge(name, description, labelnames=None):
        if labelnames:
            return Gauge(name, description, labelnames=labelnames, registry=registry)
        return Gauge(name, description, registry=registry)

    def _patched_get_or_create_counter(name, description):
        return Counter(name, description, registry=registry)

    with patch(
        "litellm.proxy.db.prisma_metrics_collector._get_or_create_gauge",
        side_effect=_patched_get_or_create_gauge,
    ), patch(
        "litellm.proxy.db.prisma_metrics_collector._get_or_create_counter",
        side_effect=_patched_get_or_create_counter,
    ):
        collector = PrismaMetricsCollector(
            prisma_client=prisma_client,
            collection_interval=collection_interval,
        )

    return collector, registry


# ---------------------------------------------------------------------------
# Metric creation
# ---------------------------------------------------------------------------


def test_collector_creates_prometheus_metrics():
    """Verify all 4 metrics (pool connections gauge, lock waiting gauge, engine_up gauge, restarts counter) are created."""
    collector, registry = _make_collector()

    assert collector._pool_connections is not None
    assert collector._pool_waiting is not None
    assert collector._engine_up is not None
    assert collector._engine_restarts is not None

    # Verify names via the registry
    metric_names = {m.name for m in registry.collect()}
    expected = {
        "litellm_db_pool_connections",
        "litellm_db_pool_lock_waiting_connections",
        "litellm_db_engine_up",
        "litellm_db_engine_restarts",  # counter exposes _total suffix but name is base
    }
    assert expected.issubset(
        metric_names
    ), f"Missing metrics: {expected - metric_names}"


# ---------------------------------------------------------------------------
# Pool metrics collection
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_collect_pool_metrics_sets_gauges():
    """Mock query_raw to return pool stats grouped by state and verify labeled gauge is set."""
    client = _make_prisma_client()

    pool_rows = [
        {"state": "active", "count": 5},
        {"state": "idle", "count": 10},
        {"state": "idle in transaction", "count": 3},
    ]
    lock_rows = [{"waiting": 2}]
    client.db.query_raw = AsyncMock(side_effect=[pool_rows, lock_rows])
    collector, registry = _make_collector(prisma_client=client)

    await collector._collect_pool_metrics()

    assert (
        registry.get_sample_value("litellm_db_pool_connections", {"state": "active"})
        == 5
    )
    assert (
        registry.get_sample_value("litellm_db_pool_connections", {"state": "idle"})
        == 10
    )
    assert (
        registry.get_sample_value(
            "litellm_db_pool_connections", {"state": "idle in transaction"}
        )
        == 3
    )
    assert registry.get_sample_value("litellm_db_pool_lock_waiting_connections") == 2


@pytest.mark.asyncio
async def test_collect_pool_metrics_handles_empty_result():
    """When query_raw returns empty lists, no crash should occur."""
    client = _make_prisma_client()
    client.db.query_raw = AsyncMock(return_value=[])
    collector, registry = _make_collector(prisma_client=client)

    await collector._collect_pool_metrics()

    # No labeled values should be set — just verify no crash
    assert (
        registry.get_sample_value("litellm_db_pool_connections", {"state": "active"})
        is None
    )


@pytest.mark.asyncio
async def test_collect_pool_metrics_handles_null_state():
    """When pg_stat_activity returns a NULL state, it should be mapped to 'unknown'."""
    client = _make_prisma_client()
    pool_rows = [{"state": None, "count": 1}]
    lock_rows = [{"waiting": 0}]
    client.db.query_raw = AsyncMock(side_effect=[pool_rows, lock_rows])
    collector, registry = _make_collector(prisma_client=client)

    await collector._collect_pool_metrics()

    assert (
        registry.get_sample_value("litellm_db_pool_connections", {"state": "unknown"})
        == 1
    )


@pytest.mark.asyncio
async def test_collect_pool_metrics_handles_query_error():
    """When query_raw raises an exception, the collector should log a warning and not crash."""
    client = _make_prisma_client()
    client.db.query_raw = AsyncMock(side_effect=RuntimeError("connection lost"))
    collector, _ = _make_collector(prisma_client=client)

    with patch(
        "litellm.proxy.db.prisma_metrics_collector.verbose_proxy_logger"
    ) as mock_logger:
        await collector._collect_pool_metrics()
        mock_logger.warning.assert_called_once()
        assert "connection lost" in str(mock_logger.warning.call_args)


# ---------------------------------------------------------------------------
# Engine health
# ---------------------------------------------------------------------------


def test_collect_engine_health_alive():
    """When engine is alive, engine_up gauge should be 1."""
    client = _make_prisma_client()
    client._is_engine_alive = MagicMock(return_value=True)
    collector, registry = _make_collector(prisma_client=client)

    collector._collect_engine_health()

    assert registry.get_sample_value("litellm_db_engine_up") == 1


def test_collect_engine_health_dead():
    """When engine is dead, engine_up gauge should be 0."""
    client = _make_prisma_client()
    client._is_engine_alive = MagicMock(return_value=False)
    collector, registry = _make_collector(prisma_client=client)

    collector._collect_engine_health()

    assert registry.get_sample_value("litellm_db_engine_up") == 0


# ---------------------------------------------------------------------------
# Engine restart counter
# ---------------------------------------------------------------------------


def test_increment_engine_restarts():
    """Calling increment_engine_restarts N times should result in counter value N."""
    collector, registry = _make_collector()

    for _ in range(7):
        collector.increment_engine_restarts()

    assert registry.get_sample_value("litellm_db_engine_restarts_total") == 7


# ---------------------------------------------------------------------------
# should_enable
# ---------------------------------------------------------------------------


def test_should_enable_true():
    """should_enable() returns True when prometheus_system is in service_callback."""
    original = litellm.service_callback
    try:
        litellm.service_callback = ["prometheus_system"]
        assert PrismaMetricsCollector.should_enable() is True
    finally:
        litellm.service_callback = original


def test_should_enable_false():
    """should_enable() returns False when service_callback is empty."""
    original = litellm.service_callback
    try:
        litellm.service_callback = []
        assert PrismaMetricsCollector.should_enable() is False
    finally:
        litellm.service_callback = original


# ---------------------------------------------------------------------------
# Collection interval configuration
# ---------------------------------------------------------------------------


def test_collection_interval_from_env():
    """Interval should be read from PRISMA_METRICS_COLLECTION_INTERVAL_SECONDS env var."""
    with patch.dict(os.environ, {"PRISMA_METRICS_COLLECTION_INTERVAL_SECONDS": "60"}):
        collector, _ = _make_collector()
        assert collector._interval == 60


def test_collection_interval_minimum_enforced():
    """Interval below the minimum should be clamped to _MIN_COLLECTION_INTERVAL."""
    with patch.dict(os.environ, {"PRISMA_METRICS_COLLECTION_INTERVAL_SECONDS": "1"}):
        collector, _ = _make_collector()
        assert collector._interval == _MIN_COLLECTION_INTERVAL


def test_collection_interval_constructor_override():
    """Explicit collection_interval parameter should take precedence over env."""
    with patch.dict(os.environ, {"PRISMA_METRICS_COLLECTION_INTERVAL_SECONDS": "999"}):
        collector, _ = _make_collector(collection_interval=45)
        assert collector._interval == 45


def test_collection_interval_default():
    """Without env var or constructor arg, the default interval is used."""
    with patch.dict(os.environ, {}, clear=False):
        # Remove the env var if present
        env_copy = os.environ.copy()
        env_copy.pop("PRISMA_METRICS_COLLECTION_INTERVAL_SECONDS", None)
        with patch.dict(os.environ, env_copy, clear=True):
            collector, _ = _make_collector()
            assert collector._interval == _DEFAULT_COLLECTION_INTERVAL


# ---------------------------------------------------------------------------
# Start / Stop lifecycle
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_start_creates_task():
    """Calling start() should create a background asyncio task."""
    collector, _ = _make_collector()

    collector.start()
    assert collector._task is not None
    # Clean up
    await collector.stop()


@pytest.mark.asyncio
async def test_start_idempotent():
    """Calling start() twice should not create a second task."""
    collector, _ = _make_collector()

    collector.start()
    first_task = collector._task
    collector.start()
    assert collector._task is first_task
    # Clean up
    await collector.stop()


@pytest.mark.asyncio
async def test_stop_cancels_task():
    """Calling stop() after start() should cancel the task and set it to None."""
    collector, _ = _make_collector()

    collector.start()
    assert collector._task is not None

    await collector.stop()
    assert collector._task is None
