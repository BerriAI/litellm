"""Prometheus surface for the Prisma connection-pool saturation metrics."""

from typing import get_args
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY
from prisma.errors import DataError

from litellm.proxy.db.db_pool_metrics import DBPoolMetricsUpdate, DBPoolSample
from litellm.types.integrations.prometheus import (
    DEFINED_PROMETHEUS_METRICS,
    PrometheusMetricLabels,
)

DB_POOL_METRICS = (
    "litellm_db_pool_connections_max",
    "litellm_db_pool_connections_busy",
    "litellm_db_pool_connections_idle",
    "litellm_db_pool_connections_open",
    "litellm_db_pool_pending_acquirers",
    "litellm_db_pool_acquire_wait_seconds_total",
    "litellm_db_pool_acquire_total",
    "litellm_db_query_duration_seconds_total",
    "litellm_db_query_total",
    "litellm_db_pool_timeouts_total",
)


@pytest.mark.parametrize("metric", DB_POOL_METRICS)
def test_metric_is_registered_so_the_config_and_exclude_lists_accept_it(metric):
    assert metric in get_args(DEFINED_PROMETHEUS_METRICS)
    # Asserted on the class attribute rather than get_labels(), which also folds
    # in whatever custom metadata labels and tags happen to be configured
    # process-wide and so depends on global state another test may have set.
    assert getattr(PrometheusMetricLabels, metric) == (), (
        f"{metric} must stay unlabelled: the pool is a per-worker resource and the ticket "
        "forbids api-key/team labels on saturation metrics"
    )


def _update(**kwargs) -> DBPoolMetricsUpdate:
    sample = DBPoolSample(
        busy_connections=kwargs.pop("busy", 0.0),
        idle_connections=kwargs.pop("idle", 0.0),
        open_connections=kwargs.pop("open_", 0.0),
        pending_acquirers=kwargs.pop("pending", 0.0),
        acquire_wait_seconds_total=0.0,
        acquire_count_total=0,
        query_duration_seconds_total=0.0,
        query_count_total=0,
    )
    return DBPoolMetricsUpdate(
        sample=sample,
        pending_acquirers=sample.pending_acquirers,
        acquire_wait_seconds_delta=kwargs.pop("wait_delta", 0.0),
        acquire_count_delta=kwargs.pop("acquire_delta", 0),
        query_duration_seconds_delta=kwargs.pop("query_seconds_delta", 0.0),
        query_count_delta=kwargs.pop("query_delta", 0),
    )


def _clear_registry() -> None:
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


@pytest.fixture
def logger():
    from litellm.integrations.prometheus import PrometheusLogger

    _clear_registry()
    yield PrometheusLogger()
    _clear_registry()


def _value(name: str) -> float:
    return REGISTRY.get_sample_value(name) or 0.0


def test_a_saturated_pool_is_visible_in_the_gauges(logger):
    logger.record_db_pool_sample(_update(busy=5.0, idle=0.0, open_=5.0, pending=3.0))

    assert _value("litellm_db_pool_connections_busy") == 5.0
    assert _value("litellm_db_pool_connections_idle") == 0.0
    assert _value("litellm_db_pool_connections_open") == 5.0
    assert _value("litellm_db_pool_pending_acquirers") == 3.0
    assert _value("litellm_db_pool_connections_max") == 5.0, "busy + idle is the configured limit"


def test_pool_wait_and_query_execution_are_counted_separately(logger):
    logger.record_db_pool_sample(
        _update(wait_delta=1.5, acquire_delta=6, query_seconds_delta=0.4, query_delta=4)
    )

    assert _value("litellm_db_pool_acquire_wait_seconds_total") == pytest.approx(1.5)
    assert _value("litellm_db_pool_acquire_total") == 6.0
    assert _value("litellm_db_query_duration_seconds_total") == pytest.approx(0.4)
    assert _value("litellm_db_query_total") == 4.0


def test_counters_accumulate_across_samples(logger):
    logger.record_db_pool_sample(_update(acquire_delta=6, wait_delta=1.5))
    logger.record_db_pool_sample(_update(acquire_delta=0, wait_delta=0.0))
    logger.record_db_pool_sample(_update(acquire_delta=4, wait_delta=0.5))

    assert _value("litellm_db_pool_acquire_total") == 10.0
    assert _value("litellm_db_pool_acquire_wait_seconds_total") == pytest.approx(2.0)


def test_pool_timeouts_are_counted(logger):
    logger.record_db_pool_timeout()
    logger.record_db_pool_timeout()

    assert _value("litellm_db_pool_timeouts_total") == 2.0


@pytest.mark.asyncio
async def test_a_broken_metric_never_fails_the_database_call_it_rides_on(logger):
    """The publish path sits inline on a DB call, so the safety boundary lives
    in log_db_metrics rather than in each recorder."""
    from unittest.mock import patch

    from litellm.proxy.db.log_db_metrics import (
        _record_db_pool_timeout_if_exhausted,
        _sample_db_pool_metrics,
    )

    logger.record_db_pool_sample = MagicMock(side_effect=RuntimeError("registry gone"))
    logger.record_db_pool_timeout = MagicMock(side_effect=RuntimeError("registry gone"))

    prisma_client = MagicMock()

    async def _sample():
        raise RuntimeError("engine gone")

    prisma_client.db.get_pool_sample = _sample

    with (
        patch("litellm.proxy.db.log_db_metrics._prometheus_logger", return_value=logger),
        patch("litellm.proxy.proxy_server.prisma_client", prisma_client),
    ):
        await _sample_db_pool_metrics()
        _record_db_pool_timeout_if_exhausted(
            DataError({"user_facing_error": {"error_code": "P2024", "message": "pool timeout"}})
        )


@pytest.mark.asyncio
async def test_an_unimportable_prometheus_module_does_not_break_the_db_call():
    """_prometheus_logger imports lazily. If that import fails, a database call
    that already succeeded must still succeed, and a pool timeout must still
    surface as the original P2024 rather than as an ImportError."""
    from unittest.mock import patch

    from litellm.proxy.db.log_db_metrics import (
        _record_db_pool_timeout_if_exhausted,
        _sample_db_pool_metrics,
    )

    with patch(
        "litellm.proxy.db.log_db_metrics._prometheus_logger",
        side_effect=ImportError("prometheus integration unavailable"),
    ):
        await _sample_db_pool_metrics()
        _record_db_pool_timeout_if_exhausted(
            DataError({"user_facing_error": {"error_code": "P2024", "message": "pool timeout"}})
        )


@pytest.mark.parametrize("metric", DB_POOL_METRICS)
def test_the_emitted_series_carries_no_labels(logger, metric):
    """The ticket forbids unbounded labels on saturation metrics. Asserted on
    the constructed metric rather than on PrometheusMetricLabels, which these
    metrics never consult, so this fails if someone adds a labelname."""
    assert getattr(logger, metric)._labelnames == ()


POOL_GAUGES = (
    "litellm_db_pool_connections_max",
    "litellm_db_pool_connections_busy",
    "litellm_db_pool_connections_idle",
    "litellm_db_pool_connections_open",
    "litellm_db_pool_pending_acquirers",
)


@pytest.mark.parametrize("metric", POOL_GAUGES)
def test_pool_gauges_aggregate_across_workers_instead_of_fanning_out_per_pid(logger, metric):
    """Under PROMETHEUS_MULTIPROC_DIR a Gauge with no multiprocess_mode defaults
    to 'all', which stamps an unbounded pid label on every series and never
    aggregates. Worse, mark_process_dead only reaps gauge_live* files, so a
    recycled worker leaves its last reading in every later scrape. livesum sums
    over live workers and drops dead ones, which is what a pod-level pool number
    means."""
    assert getattr(logger, metric)._multiprocess_mode == "livesum"


def test_the_gauge_publishes_the_corrected_pending_not_the_raw_engine_reading(logger):
    """The engine's waiter gauge latches on pool timeouts, so the sampler
    corrects it. Publishing sample.pending_acquirers instead would put the
    latched value on the dashboard."""
    latched = DBPoolSample(
        busy_connections=0.0,
        idle_connections=2.0,
        open_connections=2.0,
        pending_acquirers=7.0,
        acquire_wait_seconds_total=0.0,
        acquire_count_total=0,
        query_duration_seconds_total=0.0,
        query_count_total=0,
    )
    logger.record_db_pool_sample(
        DBPoolMetricsUpdate(
            sample=latched,
            pending_acquirers=0.0,
            acquire_wait_seconds_delta=0.0,
            acquire_count_delta=0,
            query_duration_seconds_delta=0.0,
            query_count_delta=0,
        )
    )

    assert _value("litellm_db_pool_pending_acquirers") == 0.0


def test_get_instance_finds_a_logger_registered_via_success_callback(logger):
    """litellm_settings.success_callback: ["prometheus"] is an equally supported
    registration and lands the logger on the success-callback lists rather than
    litellm.callbacks. Searching only the latter left every pool metric
    registered and permanently at zero, verified on a live proxy."""
    import litellm
    from litellm.integrations.prometheus import PrometheusLogger

    saved_callbacks = litellm.callbacks
    saved_success = litellm.success_callback
    try:
        litellm.callbacks = []
        litellm.success_callback = [logger]
        assert PrometheusLogger.get_instance() is logger
    finally:
        litellm.callbacks = saved_callbacks
        litellm.success_callback = saved_success


def test_get_instance_returns_none_when_prometheus_is_not_registered(logger):
    import litellm
    from litellm.integrations.prometheus import PrometheusLogger

    saved_callbacks = litellm.callbacks
    saved_success = litellm.success_callback
    try:
        litellm.callbacks = []
        litellm.success_callback = []
        assert PrometheusLogger.get_instance() is None
    finally:
        litellm.callbacks = saved_callbacks
        litellm.success_callback = saved_success
