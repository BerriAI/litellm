import asyncio
import json
import time
from typing import Final, cast
from unittest.mock import AsyncMock, patch

import litellm
import pytest
from fastapi.testclient import TestClient

from litellm._service_logger import ServiceLogging
from litellm.caching.redis_cache import RedisCache
from litellm.integrations.prometheus_services import (
    PrometheusServicesLogger,
    ServiceMetrics,
    ServiceTypes,
)
from litellm.types.services import ServiceLoggerPayload



def test_is_metric_registered_does_not_use_registry_collect():
    """is_metric_registered() must use _names_to_collectors, not REGISTRY.collect() (perf; #19921)."""
    from prometheus_client import CollectorRegistry, Counter, Histogram

    registry = CollectorRegistry()
    for i in range(80):
        Counter(
            f"litellm_service_{i}_total_requests",
            "Total requests",
            labelnames=["service"],
            registry=registry,
        )
        Histogram(
            f"litellm_service_{i}_latency",
            "Latency",
            labelnames=["service"],
            registry=registry,
        )

    pl = PrometheusServicesLogger()
    pl.REGISTRY = registry

    original_collect = registry.collect
    collect_called = []

    def track_collect(*args, **kwargs):
        collect_called.append(1)
        return original_collect(*args, **kwargs)

    registry.collect = track_collect

    n_calls = 30 * 2
    start = time.perf_counter()
    for _ in range(30):
        pl.is_metric_registered("litellm_service_0_latency")
        pl.is_metric_registered("litellm_service_79_total_requests")
    elapsed_s = time.perf_counter() - start
    elapsed_ms = elapsed_s * 1000
    per_call_us = (elapsed_s / n_calls) * 1_000_000 if n_calls else 0
    n_collect = len(collect_called)

    path = "slow (REGISTRY.collect)" if n_collect else "fast (_names_to_collectors)"
    print(
        f"\n  is_metric_registered: {elapsed_ms:.2f} ms total | "
        f"{per_call_us:.1f} µs/call | {n_calls} calls | {n_collect} collect() | {path}\n"
    )

    assert n_collect == 0, (
        f"is_metric_registered() must not use REGISTRY.collect() when _names_to_collectors "
        f"is available. Latency: {elapsed_ms:.2f} ms, {per_call_us:.1f} µs/call, {n_calls} calls, "
        f"collect() called {n_collect} times."
    )
    assert (
        elapsed_s < 0.05
    ), f"is_metric_registered() took {elapsed_ms:.2f} ms for {n_calls} calls; expected <50 ms."


def test_create_gauge_new():
    """Test creating a new gauge"""
    pl = PrometheusServicesLogger()

    # Create new gauge
    gauge = pl.create_gauge(service="test_service", type_of_request="size")

    assert gauge is not None
    assert pl._get_metric("litellm_test_service_size") is gauge


def test_update_gauge():
    """Test updating a gauge's value"""
    pl = PrometheusServicesLogger()

    # Create a gauge to test with
    gauge = pl.create_gauge(service="test_service", type_of_request="size")

    # Mock the labels method to verify it's called correctly
    with patch.object(gauge, "labels") as mock_labels:
        mock_gauge = AsyncMock()
        mock_labels.return_value = mock_gauge

        # Call update_gauge
        pl.update_gauge(gauge=gauge, labels="test_label", amount=42.5)

        # Verify correct methods were called
        mock_labels.assert_called_once_with("test_label")
        mock_gauge.set.assert_called_once_with(42.5)


def test_services_logger_default_latency_buckets():
    """PrometheusServicesLogger uses the new reduced default latency buckets."""
    from litellm.types.integrations.prometheus import LATENCY_BUCKETS

    pl = PrometheusServicesLogger()
    assert pl.latency_buckets == LATENCY_BUCKETS
    assert 420.0 in pl.latency_buckets
    assert 600.0 in pl.latency_buckets
    assert 1.5 not in pl.latency_buckets


def test_services_logger_custom_latency_buckets():
    """prometheus_latency_buckets setting is respected by PrometheusServicesLogger."""
    import litellm
    from prometheus_client import REGISTRY

    custom_buckets = [0.1, 0.5, 1.0, 5.0, 10.0]
    original = litellm.prometheus_latency_buckets
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    try:
        litellm.prometheus_latency_buckets = custom_buckets
        pl = PrometheusServicesLogger()
        assert pl.latency_buckets == tuple(custom_buckets)
    finally:
        litellm.prometheus_latency_buckets = original
        for collector in list(REGISTRY._collector_to_names.keys()):
            try:
                REGISTRY.unregister(collector)
            except Exception:
                pass


def test_anthropic_wif_services_are_wired_into_the_registry():
    """Reverting the ANTHROPIC_WIF/ANTHROPIC_WIF_CACHE ServiceTypes members or their
    DEFAULT_SERVICE_CONFIGS entries must fail here: the exchange service gets counters plus a
    latency histogram, while the cache-hit service is counter-only so a hit can never fake a latency."""
    from litellm.types.services import DEFAULT_SERVICE_CONFIGS

    assert ServiceTypes.ANTHROPIC_WIF.value == "anthropic_wif"
    assert ServiceTypes.ANTHROPIC_WIF_CACHE.value == "anthropic_wif_cache"
    assert DEFAULT_SERVICE_CONFIGS["anthropic_wif"]["metrics"] == [ServiceMetrics.COUNTER, ServiceMetrics.HISTOGRAM]
    assert DEFAULT_SERVICE_CONFIGS["anthropic_wif_cache"]["metrics"] == [ServiceMetrics.COUNTER]

    pl = PrometheusServicesLogger()
    wif_names = {obj._name for obj in pl.payload_to_prometheus_map["anthropic_wif"]}
    assert wif_names == {
        "litellm_anthropic_wif_latency",
        "litellm_anthropic_wif_failed_requests",
        "litellm_anthropic_wif_total_requests",
    }
    cache_names = {obj._name for obj in pl.payload_to_prometheus_map["anthropic_wif_cache"]}
    assert cache_names == {
        "litellm_anthropic_wif_cache_failed_requests",
        "litellm_anthropic_wif_cache_total_requests",
    }


@pytest.mark.asyncio
async def test_init_prometheus():
    """
    - Run completion with caching
    - Assert success callback gets called
    """

    pl = PrometheusServicesLogger(mock_testing=True)


@pytest.mark.asyncio
async def test_service_logger_db_monitoring():
    """
    Test prometheus monitoring for database operations
    """
    litellm.service_callback = ["prometheus_system"]
    sl = ServiceLogging()

    # Create spy on prometheus logger's async_service_success_hook
    with patch.object(
        sl.prometheusServicesLogger,
        "async_service_success_hook",
        new_callable=AsyncMock,
    ) as mock_prometheus_success:
        # Test DB success monitoring
        await sl.async_service_success_hook(
            service=ServiceTypes.DB,
            duration=0.3,
            call_type="query",
            event_metadata={"query_type": "SELECT", "table": "api_keys"},
        )

        # Assert prometheus logger's success hook was called
        mock_prometheus_success.assert_called_once()
        # Optionally verify the payload
        actual_payload = mock_prometheus_success.call_args[1]["payload"]
        print("actual_payload sent to prometheus: ", actual_payload)
        assert actual_payload.service == ServiceTypes.DB
        assert actual_payload.duration == 0.3
        assert actual_payload.call_type == "query"
        assert actual_payload.is_error is False


@pytest.mark.asyncio
async def test_service_logger_db_monitoring_failure():
    """
    Test prometheus monitoring for failed database operations
    """
    litellm.service_callback = ["prometheus_system"]
    sl = ServiceLogging()

    # Create spy on prometheus logger's async_service_failure_hook
    with patch.object(
        sl.prometheusServicesLogger,
        "async_service_failure_hook",
        new_callable=AsyncMock,
    ) as mock_prometheus_failure:
        # Test DB failure monitoring
        test_error = Exception("Database connection failed")
        await sl.async_service_failure_hook(
            service=ServiceTypes.DB,
            duration=0.3,
            error=test_error,
            call_type="query",
            event_metadata={"query_type": "SELECT", "table": "api_keys"},
        )

        # Assert prometheus logger's failure hook was called
        mock_prometheus_failure.assert_called_once()
        # Verify the payload
        actual_payload = mock_prometheus_failure.call_args[1]["payload"]
        print("actual_payload sent to prometheus: ", actual_payload)
        assert actual_payload.service == ServiceTypes.DB
        assert actual_payload.duration == 0.3
        assert actual_payload.call_type == "query"
        assert actual_payload.is_error is True
        assert actual_payload.error == "Database connection failed"


@pytest.mark.asyncio
async def test_completion_with_caching_bad_call(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "service_callback", ["prometheus_system"])
    service_logger: Final = ServiceLogging(mock_testing=True)
    service_logger.prometheusServicesLogger.mock_testing = True
    cache: Final = RedisCache(host="redis.invalid", port=6379, service_logger_obj=service_logger)

    await cache.async_set_cache("bad-call-key", "value")
    for _ in range(10):
        await asyncio.sleep(0)

    assert service_logger.mock_testing_async_failure_hook >= 1
    assert (
        service_logger.prometheusServicesLogger.mock_testing_failure_calls
        == service_logger.mock_testing_async_failure_hook
    )
    assert service_logger.mock_testing_async_success_hook == 0
    assert service_logger.prometheusServicesLogger.mock_testing_success_calls == 0


def test_get_metric_existing():
    """Test _get_metric when metric exists. _get_metric should return the metric object"""
    pl = PrometheusServicesLogger()
    # Create a metric first
    hist = pl.create_histogram(
        service="test_service", type_of_request="test_type_of_request"
    )

    # Test retrieving existing metric
    retrieved_metric = pl._get_metric("litellm_test_service_test_type_of_request")
    assert retrieved_metric is hist
    assert retrieved_metric is not None


def test_get_metric_non_existing():
    """Test _get_metric when metric doesn't exist, returns None"""
    pl = PrometheusServicesLogger()

    # Test retrieving non-existent metric
    non_existent = pl._get_metric("non_existent_metric")
    assert non_existent is None


def test_create_histogram_new():
    """Test creating a new histogram"""
    pl = PrometheusServicesLogger()

    # Create new histogram
    hist = pl.create_histogram(
        service="test_service", type_of_request="test_type_of_request"
    )

    assert hist is not None
    assert pl._get_metric("litellm_test_service_test_type_of_request") is hist


def test_create_histogram_existing():
    """Test creating a histogram that already exists"""
    pl = PrometheusServicesLogger()

    # Create initial histogram
    hist1 = pl.create_histogram(
        service="test_service", type_of_request="test_type_of_request"
    )

    # Create same histogram again
    hist2 = pl.create_histogram(
        service="test_service", type_of_request="test_type_of_request"
    )

    assert hist2 is hist1  # same object
    assert pl._get_metric("litellm_test_service_test_type_of_request") is hist1


def test_create_counter_new():
    """Test creating a new counter"""
    pl = PrometheusServicesLogger()

    # Create new counter
    counter = pl.create_counter(
        service="test_service", type_of_request="test_type_of_request"
    )

    assert counter is not None
    assert pl._get_metric("litellm_test_service_test_type_of_request") is counter


def test_create_counter_existing():
    """Test creating a counter that already exists"""
    pl = PrometheusServicesLogger()

    # Create initial counter
    counter1 = pl.create_counter(
        service="test_service", type_of_request="test_type_of_request"
    )

    # Create same counter again
    counter2 = pl.create_counter(
        service="test_service", type_of_request="test_type_of_request"
    )

    assert counter2 is counter1
    assert pl._get_metric("litellm_test_service_test_type_of_request") is counter1
