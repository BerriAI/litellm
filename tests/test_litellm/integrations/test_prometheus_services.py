import json
import os
import sys
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from litellm.integrations.prometheus_services import (
    PrometheusServicesLogger,
    ServiceMetrics,
    ServiceTypes,
)

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path


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
