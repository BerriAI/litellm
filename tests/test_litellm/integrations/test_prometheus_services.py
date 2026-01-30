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
    """
    Validates the perf regression fix: is_metric_registered() must NOT call REGISTRY.collect()
    when _names_to_collectors is available. collect() is O(all metrics) and causes severe
    slowdown when a new Router (and thus PrometheusServicesLogger) is created per request
    (e.g. hierarchical router_settings from DB). See GitHub issue #19921.
    """
    from prometheus_client import CollectorRegistry, Counter, Histogram

    # Simulate a proxy with many metrics (e.g. router, redis, postgres, llm providers)
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

    # Mock collect() - the bug is that is_metric_registered() calls it (expensive)
    original_collect = registry.collect
    collect_called = []

    def track_collect(*args, **kwargs):
        collect_called.append(1)
        return original_collect(*args, **kwargs)

    registry.collect = track_collect

    # Simulate Router init: many is_metric_registered() calls; time to show fix vs no-fix difference
    n_calls = 30 * 2  # 30 iters × 2 names
    start = time.perf_counter()
    for _ in range(30):
        pl.is_metric_registered("litellm_service_0_latency")
        pl.is_metric_registered("litellm_service_79_total_requests")
    elapsed_s = time.perf_counter() - start
    elapsed_ms = elapsed_s * 1000
    per_call_us = (elapsed_s / n_calls) * 1_000_000 if n_calls else 0
    n_collect = len(collect_called)

    # Clear latency output (visible with pytest -s)
    path = "slow (REGISTRY.collect)" if n_collect else "fast (_names_to_collectors)"
    print(
        f"\n  is_metric_registered latency: {elapsed_ms:.2f} ms total | "
        f"{per_call_us:.1f} µs/call | {n_calls} calls | {n_collect} collect() invocations | {path}\n"
    )

    # With fix: _names_to_collectors is used, collect() must not be called, elapsed ~ms
    # With fix commented out: collect() every time → slow (e.g. 50–500ms+), test fails
    assert n_collect == 0, (
        f"is_metric_registered() must not use REGISTRY.collect() when _names_to_collectors "
        f"is available (perf regression).\n"
        f"  Latency: {elapsed_ms:.2f} ms total, {per_call_us:.1f} µs per call, {n_calls} calls.\n"
        f"  collect() was called {n_collect} times (slow path).\n"
        f"  With fix: typically <1 ms total, 0 collect() calls. "
        "Uncomment the _names_to_collectors branch in prometheus_services.is_metric_registered()"
    )
    # With fix, assert we're in the fast path (optional; keeps regression visible)
    assert elapsed_s < 0.05, (
        f"is_metric_registered() took {elapsed_ms:.2f} ms for {n_calls} calls "
        f"({per_call_us:.1f} µs/call); expected <50 ms with _names_to_collectors."
    )


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
