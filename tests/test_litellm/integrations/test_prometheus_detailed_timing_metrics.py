"""
Unit tests for litellm_pre_processing_latency_metric and
litellm_post_processing_latency_metric.

These histograms export the per-phase timing breakdown that
LITELLM_DETAILED_TIMING already computes onto the response hidden params. They
are only observed when the flag is enabled, and a phase whose value is absent is
skipped rather than observed as 0.
"""

from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

import litellm.integrations.prometheus as prometheus_mod
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.utils import StandardLoggingPayload


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)
    yield
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)


def _patch_label_factory(monkeypatch):
    monkeypatch.setattr(
        "litellm.integrations.prometheus.prometheus_label_factory",
        lambda **kwargs: {},
    )


def test_detailed_timing_observed_in_seconds_when_enabled(monkeypatch):
    """With the flag on, pre/post timings (ms) are observed as seconds."""
    _patch_label_factory(monkeypatch)
    monkeypatch.setattr(prometheus_mod, "LITELLM_DETAILED_TIMING", True)
    logger = PrometheusLogger()
    logger.litellm_pre_processing_latency_metric = MagicMock()
    logger.litellm_post_processing_latency_metric = MagicMock()

    payload = StandardLoggingPayload(
        hidden_params={
            "timing_pre_processing_ms": 20.0,
            "timing_post_processing_ms": 10.0,
        },
    )
    logger._set_detailed_timing_metrics(
        payload, enum_values=MagicMock(), label_context=MagicMock()
    )

    pre = logger.litellm_pre_processing_latency_metric.labels.return_value.observe
    post = logger.litellm_post_processing_latency_metric.labels.return_value.observe
    pre.assert_called_once_with(0.02)
    post.assert_called_once_with(0.01)


def test_detailed_timing_not_observed_when_disabled(monkeypatch):
    """With the flag off, nothing is observed even if values are present."""
    _patch_label_factory(monkeypatch)
    monkeypatch.setattr(prometheus_mod, "LITELLM_DETAILED_TIMING", False)
    logger = PrometheusLogger()
    logger.litellm_pre_processing_latency_metric = MagicMock()
    logger.litellm_post_processing_latency_metric = MagicMock()

    payload = StandardLoggingPayload(
        hidden_params={
            "timing_pre_processing_ms": 20.0,
            "timing_post_processing_ms": 10.0,
        },
    )
    logger._set_detailed_timing_metrics(
        payload, enum_values=MagicMock(), label_context=MagicMock()
    )

    logger.litellm_pre_processing_latency_metric.labels.assert_not_called()
    logger.litellm_post_processing_latency_metric.labels.assert_not_called()


def test_detailed_timing_skips_phase_with_missing_value(monkeypatch):
    """A phase with no timing value must not emit a series (no bogus 0 observation)."""
    _patch_label_factory(monkeypatch)
    monkeypatch.setattr(prometheus_mod, "LITELLM_DETAILED_TIMING", True)
    logger = PrometheusLogger()
    logger.litellm_pre_processing_latency_metric = MagicMock()
    logger.litellm_post_processing_latency_metric = MagicMock()

    payload = StandardLoggingPayload(
        hidden_params={"timing_pre_processing_ms": 20.0},
    )
    logger._set_detailed_timing_metrics(
        payload, enum_values=MagicMock(), label_context=MagicMock()
    )

    logger.litellm_pre_processing_latency_metric.labels.return_value.observe.assert_called_once_with(0.02)
    logger.litellm_post_processing_latency_metric.labels.assert_not_called()


def test_detailed_timing_metrics_are_registered():
    """Both per-phase histograms are defined and registered on logger init."""
    logger = PrometheusLogger()
    assert logger.litellm_pre_processing_latency_metric is not None
    assert logger.litellm_post_processing_latency_metric is not None

    for metric_name in (
        "litellm_pre_processing_latency_metric",
        "litellm_post_processing_latency_metric",
    ):
        registered = [name for name in REGISTRY._names_to_collectors if name.startswith(metric_name)]
        assert registered, f"{metric_name} not registered"
