"""
Unit tests for Prometheus queue depth metrics.

Tests the deployment active/queued request metrics for
Prometheus monitoring support.

GitHub Issue: https://github.com/BerriAI/litellm/issues/17764
"""
import sys
import os
from typing import get_args

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.types.integrations.prometheus import (
    DEFINED_PROMETHEUS_METRICS,
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
)


class TestQueueDepthMetricDefinitions:
    """Tests for queue depth metric definitions in types/integrations/prometheus.py."""

    def test_active_requests_metric_defined(self):
        """litellm_deployment_active_requests should be in DEFINED_PROMETHEUS_METRICS."""
        defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)
        assert "litellm_deployment_active_requests" in defined_metrics, (
            "litellm_deployment_active_requests should be in DEFINED_PROMETHEUS_METRICS"
        )

    def test_queued_requests_metric_defined(self):
        """litellm_deployment_queued_requests should be in DEFINED_PROMETHEUS_METRICS."""
        defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)
        assert "litellm_deployment_queued_requests" in defined_metrics, (
            "litellm_deployment_queued_requests should be in DEFINED_PROMETHEUS_METRICS"
        )


class TestQueueDepthMetricLabels:
    """Tests for queue depth metric labels configuration."""

    def test_active_requests_has_model_name_label(self):
        """litellm_deployment_active_requests should have litellm_model_name label."""
        labels = PrometheusMetricLabels.get_labels("litellm_deployment_active_requests")
        assert UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value in labels, (
            "litellm_deployment_active_requests should have litellm_model_name label"
        )

    def test_active_requests_has_model_group_label(self):
        """litellm_deployment_active_requests should have model_group label."""
        labels = PrometheusMetricLabels.get_labels("litellm_deployment_active_requests")
        assert UserAPIKeyLabelNames.MODEL_GROUP.value in labels, (
            "litellm_deployment_active_requests should have model_group label"
        )

    def test_queued_requests_has_model_name_label(self):
        """litellm_deployment_queued_requests should have litellm_model_name label."""
        labels = PrometheusMetricLabels.get_labels("litellm_deployment_queued_requests")
        assert UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value in labels, (
            "litellm_deployment_queued_requests should have litellm_model_name label"
        )

    def test_queued_requests_has_model_group_label(self):
        """litellm_deployment_queued_requests should have model_group label."""
        labels = PrometheusMetricLabels.get_labels("litellm_deployment_queued_requests")
        assert UserAPIKeyLabelNames.MODEL_GROUP.value in labels, (
            "litellm_deployment_queued_requests should have model_group label"
        )

    def test_both_metrics_have_same_labels(self):
        """Active and queued metrics should have the same label set."""
        active_labels = PrometheusMetricLabels.get_labels("litellm_deployment_active_requests")
        queued_labels = PrometheusMetricLabels.get_labels("litellm_deployment_queued_requests")
        assert active_labels == queued_labels, (
            "Active and queued request metrics should have matching labels"
        )


class TestQueueDepthMetricLabelValues:
    """Tests for label value consistency with existing deployment metrics."""

    def test_exactly_two_labels(self):
        """Queue depth metrics should have exactly 2 labels (model and model_group)."""
        for metric_name in ["litellm_deployment_active_requests", "litellm_deployment_queued_requests"]:
            labels = PrometheusMetricLabels.get_labels(metric_name)
            assert len(labels) == 2, (
                f"Metric {metric_name} should have exactly 2 labels, got {len(labels)}: {labels}"
            )

    def test_required_labels_present(self):
        """Both queue depth metrics should have required labels for aggregation."""
        # Note: v1_LITELLM_MODEL_NAME = "model" (not "litellm_model_name")
        required_labels = [
            UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,  # "model"
            UserAPIKeyLabelNames.MODEL_GROUP.value,            # "model_group"
        ]

        for metric_name in ["litellm_deployment_active_requests", "litellm_deployment_queued_requests"]:
            labels = PrometheusMetricLabels.get_labels(metric_name)
            for required_label in required_labels:
                assert required_label in labels, (
                    f"Metric {metric_name} missing required label {required_label}"
                )
