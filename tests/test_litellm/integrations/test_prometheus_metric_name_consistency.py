"""
Unit tests for prometheus metric name consistency

This test ensures that the metric names used when creating Prometheus metrics
match the names defined in DEFINED_PROMETHEUS_METRICS, so that metric filtering
configuration works correctly.

Related issue: https://github.com/BerriAI/litellm/issues/18221
"""
from typing import get_args

import pytest


def test_remaining_requests_metric_name_in_defined_metrics():
    """
    Test that litellm_remaining_requests_metric is defined in DEFINED_PROMETHEUS_METRICS.

    The metric name should include the _metric suffix to be consistent with the
    configuration format users specify in prometheus_metrics_config.
    """
    from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS

    defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)
    assert (
        "litellm_remaining_requests_metric" in defined_metrics
    ), "litellm_remaining_requests_metric should be in DEFINED_PROMETHEUS_METRICS"


def test_remaining_tokens_metric_name_in_defined_metrics():
    """
    Test that litellm_remaining_tokens_metric is defined in DEFINED_PROMETHEUS_METRICS.

    The metric name should include the _metric suffix to be consistent with the
    configuration format users specify in prometheus_metrics_config.
    """
    from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS

    defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)
    assert (
        "litellm_remaining_tokens_metric" in defined_metrics
    ), "litellm_remaining_tokens_metric should be in DEFINED_PROMETHEUS_METRICS"


def test_prometheus_metric_labels_have_remaining_metrics():
    """
    Test that PrometheusMetricLabels has label definitions for remaining metrics.

    This ensures that the labels can be retrieved when creating the metrics.
    """
    from litellm.types.integrations.prometheus import PrometheusMetricLabels

    # Test that labels can be retrieved for remaining metrics
    remaining_requests_labels = PrometheusMetricLabels.get_labels(
        "litellm_remaining_requests_metric"
    )
    remaining_tokens_labels = PrometheusMetricLabels.get_labels(
        "litellm_remaining_tokens_metric"
    )

    assert isinstance(
        remaining_requests_labels, list
    ), "Labels for litellm_remaining_requests_metric should be a list"
    assert isinstance(
        remaining_tokens_labels, list
    ), "Labels for litellm_remaining_tokens_metric should be a list"

    # These metrics should have api_provider and api_base labels
    assert (
        "api_provider" in remaining_requests_labels
    ), "litellm_remaining_requests_metric should have api_provider label"
    assert (
        "api_base" in remaining_requests_labels
    ), "litellm_remaining_requests_metric should have api_base label"
    assert (
        "api_provider" in remaining_tokens_labels
    ), "litellm_remaining_tokens_metric should have api_provider label"
    assert (
        "api_base" in remaining_tokens_labels
    ), "litellm_remaining_tokens_metric should have api_base label"


def test_all_defined_metrics_have_consistent_naming():
    """
    Test that all metrics defined in DEFINED_PROMETHEUS_METRICS follow
    a consistent naming convention.

    This helps prevent similar inconsistencies in the future.
    """
    from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS

    defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)

    for metric_name in defined_metrics:
        # All metrics should start with 'litellm_'
        assert metric_name.startswith(
            "litellm_"
        ), f"Metric {metric_name} should start with 'litellm_'"


if __name__ == "__main__":
    test_remaining_requests_metric_name_in_defined_metrics()
    test_remaining_tokens_metric_name_in_defined_metrics()
    test_prometheus_metric_labels_have_remaining_metrics()
    test_all_defined_metrics_have_consistent_naming()
    print("All prometheus metric name consistency tests passed!")
