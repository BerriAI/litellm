"""
Unit tests for the new Prometheus metrics that were previously missing from validation.

Tests for:
- litellm_remaining_api_key_requests_for_model
- litellm_remaining_api_key_tokens_for_model
- litellm_callback_logging_failures_metric
"""
from typing import get_args
from litellm.types.integrations.prometheus import (
    DEFINED_PROMETHEUS_METRICS,
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
)


def test_new_metrics_in_defined_metrics():
    """
    Test that the new metrics are present in DEFINED_PROMETHEUS_METRICS.
    """
    defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)

    new_metrics = [
        "litellm_remaining_api_key_requests_for_model",
        "litellm_remaining_api_key_tokens_for_model",
        "litellm_callback_logging_failures_metric",
    ]

    for metric in new_metrics:
        assert (
            metric in defined_metrics
        ), f"{metric} should be in DEFINED_PROMETHEUS_METRICS"


def test_new_metrics_have_correct_labels():
    """
    Test that the new metrics have the correct labels defined.
    """
    # Test API Key limits metrics labels
    api_key_metrics = [
        "litellm_remaining_api_key_requests_for_model",
        "litellm_remaining_api_key_tokens_for_model",
    ]

    expected_api_key_labels = [
        UserAPIKeyLabelNames.API_KEY_HASH.value,
        UserAPIKeyLabelNames.API_KEY_ALIAS.value,
        UserAPIKeyLabelNames.v1_LITELLM_MODEL_NAME.value,
    ]

    for metric in api_key_metrics:
        labels = PrometheusMetricLabels.get_labels(metric)
        for expected_label in expected_api_key_labels:
            assert (
                expected_label in labels
            ), f"{metric} should have label {expected_label}"

    # Test Callback failure metric labels
    callback_metric = "litellm_callback_logging_failures_metric"
    callback_labels = PrometheusMetricLabels.get_labels(callback_metric)

    assert (
        UserAPIKeyLabelNames.CALLBACK_NAME.value in callback_labels
    ), f"{callback_metric} should have label {UserAPIKeyLabelNames.CALLBACK_NAME.value}"


def test_callback_name_label_definition():
    """
    Test that CALLBACK_NAME is defined correctly in UserAPIKeyLabelNames.
    """
    assert UserAPIKeyLabelNames.CALLBACK_NAME.value == "callback_name"


if __name__ == "__main__":
    test_new_metrics_in_defined_metrics()
    test_new_metrics_have_correct_labels()
    test_callback_name_label_definition()
