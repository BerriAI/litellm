"""
Unit tests for prometheus metric labels configuration
"""
from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    UserAPIKeyLabelNames
)


def test_user_email_in_required_metrics():
    """
    Test that user_email label is present in all the metrics that should have it:
    - litellm_proxy_total_requests_metric (already had it)
    - litellm_proxy_failed_requests_metric (added)
    - litellm_input_tokens_metric (added)
    - litellm_output_tokens_metric (added)
    - litellm_requests_metric (already had it)
    - litellm_spend_metric (added)
    """
    user_email_label = UserAPIKeyLabelNames.USER_EMAIL.value

    # Metrics that should have user_email
    metrics_with_user_email = [
        "litellm_proxy_total_requests_metric",
        "litellm_proxy_failed_requests_metric",
        "litellm_input_tokens_metric",
        "litellm_output_tokens_metric",
        "litellm_requests_metric",
        "litellm_spend_metric"
    ]

    for metric_name in metrics_with_user_email:
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert user_email_label in labels, f"Metric {metric_name} should contain user_email label"
        print(f"✅ {metric_name} contains user_email label")


def test_user_email_label_exists():
    """Test that the USER_EMAIL label is properly defined"""
    assert UserAPIKeyLabelNames.USER_EMAIL.value == "user_email"


def test_prometheus_metric_labels_structure():
    """Test that all required prometheus metrics have proper label structure"""
    from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS
    from typing import get_args

    # Test a few key metrics to ensure they have proper label structure
    test_metrics = [
        "litellm_proxy_total_requests_metric",
        "litellm_proxy_failed_requests_metric",
        "litellm_input_tokens_metric",
        "litellm_output_tokens_metric",
        "litellm_spend_metric"
    ]

    for metric_name in test_metrics:
        # Check metric is in DEFINED_PROMETHEUS_METRICS
        assert metric_name in get_args(DEFINED_PROMETHEUS_METRICS), f"{metric_name} should be in DEFINED_PROMETHEUS_METRICS"

        # Check labels can be retrieved
        labels = PrometheusMetricLabels.get_labels(metric_name)
        assert isinstance(labels, list), f"Labels for {metric_name} should be a list"
        assert len(labels) > 0, f"Labels for {metric_name} should not be empty"

        # Check user_email is in the labels
        assert "user_email" in labels, f"{metric_name} should have user_email label"

        print(f"✅ {metric_name} has proper label structure with user_email")


if __name__ == "__main__":
    test_user_email_in_required_metrics()
    test_user_email_label_exists()
    test_prometheus_metric_labels_structure()
    print("All prometheus label tests passed!")