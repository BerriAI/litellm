"""
Mock prometheus unit tests, these don't rely on LLM API calls
"""

import json
import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(
    0, os.path.abspath("../../..")
)  # Adds the parent directory to the system path

from apscheduler.schedulers.asyncio import AsyncIOScheduler

# Add prometheus_client import for registry cleanup
from prometheus_client import REGISTRY

import litellm
from litellm.constants import PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES
from litellm.integrations.prometheus import PrometheusLogger, prometheus_label_factory
from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    PrometheusMetricsConfig,
    UserAPIKeyLabelValues,
)


@pytest.fixture
def prometheus_logger() -> PrometheusLogger:
    """
    Fixture that creates a clean PrometheusLogger instance by clearing the registry first.
    This prevents "Duplicated timeseries in CollectorRegistry" errors.
    """
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def clear_prometheus_registry():
    """Helper function to clear the Prometheus registry"""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


def test_initialize_budget_metrics_cron_job():
    # Clear registry before test
    clear_prometheus_registry()

    # Create a scheduler
    scheduler = AsyncIOScheduler()

    # Create and register a PrometheusLogger
    prometheus_logger = PrometheusLogger()
    litellm.callbacks = [prometheus_logger]

    # Initialize the cron job
    PrometheusLogger.initialize_budget_metrics_cron_job(scheduler)

    # Verify that a job was added to the scheduler
    jobs = scheduler.get_jobs()
    assert len(jobs) == 1

    # Verify job properties
    job = jobs[0]
    assert (
        job.trigger.interval.total_seconds() / 60
        == PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES
    )
    assert job.func.__name__ == "initialize_remaining_budget_metrics"


def test_end_user_not_tracked_for_all_prometheus_metrics():
    """
    Test that end_user is not tracked for all Prometheus metrics by default.

    This test ensures that:
    1. By default, end_user is filtered out from all Prometheus metrics
    2. Future metrics that include end_user in their label definitions will also be filtered
    3. The filtering happens through the prometheus_label_factory function
    """
    # Reset any previous settings
    original_setting = getattr(
        litellm, "enable_end_user_cost_tracking_prometheus_only", None
    )
    litellm.enable_end_user_cost_tracking_prometheus_only = None  # Default behavior

    try:
        # Test data with end_user present
        test_end_user_id = "test_user_123"
        enum_values = UserAPIKeyLabelValues(
            end_user=test_end_user_id,
            hashed_api_key="test_key",
            api_key_alias="test_alias",
            team="test_team",
            team_alias="test_team_alias",
            user="test_user",
            requested_model="gpt-4",
            model="gpt-4",
            litellm_model_name="gpt-4",
        )

        # Get all defined Prometheus metrics that include end_user in their labels
        metrics_with_end_user = []
        for metric_name in PrometheusMetricLabels.__dict__:
            if not metric_name.startswith("_") and metric_name != "get_labels":
                labels = getattr(PrometheusMetricLabels, metric_name)
                if isinstance(labels, list) and "end_user" in labels:
                    metrics_with_end_user.append(metric_name)

        # Ensure we found some metrics with end_user (sanity check)
        assert (
            len(metrics_with_end_user) > 0
        ), "No metrics with end_user found - test setup issue"

        # Test each metric that includes end_user in its label definition
        for metric_name in metrics_with_end_user:
            supported_labels = PrometheusMetricLabels.get_labels(metric_name)

            # Verify that end_user is in the supported labels (before filtering)
            assert (
                "end_user" in supported_labels
            ), f"end_user should be in {metric_name} labels"

            # Call prometheus_label_factory to get filtered labels
            filtered_labels = prometheus_label_factory(
                supported_enum_labels=supported_labels, enum_values=enum_values
            )
            print("filtered labels logged on prometheus=", filtered_labels)

            # Verify that end_user is None in the filtered labels (filtered out)
            assert filtered_labels.get("end_user") is None, (
                f"end_user should be None for metric {metric_name} when "
                f"enable_end_user_cost_tracking_prometheus_only is not True. "
                f"Got: {filtered_labels.get('end_user')}"
            )

        # Test that when enable_end_user_cost_tracking_prometheus_only is True, end_user is tracked
        litellm.enable_end_user_cost_tracking_prometheus_only = True

        # Test one metric to verify end_user is now included
        test_metric = metrics_with_end_user[0]
        supported_labels = PrometheusMetricLabels.get_labels(test_metric)
        filtered_labels = prometheus_label_factory(
            supported_enum_labels=supported_labels, enum_values=enum_values
        )

        # Now end_user should be present
        assert filtered_labels.get("end_user") == test_end_user_id, (
            f"end_user should be present for metric {test_metric} when "
            f"enable_end_user_cost_tracking_prometheus_only is True"
        )

    finally:
        # Restore original setting
        litellm.enable_end_user_cost_tracking_prometheus_only = original_setting


def test_future_metrics_with_end_user_are_filtered():
    """
    Test that ensures future metrics that include end_user will also be filtered.
    This simulates adding a new metric with end_user in its labels.
    """
    # Reset setting
    original_setting = getattr(
        litellm, "enable_end_user_cost_tracking_prometheus_only", None
    )
    litellm.enable_end_user_cost_tracking_prometheus_only = None

    try:
        # Simulate a new metric that includes end_user
        simulated_new_metric_labels = [
            "end_user",
            "hashed_api_key",
            "api_key_alias",
            "model",
            "team",
            "new_label",  # Some new label that might be added in the future
        ]

        test_end_user_id = "future_test_user"
        enum_values = UserAPIKeyLabelValues(
            end_user=test_end_user_id,
            hashed_api_key="test_key",
            api_key_alias="test_alias",
            team="test_team",
            model="gpt-4",
        )

        # Test the filtering
        filtered_labels = prometheus_label_factory(
            supported_enum_labels=simulated_new_metric_labels, enum_values=enum_values
        )
        print("filtered labels logged on prometheus=", filtered_labels)

        # Verify end_user is filtered out even for this "new" metric
        assert (
            filtered_labels.get("end_user") is None
        ), "end_user should be filtered out for future metrics by default"

        # Verify other labels are present
        assert filtered_labels.get("hashed_api_key") == "test_key"
        assert filtered_labels.get("team") == "test_team"

    finally:
        # Restore original setting
        litellm.enable_end_user_cost_tracking_prometheus_only = original_setting


def test_prometheus_config_parsing():
    """Test that prometheus metrics configuration is parsed correctly"""
    # Clear registry before test
    clear_prometheus_registry()

    # Set up test configuration
    test_config = [
        {
            "group": "service_metrics",
            "metrics": [
                "litellm_deployment_failure_responses",
                "litellm_deployment_total_requests",
                "litellm_proxy_failed_requests_metric",
                "litellm_proxy_total_requests_metric",
            ],
            "include_labels": [
                "litellm_model_name",
                "requested_model",
                "api_base",
                "api_provider",
                "exception_status",
                "exception_class",
            ],
        }
    ]

    # Set configuration
    litellm.prometheus_metrics_config = test_config

    # Create PrometheusLogger instance
    logger = PrometheusLogger()

    # Parse configuration
    label_filters = logger._parse_prometheus_config()

    # Verify label filters exist for each metric
    expected_labels = [
        "litellm_model_name",
        "requested_model",
        "api_base",
        "api_provider",
        "exception_status",
        "exception_class",
    ]

    expected_metrics = [
        "litellm_deployment_failure_responses",
        "litellm_deployment_total_requests",
        "litellm_proxy_failed_requests_metric",
        "litellm_proxy_total_requests_metric",
    ]

    for metric in expected_metrics:
        assert metric in label_filters
        assert label_filters[metric] == expected_labels


def test_get_metric_labels():
    """Test that metric label filtering works correctly"""
    # Clear registry before test
    clear_prometheus_registry()

    # Set up test configuration
    test_config = [
        {
            "group": "service_metrics",
            "metrics": ["litellm_deployment_failure_responses"],
            "include_labels": ["litellm_model_name", "api_provider"],
        }
    ]

    litellm.prometheus_metrics_config = test_config

    logger = PrometheusLogger()

    # Get filtered labels
    labels = logger._get_metric_labels("litellm_deployment_failure_responses")

    # Verify only configured labels are returned
    assert "litellm_model_name" in labels
    assert "api_provider" in labels
    # These should be filtered out even if they're in the default labels
    assert (
        len([l for l in labels if l not in ["litellm_model_name", "api_provider"]]) == 0
    )


def test_no_prometheus_config():
    """Test behavior when no prometheus config is set"""
    # Clear registry before test
    clear_prometheus_registry()

    # Clear any existing config
    litellm.prometheus_metrics_config = None

    logger = PrometheusLogger()

    # Should return default labels when no config is set
    labels = logger._get_metric_labels("litellm_deployment_failure_responses")
    # Should return some labels (the default ones)
    assert isinstance(labels, list)
    # Should have more than 0 labels (the default ones)
    assert len(labels) > 0


def test_prometheus_metrics_config_type():
    """Test that PrometheusMetricsConfig type validation works"""
    # Valid configuration
    valid_config = PrometheusMetricsConfig(
        group="service_metrics",
        metrics=["litellm_deployment_failure_responses"],
        include_labels=["litellm_model_name"],
    )

    assert valid_config.group == "service_metrics"
    assert valid_config.metrics == ["litellm_deployment_failure_responses"]
    assert valid_config.include_labels == ["litellm_model_name"]

    # Test with None include_labels (should be allowed)
    config_no_labels = PrometheusMetricsConfig(
        group="service_metrics",
        metrics=["litellm_deployment_failure_responses"],
        include_labels=None,
    )

    assert config_no_labels.include_labels is None
    print("PrometheusMetricsConfig type validation passed!")


def test_basic_functionality():
    """Test basic functionality without creating multiple instances"""
    # Clear registry before test
    clear_prometheus_registry()

    # Set up test configuration
    test_config = [
        {
            "group": "service_metrics",
            "metrics": [
                "litellm_deployment_failure_responses",
                "litellm_deployment_total_requests",
            ],
            "include_labels": ["litellm_model_name", "api_provider"],
        }
    ]

    # Set configuration
    litellm.prometheus_metrics_config = test_config

    # Test that the configuration is properly set
    assert litellm.prometheus_metrics_config is not None
    assert len(litellm.prometheus_metrics_config) == 1
    assert litellm.prometheus_metrics_config[0]["group"] == "service_metrics"
    assert (
        "litellm_deployment_failure_responses"
        in litellm.prometheus_metrics_config[0]["metrics"]
    )

    print("Basic prometheus configuration test passed!")


if __name__ == "__main__":
    test_prometheus_metrics_config_type()
    test_basic_functionality()
    test_prometheus_config_parsing()
    test_get_metric_labels()
    test_no_prometheus_config()
    print("All tests passed!")
