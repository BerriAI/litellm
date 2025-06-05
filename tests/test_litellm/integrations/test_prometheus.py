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

import litellm
from litellm.constants import PROMETHEUS_BUDGET_METRICS_REFRESH_INTERVAL_MINUTES
from litellm.integrations.prometheus import PrometheusLogger, prometheus_label_factory
from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    UserAPIKeyLabelValues,
)


def test_initialize_budget_metrics_cron_job():
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
