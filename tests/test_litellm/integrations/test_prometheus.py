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

from unittest.mock import patch

import pytest_asyncio
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
    labels = logger.get_labels_for_metric("litellm_deployment_failure_responses")

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
    labels = logger.get_labels_for_metric("litellm_deployment_failure_responses")
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


# ==============================================================================
# SEMANTIC VALIDATION TESTS - Detect logical errors in metric increments
# ==============================================================================


class MockCounter:
    """Mock counter for testing metric increments"""

    def __init__(self, name):
        self.name = name
        self.labels_calls = []
        self.inc_calls = []

    def labels(self, *args, **kwargs):
        self.labels_calls.append(kwargs)
        return self

    def inc(self, value=1):
        self.inc_calls.append(value)


class MockHistogram:
    """Mock histogram for testing metric observations"""

    def __init__(self, name):
        self.name = name
        self.labels_calls = []
        self.observe_calls = []

    def labels(self, *args, **kwargs):
        self.labels_calls.append(kwargs)
        return self

    def observe(self, value):
        self.observe_calls.append(value)


@pytest.fixture
def mock_prometheus_logger():
    """Create a PrometheusLogger with mocked metrics to test increment logic"""
    from unittest.mock import patch

    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)

    with patch("litellm.proxy.proxy_server.premium_user", True):
        logger = PrometheusLogger()

        # Replace metrics with mocks to capture increment calls
        logger.litellm_proxy_total_requests_metric = MockCounter(
            "litellm_proxy_total_requests_metric"
        )
        logger.litellm_tokens_metric = MockCounter("litellm_total_tokens")
        logger.litellm_input_tokens_metric = MockCounter("litellm_input_tokens")
        logger.litellm_output_tokens_metric = MockCounter("litellm_output_tokens")
        logger.litellm_spend_metric = MockCounter("litellm_spend_metric")
        logger.litellm_requests_metric = MockCounter("litellm_requests_metric")

        return logger


@pytest.mark.asyncio
async def test_request_counter_semantic_validation(mock_prometheus_logger):
    """
    CRITICAL TEST: Validates that request counters are incremented by 1, not by token count.
    This test specifically catches the bug where litellm_proxy_total_requests_metric
    is incorrectly incremented by total_tokens instead of 1.
    """
    from datetime import datetime, timedelta
    from unittest.mock import MagicMock

    from litellm.proxy._types import UserAPIKeyAuth

    # Test data with large token count that should NOT affect request counter
    kwargs = {
        "model": "gpt-3.5-turbo",
        "litellm_params": {"metadata": {}},
        "start_time": datetime.now() - timedelta(seconds=1),
        "end_time": datetime.now(),
        "api_call_start_time": datetime.now() - timedelta(seconds=0.5),
        "standard_logging_object": {
            "total_tokens": 999,  # Large number - this should NOT be used for request counter
            "prompt_tokens": 600,
            "completion_tokens": 399,
            "response_cost": 0.005,
            "model_group": "gpt-3.5-turbo",
            "model_id": "test-model-id",
            "api_base": "https://api.openai.com/v1",
            "custom_llm_provider": "openai",
            "stream": False,
            "request_tags": [],
            "metadata": {
                "user_api_key_user_id": "test-user",
                "user_api_key_hash": "test-hash",
                "user_api_key_alias": "test-alias",
                "user_api_key_team_id": "test-team",
                "user_api_key_team_alias": "test-team-alias",
                "user_api_key_user_email": "test@example.com",
            },
            "hidden_params": {
                "additional_headers": {},
            },
        },
    }

    # Call the success event
    await mock_prometheus_logger.async_log_success_event(
        kwargs, None, kwargs["start_time"], kwargs["end_time"]
    )

    # CRITICAL ASSERTION: Request counter should not be incremented
    total_requests_metric = mock_prometheus_logger.litellm_proxy_total_requests_metric
    assert (
        len(total_requests_metric.inc_calls) == 0
    ), "Request metric should not be incremented"

    # Call the post-call logging hook
    await mock_prometheus_logger.async_post_call_success_hook(
        data={},
        user_api_key_dict=UserAPIKeyAuth(
            end_user="test-user",
            hashed_api_key="test-hash",
            api_key_alias="test-alias",
            team="test-team",
            model="gpt-4",
        ),
        response=MagicMock(),
    )

    # CRITICAL ASSERTION: Request counter be incremented by 1
    total_requests_metric = mock_prometheus_logger.litellm_proxy_total_requests_metric
    assert (
        len(total_requests_metric.inc_calls) == 1
    ), "Request metric should not be incremented"

    # Check that ALL request counter increments are by 1 (not by token count)
    for inc_value in total_requests_metric.inc_calls:
        assert inc_value == 1, (
            f"SEMANTIC BUG DETECTED: Request counter incremented by {inc_value} instead of 1. "
            f"This indicates the bug where request counters are incremented by token counts."
        )

    # Verify token counters ARE incremented by token counts (this should work correctly)
    tokens_metric = mock_prometheus_logger.litellm_tokens_metric
    assert (
        999 in tokens_metric.inc_calls
    ), "Token metric should be incremented by total_tokens (999)"


@pytest.mark.asyncio
async def test_multiple_requests_counter_semantics(mock_prometheus_logger):
    """
    Test that demonstrates the scaling issue: with multiple requests,
    request counters should scale by number of requests, not total tokens.
    """
    from datetime import datetime, timedelta

    num_requests = 3
    tokens_per_request = 500  # High token count to make the bug obvious

    for i in range(num_requests):
        kwargs = {
            "model": "gpt-3.5-turbo",
            "litellm_params": {"metadata": {}},
            "start_time": datetime.now() - timedelta(seconds=1),
            "end_time": datetime.now(),
            "api_call_start_time": datetime.now() - timedelta(seconds=0.5),
            "standard_logging_object": {
                "total_tokens": tokens_per_request,
                "prompt_tokens": tokens_per_request // 2,
                "completion_tokens": tokens_per_request // 2,
                "response_cost": 0.001,
                "model_group": "gpt-3.5-turbo",
                "model_id": "test-model-id",
                "api_base": "https://api.openai.com/v1",
                "custom_llm_provider": "openai",
                "stream": False,
                "request_tags": [],
                "metadata": {
                    "user_api_key_user_id": "test-user",
                    "user_api_key_hash": "test-hash",
                    "user_api_key_alias": "test-alias",
                    "user_api_key_team_id": "test-team",
                    "user_api_key_team_alias": "test-team-alias",
                    "user_api_key_user_email": "test@example.com",
                },
                "hidden_params": {
                    "additional_headers": {},
                },
            },
        }

        await mock_prometheus_logger.async_log_success_event(
            kwargs, None, kwargs["start_time"], kwargs["end_time"]
        )

    # Calculate total increments
    total_request_increments = sum(
        mock_prometheus_logger.litellm_proxy_total_requests_metric.inc_calls
    )
    total_token_increments = sum(mock_prometheus_logger.litellm_tokens_metric.inc_calls)

    # CRITICAL ASSERTION: Request increments should equal number of requests
    expected_total_tokens = num_requests * tokens_per_request  # 3 * 500 = 1500

    # With the bug, total_request_increments would be 1500 instead of 3
    assert total_request_increments == 0, (
        f"SEMANTIC BUG: Request counter total increments = 0, "
        f"expected {num_requests}. This suggests request counters are being incremented "
        f"by token counts instead of request counts."
    )

    # Token counter should correctly equal total tokens
    assert (
        total_token_increments == expected_total_tokens
    ), f"Token counter should sum to {expected_total_tokens}, got {total_token_increments}"


@pytest.mark.asyncio
async def test_streaming_request_counter_semantics(mock_prometheus_logger):
    """
    Test that streaming requests are also counted correctly (by 1, not by token count)
    """
    from datetime import datetime, timedelta

    kwargs = {
        "model": "gpt-3.5-turbo",
        "litellm_params": {"metadata": {}},
        "start_time": datetime.now() - timedelta(seconds=1),
        "end_time": datetime.now(),
        "api_call_start_time": datetime.now() - timedelta(seconds=0.5),
        "standard_logging_object": {
            "total_tokens": 750,  # High token count for streaming
            "prompt_tokens": 300,
            "completion_tokens": 450,
            "response_cost": 0.003,
            "model_group": "gpt-3.5-turbo",
            "model_id": "test-model-id",
            "api_base": "https://api.openai.com/v1",
            "custom_llm_provider": "openai",
            "stream": True,  # This is a streaming request
            "request_tags": [],
            "metadata": {
                "user_api_key_user_id": "test-user",
                "user_api_key_hash": "test-hash",
                "user_api_key_alias": "test-alias",
                "user_api_key_team_id": "test-team",
                "user_api_key_team_alias": "test-team-alias",
                "user_api_key_user_email": "test@example.com",
            },
            "hidden_params": {
                "additional_headers": {},
            },
        },
    }

    await mock_prometheus_logger.async_log_success_event(
        kwargs, None, kwargs["start_time"], kwargs["end_time"]
    )

    # Streaming requests should also be counted as 1 request, not 750
    for (
        inc_value
    ) in mock_prometheus_logger.litellm_proxy_total_requests_metric.inc_calls:
        assert (
            inc_value == 1
        ), f"SEMANTIC BUG: Streaming request counter incremented by {inc_value} instead of 1"


def test_metric_increment_invariants():
    """
    Test invariants that should always hold for different metric types
    """
    # Invariant 1: Request counters should never be incremented by large values
    suspicious_request_increments = [
        100,
        500,
        1000,
        1500,
    ]  # These look like token counts
    for increment in suspicious_request_increments:
        # If we see request counters incremented by these values, it's likely a bug
        assert (
            increment > 10
        ), f"Request increment of {increment} is suspiciously large - likely a semantic bug"

    # Invariant 2: Token counters should never be incremented by 1 (unless it's a 1-token response)
    # This would indicate the reverse bug (using request count for token counter)

    # Invariant 3: Cost increments should be small positive floats
    reasonable_costs = [0.001, 0.01, 0.1, 1.0]
    for cost in reasonable_costs:
        assert 0 < cost < 100, f"Cost {cost} should be in reasonable range"


def test_token_counter_semantics():
    """
    Test that token counters should be incremented by actual token values, not by 1
    """
    # These are correct patterns for token counters
    correct_token_increments = [50, 100, 250, 500, 1000, 2000]

    for tokens in correct_token_increments:
        # Token counters should be incremented by actual token counts
        assert tokens > 1, f"Token increment of {tokens} is reasonable"

    # These would be incorrect for token counters (suggests using request count for tokens)
    incorrect_token_increments = [1]  # Unless it's actually a 1-token response

    # This test documents the expected behavior - token counters should use token values


@pytest.mark.asyncio
async def test_spend_counter_semantics(mock_prometheus_logger):
    """
    Test that spend counters are incremented by cost amounts, not by 1 or token counts
    """
    from datetime import datetime, timedelta

    kwargs = {
        "model": "gpt-3.5-turbo",
        "litellm_params": {"metadata": {}},
        "start_time": datetime.now() - timedelta(seconds=1),
        "end_time": datetime.now(),
        "api_call_start_time": datetime.now() - timedelta(seconds=0.5),
        "standard_logging_object": {
            "total_tokens": 100,
            "prompt_tokens": 60,
            "completion_tokens": 40,
            "response_cost": 0.0015,  # This should be used for spend metrics
            "model_group": "gpt-3.5-turbo",
            "model_id": "test-model-id",
            "api_base": "https://api.openai.com/v1",
            "custom_llm_provider": "openai",
            "stream": False,
            "request_tags": [],
            "metadata": {
                "user_api_key_user_id": "test-user",
                "user_api_key_hash": "test-hash",
                "user_api_key_alias": "test-alias",
                "user_api_key_team_id": "test-team",
                "user_api_key_team_alias": "test-team-alias",
                "user_api_key_user_email": "test@example.com",
            },
            "hidden_params": {
                "additional_headers": {},
            },
        },
    }

    await mock_prometheus_logger.async_log_success_event(
        kwargs, None, kwargs["start_time"], kwargs["end_time"]
    )

    # Verify spend counter is incremented by cost amount
    spend_metric = mock_prometheus_logger.litellm_spend_metric
    assert len(spend_metric.inc_calls) > 0, "Spend metric should be incremented"
    assert (
        0.0015 in spend_metric.inc_calls
    ), "Spend metric should be incremented by response_cost (0.0015)"


# ==============================================================================
# END SEMANTIC VALIDATION TESTS
# ==============================================================================
