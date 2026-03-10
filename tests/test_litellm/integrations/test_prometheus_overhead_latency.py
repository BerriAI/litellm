"""
Unit tests for Prometheus overhead latency metric visibility.

Verifies that litellm_overhead_latency_metric is observed for every request
where litellm_overhead_time_ms is present, including when the value is 0.
"""

import os
import sys
from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


@pytest.fixture(scope="function")
def prometheus_logger():
    """Create a PrometheusLogger instance for testing."""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


def _build_request_kwargs(overhead_time_ms):
    """Build request_kwargs with the given overhead time."""
    return {
        "model": "gpt-3.5-turbo",
        "litellm_params": {
            "custom_llm_provider": "openai",
            "metadata": {"model_info": {"id": "model-123"}},
        },
        "standard_logging_object": {
            "api_base": "https://api.openai.com",
            "model": "gpt-3.5-turbo",
            "model_group": "openai-gpt",
            "custom_llm_provider": "openai",
            "response_cost": 0.001,
            "hidden_params": {
                "additional_headers": None,
                "litellm_overhead_time_ms": overhead_time_ms,
            },
            "metadata": {
                "user_api_key_hash": "test-key",
                "user_api_key_alias": "test-alias",
                "user_api_key_team_id": "test-team",
                "user_api_key_team_alias": "test-team-alias",
            },
        },
    }


def _build_enum_values():
    return UserAPIKeyLabelValues(
        end_user=None,
        hashed_api_key="test-key",
        api_key_alias="test-alias",
        team="test-team",
        team_alias="test-team-alias",
        requested_model="openai-gpt",
        litellm_model_name="gpt-3.5-turbo",
        api_provider="openai",
        api_base="https://api.openai.com",
        model_id="model-123",
        model_group="openai-gpt",
    )


class TestOverheadLatencyMetricVisibility:
    """
    The litellm_overhead_latency_metric must be observed for every request
    where litellm_overhead_time_ms is set, regardless of the value.

    Bug: the original code used a truthiness check via walrus operator:
        if litellm_overhead_time_ms := payload["hidden_params"].get(...)
    which silently dropped observations when the value was 0 or 0.0.
    """

    def test_should_observe_metric_when_overhead_is_zero(self, prometheus_logger):
        """Metric must be observed even when overhead time is exactly 0."""
        prometheus_logger.litellm_overhead_latency_metric = MagicMock()
        prometheus_logger.litellm_deployment_success_responses = MagicMock()
        prometheus_logger.litellm_deployment_total_requests = MagicMock()
        prometheus_logger.litellm_deployment_latency_per_output_token = MagicMock()
        prometheus_logger.set_deployment_healthy = MagicMock()

        request_kwargs = _build_request_kwargs(overhead_time_ms=0)

        prometheus_logger.set_llm_deployment_success_metrics(
            request_kwargs=request_kwargs,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1),
            output_tokens=10.0,
            enum_values=_build_enum_values(),
        )

        prometheus_logger.litellm_overhead_latency_metric.labels.assert_called_once()
        prometheus_logger.litellm_overhead_latency_metric.labels().observe.assert_called_once_with(
            0.0
        )

    def test_should_observe_metric_when_overhead_is_zero_float(self, prometheus_logger):
        """Metric must be observed even when overhead time is 0.0."""
        prometheus_logger.litellm_overhead_latency_metric = MagicMock()
        prometheus_logger.litellm_deployment_success_responses = MagicMock()
        prometheus_logger.litellm_deployment_total_requests = MagicMock()
        prometheus_logger.litellm_deployment_latency_per_output_token = MagicMock()
        prometheus_logger.set_deployment_healthy = MagicMock()

        request_kwargs = _build_request_kwargs(overhead_time_ms=0.0)

        prometheus_logger.set_llm_deployment_success_metrics(
            request_kwargs=request_kwargs,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1),
            output_tokens=10.0,
            enum_values=_build_enum_values(),
        )

        prometheus_logger.litellm_overhead_latency_metric.labels.assert_called_once()
        prometheus_logger.litellm_overhead_latency_metric.labels().observe.assert_called_once_with(
            0.0
        )

    def test_should_observe_metric_when_overhead_is_positive(self, prometheus_logger):
        """Metric must be observed when overhead time is a positive number."""
        prometheus_logger.litellm_overhead_latency_metric = MagicMock()
        prometheus_logger.litellm_deployment_success_responses = MagicMock()
        prometheus_logger.litellm_deployment_total_requests = MagicMock()
        prometheus_logger.litellm_deployment_latency_per_output_token = MagicMock()
        prometheus_logger.set_deployment_healthy = MagicMock()

        request_kwargs = _build_request_kwargs(overhead_time_ms=150.5)

        prometheus_logger.set_llm_deployment_success_metrics(
            request_kwargs=request_kwargs,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1),
            output_tokens=10.0,
            enum_values=_build_enum_values(),
        )

        prometheus_logger.litellm_overhead_latency_metric.labels.assert_called_once()
        prometheus_logger.litellm_overhead_latency_metric.labels().observe.assert_called_once_with(
            150.5 / 1000
        )

    def test_should_not_observe_metric_when_overhead_is_none(self, prometheus_logger):
        """Metric must NOT be observed when overhead time is None (not available)."""
        prometheus_logger.litellm_overhead_latency_metric = MagicMock()
        prometheus_logger.litellm_deployment_success_responses = MagicMock()
        prometheus_logger.litellm_deployment_total_requests = MagicMock()
        prometheus_logger.litellm_deployment_latency_per_output_token = MagicMock()
        prometheus_logger.set_deployment_healthy = MagicMock()

        request_kwargs = _build_request_kwargs(overhead_time_ms=None)

        prometheus_logger.set_llm_deployment_success_metrics(
            request_kwargs=request_kwargs,
            start_time=datetime.now(),
            end_time=datetime.now() + timedelta(seconds=1),
            output_tokens=10.0,
            enum_values=_build_enum_values(),
        )

        prometheus_logger.litellm_overhead_latency_metric.labels.assert_not_called()
