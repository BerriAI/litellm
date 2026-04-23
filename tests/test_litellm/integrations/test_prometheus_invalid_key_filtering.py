"""
Unit tests for Prometheus invalid API key request filtering.

Tests functionality that prevents invalid API key requests (401 status codes)
from being recorded in Prometheus metrics.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest
from prometheus_client import REGISTRY

sys.path.insert(0, os.path.abspath("../../.."))

from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import UserAPIKeyAuth


@pytest.fixture(scope="function")
def prometheus_logger():
    """Create a PrometheusLogger instance for testing."""
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    return PrometheusLogger()


class ExceptionWithCode:
    """Exception-like object with 'code' attribute (ProxyException pattern)."""
    def __init__(self, code):
        self.code = code


class ExceptionWithStatusCode:
    """Exception-like object with 'status_code' attribute."""
    def __init__(self, status_code):
        self.status_code = status_code


class TestExtractStatusCode:
    """Test status code extraction from various sources."""

    @pytest.mark.parametrize("exception_class,code_value,expected", [
        (ExceptionWithCode, "401", 401),
        (ExceptionWithStatusCode, 401, 401),
    ])
    def test_extract_from_exception(self, prometheus_logger, exception_class, code_value, expected):
        exception = exception_class(code_value)
        assert prometheus_logger._extract_status_code(exception=exception) == expected

    def test_extract_from_kwargs(self, prometheus_logger):
        exception = ExceptionWithCode("401")
        assert prometheus_logger._extract_status_code(kwargs={"exception": exception}) == 401

    def test_extract_from_enum_values(self, prometheus_logger):
        enum_values = Mock(status_code="401")
        assert prometheus_logger._extract_status_code(enum_values=enum_values) == 401


class TestInvalidAPIKeyDetection:
    """Test invalid API key request detection logic."""

    @pytest.mark.parametrize("status_code,expected", [
        (401, True),
        (200, False),
        (500, False),
        (None, False),
    ])
    def test_status_code_detection(self, prometheus_logger, status_code, expected):
        assert prometheus_logger._is_invalid_api_key_request(status_code=status_code) == expected

    def test_auth_error_message_detection(self, prometheus_logger):
        exception = AssertionError("LiteLLM Virtual Key expected. Received=invalid-key-12345, expected to start with 'sk-'.")
        assert prometheus_logger._is_invalid_api_key_request(status_code=None, exception=exception) is True

    def test_non_auth_exception_not_detected(self, prometheus_logger):
        exception = ValueError("Some other error")
        assert prometheus_logger._is_invalid_api_key_request(status_code=None, exception=exception) is False


class TestSkipMetricsValidation:
    """Test high-level validation method that orchestrates detection and extraction."""

    def test_skip_for_401_exception(self, prometheus_logger):
        """Test full flow: extraction -> detection -> skip decision."""
        exception = ExceptionWithCode("401")
        assert prometheus_logger._should_skip_metrics_for_invalid_key(exception=exception) is True

    def test_skip_for_auth_error_message(self, prometheus_logger):
        """Test full flow: exception message -> detection -> skip decision."""
        exception = AssertionError("expected to start with 'sk-'")
        assert prometheus_logger._should_skip_metrics_for_invalid_key(exception=exception) is True

    def test_no_skip_for_valid_request(self, prometheus_logger):
        assert prometheus_logger._should_skip_metrics_for_invalid_key() is False


class TestAsyncHooks:
    """Test async hook methods skip metrics for invalid API keys."""

    @pytest.fixture
    def mock_user_api_key(self):
        """Create a mock UserAPIKeyAuth object."""
        user_key = Mock(spec=UserAPIKeyAuth)
        user_key.api_key = "test-key"
        user_key.end_user_id = None
        user_key.user_id = None
        user_key.user_email = None
        user_key.key_alias = None
        user_key.team_id = None
        user_key.team_alias = None
        user_key.request_route = "/test"
        return user_key

    @pytest.mark.asyncio
    async def test_post_call_failure_hook_skips_401(self, prometheus_logger, mock_user_api_key):
        exception = ExceptionWithCode("401")
        exception.__class__.__name__ = "ProxyException"

        with patch.object(prometheus_logger, 'litellm_proxy_failed_requests_metric') as mock_failed, \
             patch.object(prometheus_logger, 'litellm_proxy_total_requests_metric') as mock_total:

            await prometheus_logger.async_post_call_failure_hook(
                request_data={"model": "test-model"},
                original_exception=exception,
                user_api_key_dict=mock_user_api_key
            )

            mock_failed.labels.assert_not_called()
            mock_total.labels.assert_not_called()

    @pytest.mark.asyncio
    async def test_log_failure_event_skips_401(self, prometheus_logger):
        exception = ExceptionWithCode("401")
        kwargs = {
            "model": "test-model",
            "standard_logging_object": {
                "metadata": {
                    "user_api_key_hash": "test-key",
                    "user_api_key_user_id": "test-user",
                },
                "model_group": "test-model",
            },
            "exception": exception,
            "litellm_params": {},
        }

        with patch.object(prometheus_logger, 'litellm_llm_api_failed_requests_metric') as mock_failed, \
             patch.object(prometheus_logger, 'set_llm_deployment_failure_metrics') as mock_deployment:

            await prometheus_logger.async_log_failure_event(
                kwargs=kwargs,
                response_obj=None,
                start_time=None,
                end_time=None
            )

            mock_failed.labels.assert_not_called()
            mock_deployment.assert_not_called()
