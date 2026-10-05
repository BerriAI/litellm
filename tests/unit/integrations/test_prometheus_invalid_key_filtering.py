"""
Unit tests for Prometheus invalid API key request filtering.

Tests the 401 detection helpers, that LLM-level metrics skip invalid API key
requests, and that the proxy-level failed request counter still records them.
"""

from unittest.mock import Mock, patch

import pytest
from fastapi import HTTPException
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from litellm.proxy._types import ProxyErrorTypes, ProxyException, UserAPIKeyAuth


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

    @pytest.mark.parametrize(
        "exception_class,code_value,expected",
        [
            (ExceptionWithCode, "401", 401),
            (ExceptionWithStatusCode, 401, 401),
        ],
    )
    def test_extract_from_exception(
        self, prometheus_logger, exception_class, code_value, expected
    ):
        exception = exception_class(code_value)
        assert prometheus_logger._extract_status_code(exception=exception) == expected

    def test_extract_from_kwargs(self, prometheus_logger):
        exception = ExceptionWithCode("401")
        assert (
            prometheus_logger._extract_status_code(kwargs={"exception": exception})
            == 401
        )

    def test_extract_from_enum_values(self, prometheus_logger):
        enum_values = Mock(status_code="401")
        assert prometheus_logger._extract_status_code(enum_values=enum_values) == 401


class TestInvalidAPIKeyDetection:
    """Test invalid API key request detection logic."""

    @pytest.mark.parametrize(
        "status_code,expected",
        [
            (401, True),
            (200, False),
            (500, False),
            (None, False),
        ],
    )
    def test_status_code_detection(self, prometheus_logger, status_code, expected):
        assert (
            prometheus_logger._is_invalid_api_key_request(status_code=status_code)
            == expected
        )

    def test_auth_error_message_detection(self, prometheus_logger):
        exception = AssertionError(
            "LiteLLM Virtual Key expected. Received=invalid-key-12345, expected to start with 'sk-'."
        )
        assert (
            prometheus_logger._is_invalid_api_key_request(
                status_code=None, exception=exception
            )
            is True
        )

    def test_non_auth_exception_not_detected(self, prometheus_logger):
        exception = ValueError("Some other error")
        assert (
            prometheus_logger._is_invalid_api_key_request(
                status_code=None, exception=exception
            )
            is False
        )


class TestSkipMetricsValidation:
    """Test high-level validation method that orchestrates detection and extraction."""

    def test_skip_for_401_exception(self, prometheus_logger):
        """Test full flow: extraction -> detection -> skip decision."""
        exception = ExceptionWithCode("401")
        assert (
            prometheus_logger._should_skip_metrics_for_invalid_key(exception=exception)
            is True
        )

    def test_skip_for_auth_error_message(self, prometheus_logger):
        """Test full flow: exception message -> detection -> skip decision."""
        exception = AssertionError("expected to start with 'sk-'")
        assert (
            prometheus_logger._should_skip_metrics_for_invalid_key(exception=exception)
            is True
        )

    def test_no_skip_for_valid_request(self, prometheus_logger):
        assert prometheus_logger._should_skip_metrics_for_invalid_key() is False


class TestAsyncHooks:
    """Test how async hook methods treat invalid API key requests."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "exception",
        [
            HTTPException(
                status_code=401,
                detail="LiteLLM Virtual Key expected. Received=nota****tall, expected to start with 'sk-'.",
            ),
            ProxyException(
                message="Authentication Error, Invalid proxy server token passed.",
                type=ProxyErrorTypes.token_not_found_in_db,
                param="key",
                code=401,
            ),
        ],
    )
    async def test_post_call_failure_hook_counts_401_without_key_hash(
        self, prometheus_logger, exception
    ):
        unauthenticated = UserAPIKeyAuth(request_route="/v1/chat/completions")
        unauthenticated.api_key = "notakeyatall"

        with (
            patch.object(
                prometheus_logger, "litellm_proxy_failed_requests_metric"
            ) as mock_failed,
            patch.object(
                prometheus_logger, "litellm_proxy_total_requests_metric"
            ) as mock_total,
        ):
            await prometheus_logger.async_post_call_failure_hook(
                request_data={"model": "test-model"},
                original_exception=exception,
                user_api_key_dict=unauthenticated,
            )

        failed_labels = mock_failed.labels.call_args.kwargs
        assert failed_labels["exception_status"] == "401"
        assert failed_labels["hashed_api_key"] is None
        assert failed_labels["route"] == "/v1/chat/completions"
        mock_failed.labels.return_value.inc.assert_called_once()
        assert mock_total.labels.call_args.kwargs["status_code"] == "401"
        mock_total.labels.return_value.inc.assert_called_once()

    @pytest.mark.asyncio
    async def test_post_call_failure_hook_keeps_resolved_identity_labels_for_401(
        self, prometheus_logger
    ):
        expired_key = UserAPIKeyAuth(
            api_key="sk-expired",
            key_alias="expired-alias",
            team_id="team-1",
        )
        exception = ProxyException(
            message="Authentication Error - Expired Key.",
            type=ProxyErrorTypes.expired_key,
            param="key",
            code=401,
        )

        with patch.object(
            prometheus_logger, "litellm_proxy_failed_requests_metric"
        ) as mock_failed:
            await prometheus_logger.async_post_call_failure_hook(
                request_data={"model": "test-model"},
                original_exception=exception,
                user_api_key_dict=expired_key,
            )

        failed_labels = mock_failed.labels.call_args.kwargs
        assert failed_labels["exception_status"] == "401"
        assert failed_labels["hashed_api_key"] is None
        assert failed_labels["api_key_alias"] == "expired-alias"
        assert failed_labels["team"] == "team-1"

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

        with (
            patch.object(
                prometheus_logger, "litellm_llm_api_failed_requests_metric"
            ) as mock_failed,
            patch.object(
                prometheus_logger, "set_llm_deployment_failure_metrics"
            ) as mock_deployment,
        ):

            await prometheus_logger.async_log_failure_event(
                kwargs=kwargs, response_obj=None, start_time=None, end_time=None
            )

            mock_failed.labels.assert_not_called()
            mock_deployment.assert_not_called()
