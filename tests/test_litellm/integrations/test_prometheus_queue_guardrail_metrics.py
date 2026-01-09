"""
Unit tests for prometheus queue time and guardrail metrics
"""
from datetime import datetime
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    """Clean up prometheus registry between tests"""
    # Clear the registry before each test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    yield
    # Clean up after test
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


class TestPrometheusQueueTimeMetric:
    """Test request queue time metric recording"""

    def test_queue_time_metric_recorded_in_set_latency_metrics(self):
        """Test that queue time metric is recorded when queue_time_seconds is present in metadata"""
        # Arrange
        prometheus_logger = PrometheusLogger()

        # Mock the metric
        mock_metric = MagicMock()
        mock_labeled_metric = MagicMock()
        mock_metric.labels.return_value = mock_labeled_metric
        prometheus_logger.litellm_request_queue_time_metric = mock_metric

        # Create mock kwargs with queue_time_seconds in metadata
        queue_time_seconds = 0.5

        kwargs = {
            "litellm_params": {"metadata": {"queue_time_seconds": queue_time_seconds}},
            "model": "gpt-3.5-turbo",
            "start_time": datetime.now(),
            "end_time": datetime.now(),
        }

        enum_values = UserAPIKeyLabelValues(
            end_user=None,
            hashed_api_key="test-key",
            api_key_alias="test-alias",
            requested_model="gpt-3.5-turbo",
            model_group="gpt-3.5-turbo",
            team=None,
            team_alias=None,
            user=None,
            user_email=None,
            status_code="200",
            model="gpt-3.5-turbo",
            litellm_model_name="gpt-3.5-turbo",
            tags=[],
            model_id="gpt-3.5-turbo",
            api_base="https://api.openai.com",
            api_provider="openai",
            exception_status=None,
            exception_class=None,
            custom_metadata_labels={},
            route=None,
        )

        # Act
        prometheus_logger._set_latency_metrics(
            kwargs=kwargs,
            model="gpt-3.5-turbo",
            user_api_key="test-key",
            user_api_key_alias="test-alias",
            user_api_team=None,
            user_api_team_alias=None,
            enum_values=enum_values,
        )

        # Assert - queue time metric should be called
        mock_metric.labels.assert_called()
        # Check that observe was called on the queue time metric
        assert mock_labeled_metric.observe.called
        # Verify the observed value
        observed_value = None
        for call in mock_labeled_metric.observe.call_args_list:
            if len(call[0]) > 0:
                observed_value = call[0][0]
                if observed_value == queue_time_seconds:
                    break
        assert observed_value == queue_time_seconds
        assert observed_value >= 0

    def test_queue_time_metric_not_recorded_when_missing(self):
        """Test that queue time metric is not recorded when queue_time_seconds is missing"""
        # Arrange
        prometheus_logger = PrometheusLogger()

        # Mock the metric
        mock_metric = MagicMock()
        mock_labeled_metric = MagicMock()
        mock_metric.labels.return_value = mock_labeled_metric
        prometheus_logger.litellm_request_queue_time_metric = mock_metric

        # Create mock kwargs without queue_time_seconds
        kwargs = {
            "litellm_params": {"metadata": {}},
            "model": "gpt-3.5-turbo",
            "start_time": datetime.now(),
            "end_time": datetime.now(),
        }

        enum_values = UserAPIKeyLabelValues(
            end_user=None,
            hashed_api_key="test-key",
            api_key_alias="test-alias",
            requested_model="gpt-3.5-turbo",
            model_group="gpt-3.5-turbo",
            team=None,
            team_alias=None,
            user=None,
            user_email=None,
            status_code="200",
            model="gpt-3.5-turbo",
            litellm_model_name="gpt-3.5-turbo",
            tags=[],
            model_id="gpt-3.5-turbo",
            api_base="https://api.openai.com",
            api_provider="openai",
            exception_status=None,
            exception_class=None,
            custom_metadata_labels={},
            route=None,
        )

        # Act
        prometheus_logger._set_latency_metrics(
            kwargs=kwargs,
            model="gpt-3.5-turbo",
            user_api_key="test-key",
            user_api_key_alias="test-alias",
            user_api_team=None,
            user_api_team_alias=None,
            enum_values=enum_values,
        )

        # Assert - queue time metric should not be called (queue_time_seconds is None)
        # We check that observe was not called with queue_time_seconds
        queue_time_called = False
        for call in mock_labeled_metric.observe.call_args_list:
            if len(call[0]) > 0 and call[0][0] == 0.5:  # Our test queue time value
                queue_time_called = True
                break
        assert (
            not queue_time_called
        ), "Queue time metric should not be recorded when queue_time_seconds is missing"

    def test_queue_time_metric_not_recorded_when_negative(self):
        """Test that queue time metric is not recorded when queue_time_seconds is negative"""
        # Arrange
        prometheus_logger = PrometheusLogger()

        # Mock the metric
        mock_metric = MagicMock()
        mock_labeled_metric = MagicMock()
        mock_metric.labels.return_value = mock_labeled_metric
        prometheus_logger.litellm_request_queue_time_metric = mock_metric

        # Create mock kwargs with negative queue_time_seconds
        kwargs = {
            "litellm_params": {
                "metadata": {"queue_time_seconds": -0.1}  # Negative value
            },
            "model": "gpt-3.5-turbo",
            "start_time": datetime.now(),
            "end_time": datetime.now(),
        }

        enum_values = UserAPIKeyLabelValues(
            end_user=None,
            hashed_api_key="test-key",
            api_key_alias="test-alias",
            requested_model="gpt-3.5-turbo",
            model_group="gpt-3.5-turbo",
            team=None,
            team_alias=None,
            user=None,
            user_email=None,
            status_code="200",
            model="gpt-3.5-turbo",
            litellm_model_name="gpt-3.5-turbo",
            tags=[],
            model_id="gpt-3.5-turbo",
            api_base="https://api.openai.com",
            api_provider="openai",
            exception_status=None,
            exception_class=None,
            custom_metadata_labels={},
            route=None,
        )

        # Act
        prometheus_logger._set_latency_metrics(
            kwargs=kwargs,
            model="gpt-3.5-turbo",
            user_api_key="test-key",
            user_api_key_alias="test-alias",
            user_api_team=None,
            user_api_team_alias=None,
            enum_values=enum_values,
        )

        # Assert - queue time metric should not be called for negative values
        # We check that observe was not called with the negative value
        negative_value_called = False
        for call in mock_labeled_metric.observe.call_args_list:
            if len(call[0]) > 0 and call[0][0] == -0.1:
                negative_value_called = True
                break
        assert (
            not negative_value_called
        ), "Queue time metric should not be recorded for negative values"


class TestPrometheusGuardrailMetrics:
    """Test guardrail metrics recording"""

    def test_record_guardrail_metrics_success(self):
        """Test recording guardrail metrics for successful execution"""
        # Arrange
        prometheus_logger = PrometheusLogger()

        # Mock metrics
        mock_latency_metric = MagicMock()
        mock_requests_metric = MagicMock()
        mock_errors_metric = MagicMock()

        prometheus_logger.litellm_guardrail_latency_metric = mock_latency_metric
        prometheus_logger.litellm_guardrail_requests_total = mock_requests_metric
        prometheus_logger.litellm_guardrail_errors_total = mock_errors_metric

        guardrail_name = "test_guardrail"
        latency_seconds = 0.15
        status = "success"
        error_type = None
        hook_type = "pre_call"

        # Act
        prometheus_logger._record_guardrail_metrics(
            guardrail_name=guardrail_name,
            latency_seconds=latency_seconds,
            status=status,
            error_type=error_type,
            hook_type=hook_type,
        )

        # Assert - latency metric should be recorded
        mock_latency_metric.labels.assert_called_once_with(
            guardrail_name=guardrail_name,
            status=status,
            error_type="none",
            hook_type=hook_type,
        )
        mock_latency_metric.labels.return_value.observe.assert_called_once_with(
            latency_seconds
        )

        # Assert - requests metric should be incremented
        mock_requests_metric.labels.assert_called_once_with(
            guardrail_name=guardrail_name,
            status=status,
            hook_type=hook_type,
        )
        mock_requests_metric.labels.return_value.inc.assert_called_once()

        # Assert - errors metric should NOT be called for success
        mock_errors_metric.labels.assert_not_called()

    def test_record_guardrail_metrics_error(self):
        """Test recording guardrail metrics for failed execution"""
        # Arrange
        prometheus_logger = PrometheusLogger()

        # Mock metrics
        mock_latency_metric = MagicMock()
        mock_requests_metric = MagicMock()
        mock_errors_metric = MagicMock()

        prometheus_logger.litellm_guardrail_latency_metric = mock_latency_metric
        prometheus_logger.litellm_guardrail_requests_total = mock_requests_metric
        prometheus_logger.litellm_guardrail_errors_total = mock_errors_metric

        guardrail_name = "test_guardrail"
        latency_seconds = 0.2
        status = "error"
        error_type = "ValueError"
        hook_type = "pre_call"

        # Act
        prometheus_logger._record_guardrail_metrics(
            guardrail_name=guardrail_name,
            latency_seconds=latency_seconds,
            status=status,
            error_type=error_type,
            hook_type=hook_type,
        )

        # Assert - latency metric should be recorded
        mock_latency_metric.labels.assert_called_once_with(
            guardrail_name=guardrail_name,
            status=status,
            error_type=error_type,
            hook_type=hook_type,
        )
        mock_latency_metric.labels.return_value.observe.assert_called_once_with(
            latency_seconds
        )

        # Assert - requests metric should be incremented
        mock_requests_metric.labels.assert_called_once_with(
            guardrail_name=guardrail_name,
            status=status,
            hook_type=hook_type,
        )
        mock_requests_metric.labels.return_value.inc.assert_called_once()

        # Assert - errors metric should be incremented
        mock_errors_metric.labels.assert_called_once_with(
            guardrail_name=guardrail_name,
            error_type=error_type,
            hook_type=hook_type,
        )
        mock_errors_metric.labels.return_value.inc.assert_called_once()

    def test_record_guardrail_metrics_during_call_hook(self):
        """Test recording guardrail metrics for during_call hook"""
        # Arrange
        prometheus_logger = PrometheusLogger()

        # Mock metrics
        mock_latency_metric = MagicMock()
        mock_requests_metric = MagicMock()

        prometheus_logger.litellm_guardrail_latency_metric = mock_latency_metric
        prometheus_logger.litellm_guardrail_requests_total = mock_requests_metric

        guardrail_name = "moderation_guardrail"
        latency_seconds = 0.1
        status = "success"
        hook_type = "during_call"

        # Act
        prometheus_logger._record_guardrail_metrics(
            guardrail_name=guardrail_name,
            latency_seconds=latency_seconds,
            status=status,
            error_type=None,
            hook_type=hook_type,
        )

        # Assert - hook_type should be "during_call"
        mock_latency_metric.labels.assert_called_once()
        call_kwargs = mock_latency_metric.labels.call_args[1]
        assert call_kwargs["hook_type"] == "during_call"

    def test_record_guardrail_metrics_handles_exception(self):
        """Test that _record_guardrail_metrics handles exceptions gracefully"""
        # Arrange
        prometheus_logger = PrometheusLogger()

        # Mock metric to raise exception
        mock_metric = MagicMock()
        mock_metric.labels.side_effect = Exception("Test error")
        prometheus_logger.litellm_guardrail_latency_metric = mock_metric
        prometheus_logger.litellm_guardrail_requests_total = MagicMock()

        # Act & Assert - should not raise exception
        try:
            prometheus_logger._record_guardrail_metrics(
                guardrail_name="test",
                latency_seconds=0.1,
                status="success",
                error_type=None,
                hook_type="pre_call",
            )
        except Exception:
            pytest.fail("_record_guardrail_metrics should handle exceptions gracefully")

    def test_record_guardrail_metrics_with_guardrail_name_attribute(self):
        """Test that guardrail name is extracted from guardrail_name attribute if available"""
        # Arrange
        prometheus_logger = PrometheusLogger()

        # Mock metrics
        mock_latency_metric = MagicMock()
        mock_requests_metric = MagicMock()

        prometheus_logger.litellm_guardrail_latency_metric = mock_latency_metric
        prometheus_logger.litellm_guardrail_requests_total = mock_requests_metric

        guardrail_name = "custom_guardrail_name"
        latency_seconds = 0.1
        status = "success"
        hook_type = "pre_call"

        # Act
        prometheus_logger._record_guardrail_metrics(
            guardrail_name=guardrail_name,
            latency_seconds=latency_seconds,
            status=status,
            error_type=None,
            hook_type=hook_type,
        )

        # Assert - guardrail_name should be used
        mock_latency_metric.labels.assert_called_once()
        call_kwargs = mock_latency_metric.labels.call_args[1]
        assert call_kwargs["guardrail_name"] == guardrail_name
