"""
Unit tests for prometheus queue time and guardrail metrics
"""

from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger


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






class TestPrometheusGuardrailMetrics:
    """Test guardrail metrics recording"""




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

