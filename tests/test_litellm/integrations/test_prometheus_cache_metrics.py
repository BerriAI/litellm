"""
Unit tests for cache Prometheus metrics.

Run with: poetry run pytest tests/test_litellm/integrations/test_prometheus_cache_metrics.py -v
"""
import pytest
from unittest.mock import MagicMock, patch
from litellm.types.integrations.prometheus import UserAPIKeyLabelValues


class TestPrometheusCacheMetrics:
    """Tests for cache-related Prometheus metrics"""

    @pytest.fixture
    def sample_enum_values(self):
        """Create sample enum values for labels"""
        return UserAPIKeyLabelValues(
            end_user="test-end-user",
            hashed_api_key="test-key-hash",
            api_key_alias="test-key-alias",
            team="test-team",
            team_alias="test-team-alias",
            user="test-user",
            model="gpt-3.5-turbo",
        )

    def test_cache_metrics_defined_in_types(self):
        """Test that cache metrics are defined in DEFINED_PROMETHEUS_METRICS"""
        from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS
        from typing import get_args

        defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)

        assert "litellm_cache_hits_metric" in defined_metrics
        assert "litellm_cache_misses_metric" in defined_metrics
        assert "litellm_cached_tokens_metric" in defined_metrics

    def test_cache_metric_labels_defined(self):
        """Test that cache metric labels are properly defined"""
        from litellm.types.integrations.prometheus import PrometheusMetricLabels

        # Verify labels are defined for each cache metric
        assert hasattr(PrometheusMetricLabels, "litellm_cache_hits_metric")
        assert hasattr(PrometheusMetricLabels, "litellm_cache_misses_metric")
        assert hasattr(PrometheusMetricLabels, "litellm_cached_tokens_metric")

        # Verify labels include expected keys
        expected_labels = [
            "model",
            "hashed_api_key",
            "api_key_alias",
            "team",
            "team_alias",
            "end_user",
            "user",
        ]
        for label in expected_labels:
            assert label in PrometheusMetricLabels.litellm_cache_hits_metric
            assert label in PrometheusMetricLabels.litellm_cache_misses_metric
            assert label in PrometheusMetricLabels.litellm_cached_tokens_metric

    def test_increment_cache_metrics_on_cache_hit(self, sample_enum_values):
        """Test that cache hit increments the correct metrics"""
        # Create mock for PrometheusLogger instance
        mock_logger = MagicMock()

        # Import the method directly and bind it to our mock
        from litellm.integrations.prometheus import PrometheusLogger

        # Create a mock standard logging payload with cache_hit=True
        standard_logging_payload = {
            "cache_hit": True,
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "model_group": "openai",
            "request_tags": [],
        }

        # Create mock metrics
        mock_logger.litellm_cache_hits_metric = MagicMock()
        mock_logger.litellm_cache_misses_metric = MagicMock()
        mock_logger.litellm_cached_tokens_metric = MagicMock()
        mock_logger.get_labels_for_metric = MagicMock(
            return_value=[
                "model",
                "hashed_api_key",
                "api_key_alias",
                "team",
                "team_alias",
                "end_user",
                "user",
            ]
        )

        # Call the method using unbound method approach
        PrometheusLogger._increment_cache_metrics(
            mock_logger,
            standard_logging_payload=standard_logging_payload,
            enum_values=sample_enum_values,
        )

        # Verify cache hits metric was incremented
        mock_logger.litellm_cache_hits_metric.labels.assert_called()
        mock_logger.litellm_cache_hits_metric.labels().inc.assert_called_once()

        # Verify cached tokens metric was incremented with total_tokens
        mock_logger.litellm_cached_tokens_metric.labels.assert_called()
        mock_logger.litellm_cached_tokens_metric.labels().inc.assert_called_once_with(
            100
        )

        # Verify cache misses metric was NOT called
        mock_logger.litellm_cache_misses_metric.labels.assert_not_called()

    def test_increment_cache_metrics_on_cache_miss(self, sample_enum_values):
        """Test that cache miss increments the correct metrics"""
        # Create mock for PrometheusLogger instance
        mock_logger = MagicMock()

        from litellm.integrations.prometheus import PrometheusLogger

        # Create a mock standard logging payload with cache_hit=False
        standard_logging_payload = {
            "cache_hit": False,
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "model_group": "openai",
            "request_tags": [],
        }

        # Create mock metrics
        mock_logger.litellm_cache_hits_metric = MagicMock()
        mock_logger.litellm_cache_misses_metric = MagicMock()
        mock_logger.litellm_cached_tokens_metric = MagicMock()
        mock_logger.get_labels_for_metric = MagicMock(
            return_value=[
                "model",
                "hashed_api_key",
                "api_key_alias",
                "team",
                "team_alias",
                "end_user",
                "user",
            ]
        )

        # Call the method
        PrometheusLogger._increment_cache_metrics(
            mock_logger,
            standard_logging_payload=standard_logging_payload,
            enum_values=sample_enum_values,
        )

        # Verify cache misses metric was incremented
        mock_logger.litellm_cache_misses_metric.labels.assert_called()
        mock_logger.litellm_cache_misses_metric.labels().inc.assert_called_once()

        # Verify cache hits and cached tokens metrics were NOT called
        mock_logger.litellm_cache_hits_metric.labels.assert_not_called()
        mock_logger.litellm_cached_tokens_metric.labels.assert_not_called()

    def test_increment_cache_metrics_when_cache_hit_is_none(self, sample_enum_values):
        """Test that no metrics are incremented when cache_hit is None"""
        # Create mock for PrometheusLogger instance
        mock_logger = MagicMock()

        from litellm.integrations.prometheus import PrometheusLogger

        # Create a mock standard logging payload with cache_hit=None
        standard_logging_payload = {
            "cache_hit": None,
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "model_group": "openai",
            "request_tags": [],
        }

        # Create mock metrics
        mock_logger.litellm_cache_hits_metric = MagicMock()
        mock_logger.litellm_cache_misses_metric = MagicMock()
        mock_logger.litellm_cached_tokens_metric = MagicMock()
        mock_logger.get_labels_for_metric = MagicMock(
            return_value=[
                "model",
                "hashed_api_key",
                "api_key_alias",
                "team",
                "team_alias",
                "end_user",
                "user",
            ]
        )

        # Call the method
        PrometheusLogger._increment_cache_metrics(
            mock_logger,
            standard_logging_payload=standard_logging_payload,
            enum_values=sample_enum_values,
        )

        # Verify NO metrics were called
        mock_logger.litellm_cache_hits_metric.labels.assert_not_called()
        mock_logger.litellm_cache_misses_metric.labels.assert_not_called()
        mock_logger.litellm_cached_tokens_metric.labels.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
