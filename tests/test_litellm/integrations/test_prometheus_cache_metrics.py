"""
Unit tests for cache Prometheus metrics.

Run with: uv run pytest tests/test_litellm/integrations/test_prometheus_cache_metrics.py -v
"""

import pytest
from unittest.mock import MagicMock
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
        assert "litellm_provider_cache_read_input_tokens_metric" in defined_metrics
        assert "litellm_provider_cache_creation_input_tokens_metric" in defined_metrics

    def test_cache_metric_labels_defined(self):
        """Test that cache metric labels are properly defined"""
        from litellm.types.integrations.prometheus import PrometheusMetricLabels

        # Verify labels are defined for each cache metric
        assert hasattr(PrometheusMetricLabels, "litellm_cache_hits_metric")
        assert hasattr(PrometheusMetricLabels, "litellm_cache_misses_metric")
        assert hasattr(PrometheusMetricLabels, "litellm_cached_tokens_metric")
        assert hasattr(PrometheusMetricLabels, "litellm_provider_cache_read_input_tokens_metric")
        assert hasattr(
            PrometheusMetricLabels,
            "litellm_provider_cache_creation_input_tokens_metric",
        )

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
            assert label in PrometheusMetricLabels.litellm_provider_cache_read_input_tokens_metric
            assert label in PrometheusMetricLabels.litellm_provider_cache_creation_input_tokens_metric

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
            "metadata": {
                "usage_object": {
                    "cache_read_input_tokens": 25,
                    "cache_creation_input_tokens": 10,
                }
            },
        }

        # Create mock metrics
        mock_logger.litellm_cache_hits_metric = MagicMock()
        mock_logger.litellm_cache_misses_metric = MagicMock()
        mock_logger.litellm_cached_tokens_metric = MagicMock()
        mock_logger.litellm_provider_cache_read_input_tokens_metric = MagicMock()
        mock_logger.litellm_provider_cache_creation_input_tokens_metric = MagicMock()
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
        mock_logger.litellm_cached_tokens_metric.labels().inc.assert_called_once_with(100)

        # Verify cache misses metric was NOT called
        mock_logger.litellm_cache_misses_metric.labels.assert_not_called()

        # Verify provider prompt caching metrics were incremented
        mock_logger.litellm_provider_cache_read_input_tokens_metric.labels().inc.assert_called_once_with(25)
        mock_logger.litellm_provider_cache_creation_input_tokens_metric.labels().inc.assert_called_once_with(10)

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
            "metadata": {
                "usage_object": {
                    # Explicit provider field absent -> fallback should use prompt_tokens_details.cached_tokens
                    "prompt_tokens_details": {"cached_tokens": 20},
                }
            },
        }

        # Create mock metrics
        mock_logger.litellm_cache_hits_metric = MagicMock()
        mock_logger.litellm_cache_misses_metric = MagicMock()
        mock_logger.litellm_cached_tokens_metric = MagicMock()
        mock_logger.litellm_provider_cache_read_input_tokens_metric = MagicMock()
        mock_logger.litellm_provider_cache_creation_input_tokens_metric = MagicMock()
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

        # Provider prompt caching metrics should still be emitted
        mock_logger.litellm_provider_cache_read_input_tokens_metric.labels().inc.assert_called_once_with(20)
        mock_logger.litellm_provider_cache_creation_input_tokens_metric.labels.assert_not_called()

    def test_provider_cache_read_does_not_fallback_on_explicit_zero(self, sample_enum_values):
        """Explicit cache_read_input_tokens=0 must not trigger fallback to cached_tokens."""
        mock_logger = MagicMock()

        from litellm.integrations.prometheus import PrometheusLogger

        standard_logging_payload = {
            "cache_hit": False,
            "total_tokens": 100,
            "prompt_tokens": 50,
            "completion_tokens": 50,
            "model_group": "openai",
            "request_tags": [],
            "metadata": {
                "usage_object": {
                    "cache_read_input_tokens": 0,
                    "prompt_tokens_details": {"cached_tokens": 20},
                }
            },
        }

        mock_logger.litellm_cache_hits_metric = MagicMock()
        mock_logger.litellm_cache_misses_metric = MagicMock()
        mock_logger.litellm_cached_tokens_metric = MagicMock()
        mock_logger.litellm_provider_cache_read_input_tokens_metric = MagicMock()
        mock_logger.litellm_provider_cache_creation_input_tokens_metric = MagicMock()
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

        PrometheusLogger._increment_cache_metrics(
            mock_logger,
            standard_logging_payload=standard_logging_payload,
            enum_values=sample_enum_values,
        )

        # Should not emit read metric, because explicit provider value is zero.
        mock_logger.litellm_provider_cache_read_input_tokens_metric.labels.assert_not_called()

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
            "metadata": {
                "usage_object": {
                    "cache_read_input_tokens": 25,
                }
            },
        }

        # Create mock metrics
        mock_logger.litellm_cache_hits_metric = MagicMock()
        mock_logger.litellm_cache_misses_metric = MagicMock()
        mock_logger.litellm_cached_tokens_metric = MagicMock()
        mock_logger.litellm_provider_cache_read_input_tokens_metric = MagicMock()
        mock_logger.litellm_provider_cache_creation_input_tokens_metric = MagicMock()
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

        # Provider prompt caching metrics should still be emitted
        mock_logger.litellm_provider_cache_read_input_tokens_metric.labels().inc.assert_called_once_with(25)
        mock_logger.litellm_provider_cache_creation_input_tokens_metric.labels.assert_not_called()

    def test_provider_cache_metrics_include_model_group(self):
        """Provider-cache metrics carry both api_provider (inherited) and model_group."""
        from litellm.types.integrations.prometheus import PrometheusMetricLabels

        for metric in (
            "litellm_provider_cache_read_input_tokens_metric",
            "litellm_provider_cache_creation_input_tokens_metric",
        ):
            labels = PrometheusMetricLabels.get_labels(metric)
            assert "model_group" in labels
            assert "api_provider" in labels

        # model_group is scoped to the provider-cache metrics; the LiteLLM
        # response-cache metrics keep their existing label set.
        assert "model_group" not in PrometheusMetricLabels.get_labels("litellm_cache_hits_metric")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
