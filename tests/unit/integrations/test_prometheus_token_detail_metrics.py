"""
Unit tests for the per-token-type Prometheus detail metrics added for LIT-3220.

These metrics break out cached, cache-creation, audio and reasoning tokens
from the Usage object that providers report. They are sparse — only
incremented when the underlying detail is populated and > 0.

Run with:
    uv run pytest tests/unit/integrations/test_prometheus_token_detail_metrics.py -v
"""

from typing import get_args
from unittest.mock import MagicMock

import pytest

from litellm.types.integrations.prometheus import (
    DEFINED_PROMETHEUS_METRICS,
    PrometheusMetricLabels,
    UserAPIKeyLabelValues,
)


TOKEN_DETAIL_METRICS = [
    "litellm_input_cached_tokens_metric",
    "litellm_input_cache_creation_tokens_metric",
    "litellm_input_audio_tokens_metric",
    "litellm_output_reasoning_tokens_metric",
    "litellm_output_audio_tokens_metric",
]


@pytest.fixture
def sample_enum_values():
    return UserAPIKeyLabelValues(
        end_user="test-end-user",
        hashed_api_key="test-key-hash",
        api_key_alias="test-key-alias",
        team="test-team",
        team_alias="test-team-alias",
        user="test-user",
        model="gpt-4o",
    )


def _make_mock_logger():
    """Mock instance with the five detail counters + get_labels_for_metric."""
    logger = MagicMock()
    for name in TOKEN_DETAIL_METRICS:
        setattr(logger, name, MagicMock())
    logger.get_labels_for_metric = MagicMock(
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
    return logger


class TestTokenDetailMetricsRegistration:
    """Metric registration / wiring — no runtime needed."""

    def test_metrics_in_defined_prometheus_metrics(self):
        defined = get_args(DEFINED_PROMETHEUS_METRICS)
        for name in TOKEN_DETAIL_METRICS:
            assert name in defined, f"{name} missing from DEFINED_PROMETHEUS_METRICS"

    def test_metric_labels_defined(self):
        for name in TOKEN_DETAIL_METRICS:
            assert hasattr(
                PrometheusMetricLabels, name
            ), f"{name} missing from PrometheusMetricLabels"

    def test_input_detail_metrics_share_input_label_set(self):
        # Detail metrics should reuse the parent input/output label set so
        # dashboards can join token totals against per-type detail.
        assert (
            PrometheusMetricLabels.litellm_input_cached_tokens_metric
            == PrometheusMetricLabels.litellm_input_tokens_metric
        )
        assert (
            PrometheusMetricLabels.litellm_input_cache_creation_tokens_metric
            == PrometheusMetricLabels.litellm_input_tokens_metric
        )
        assert (
            PrometheusMetricLabels.litellm_input_audio_tokens_metric
            == PrometheusMetricLabels.litellm_input_tokens_metric
        )

    def test_output_detail_metrics_share_output_label_set(self):
        assert (
            PrometheusMetricLabels.litellm_output_reasoning_tokens_metric
            == PrometheusMetricLabels.litellm_output_tokens_metric
        )
        assert (
            PrometheusMetricLabels.litellm_output_audio_tokens_metric
            == PrometheusMetricLabels.litellm_output_tokens_metric
        )




if __name__ == "__main__":
    pytest.main([__file__, "-v"])
