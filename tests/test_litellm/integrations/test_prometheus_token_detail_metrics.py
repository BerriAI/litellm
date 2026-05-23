"""
Unit tests for the per-token-type Prometheus detail metrics added for LIT-3220.

These metrics break out cached, cache-creation, audio and reasoning tokens
from the Usage object that providers report. They are sparse — only
incremented when the underlying detail is populated and > 0.

Run with:
    uv run pytest tests/test_litellm/integrations/test_prometheus_token_detail_metrics.py -v
"""

from typing import get_args
from unittest.mock import MagicMock

import pytest

from litellm.integrations.prometheus import PrometheusLogger
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


class TestIncrementTokenDetailMetrics:
    """Behaviour of PrometheusLogger._increment_token_detail_metrics."""

    def test_increments_all_present_token_types(self, sample_enum_values):
        logger = _make_mock_logger()
        payload = {
            "metadata": {
                "usage_object": {
                    "prompt_tokens": 100,
                    "completion_tokens": 80,
                    "total_tokens": 180,
                    "prompt_tokens_details": {
                        "cached_tokens": 40,
                        "cache_creation_tokens": 25,
                        "audio_tokens": 15,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 60,
                        "audio_tokens": 10,
                    },
                }
            },
        }

        PrometheusLogger._increment_token_detail_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=sample_enum_values,
        )

        logger.litellm_input_cached_tokens_metric.labels().inc.assert_called_once_with(
            40.0
        )
        logger.litellm_input_cache_creation_tokens_metric.labels().inc.assert_called_once_with(
            25.0
        )
        logger.litellm_input_audio_tokens_metric.labels().inc.assert_called_once_with(
            15.0
        )
        logger.litellm_output_reasoning_tokens_metric.labels().inc.assert_called_once_with(
            60.0
        )
        logger.litellm_output_audio_tokens_metric.labels().inc.assert_called_once_with(
            10.0
        )

    def test_skips_metrics_when_value_is_zero(self, sample_enum_values):
        logger = _make_mock_logger()
        payload = {
            "metadata": {
                "usage_object": {
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                        "cache_creation_tokens": 0,
                        "audio_tokens": 0,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "audio_tokens": 0,
                    },
                }
            }
        }

        PrometheusLogger._increment_token_detail_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=sample_enum_values,
        )

        for name in TOKEN_DETAIL_METRICS:
            getattr(logger, name).labels.assert_not_called()

    def test_skips_metrics_when_value_is_none(self, sample_enum_values):
        logger = _make_mock_logger()
        payload = {
            "metadata": {
                "usage_object": {
                    "prompt_tokens_details": {
                        "cached_tokens": None,
                        "audio_tokens": 12,
                    },
                    "completion_tokens_details": {},
                }
            }
        }

        PrometheusLogger._increment_token_detail_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=sample_enum_values,
        )

        # Only audio_tokens was non-zero — only that counter should fire.
        logger.litellm_input_cached_tokens_metric.labels.assert_not_called()
        logger.litellm_input_cache_creation_tokens_metric.labels.assert_not_called()
        logger.litellm_input_audio_tokens_metric.labels().inc.assert_called_once_with(
            12.0
        )
        logger.litellm_output_reasoning_tokens_metric.labels.assert_not_called()
        logger.litellm_output_audio_tokens_metric.labels.assert_not_called()

    def test_no_usage_object_is_a_noop(self, sample_enum_values):
        logger = _make_mock_logger()
        payload = {"metadata": {}}

        # Should not raise and should not call any counter.
        PrometheusLogger._increment_token_detail_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=sample_enum_values,
        )

        for name in TOKEN_DETAIL_METRICS:
            getattr(logger, name).labels.assert_not_called()

    def test_missing_metadata_is_a_noop(self, sample_enum_values):
        logger = _make_mock_logger()

        # Many error / cache-hit paths leave metadata as None.
        PrometheusLogger._increment_token_detail_metrics(
            logger,
            standard_logging_payload={"metadata": None},  # type: ignore[typeddict-item]
            enum_values=sample_enum_values,
        )

        for name in TOKEN_DETAIL_METRICS:
            getattr(logger, name).labels.assert_not_called()

    def test_negative_values_are_ignored(self, sample_enum_values):
        # Defensive: a buggy upstream that returned a negative shouldn't
        # poison the counter (counters can't go down without a reset).
        logger = _make_mock_logger()
        payload = {
            "metadata": {
                "usage_object": {
                    "prompt_tokens_details": {"cached_tokens": -5},
                    "completion_tokens_details": {"reasoning_tokens": -10},
                }
            }
        }

        PrometheusLogger._increment_token_detail_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=sample_enum_values,
        )

        logger.litellm_input_cached_tokens_metric.labels.assert_not_called()
        logger.litellm_output_reasoning_tokens_metric.labels.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
