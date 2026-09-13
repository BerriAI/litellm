"""
Unit tests for the video-seconds and images-generated Prometheus counters (LIT-4254).

Video providers report ``duration_seconds`` inside the usage object that lands
on ``standard_logging_payload["metadata"]["usage_object"]``; image generation
calls report ``output_image_count`` there. Both counters are sparse: only
incremented when the value is present and > 0.
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

MEDIA_GENERATION_METRICS = [
    "litellm_video_duration_seconds_metric",
    "litellm_images_generated_metric",
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
        model="sora-2",
    )


def _make_mock_logger():
    logger = MagicMock()
    for name in MEDIA_GENERATION_METRICS:
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


class TestMediaGenerationMetricsRegistration:
    def test_metrics_in_defined_prometheus_metrics(self):
        defined = get_args(DEFINED_PROMETHEUS_METRICS)
        for name in MEDIA_GENERATION_METRICS:
            assert name in defined, f"{name} missing from DEFINED_PROMETHEUS_METRICS"

    def test_metric_labels_defined(self):
        for name in MEDIA_GENERATION_METRICS:
            assert hasattr(PrometheusMetricLabels, name), f"{name} missing from PrometheusMetricLabels"

    def test_metrics_share_output_token_label_set(self):
        assert (
            PrometheusMetricLabels.litellm_video_duration_seconds_metric
            == PrometheusMetricLabels.litellm_output_tokens_metric
        )
        assert (
            PrometheusMetricLabels.litellm_images_generated_metric
            == PrometheusMetricLabels.litellm_output_tokens_metric
        )

    def test_runtime_label_set_matches_output_tokens_metric(self):
        """Full parity with litellm_output_tokens_metric, including the org labels
        appended via _org_label_metrics, so existing token dashboards can be cloned."""
        expected = PrometheusMetricLabels.get_labels("litellm_output_tokens_metric")
        for name in MEDIA_GENERATION_METRICS:
            assert PrometheusMetricLabels.get_labels(name) == expected


class TestIncrementMediaGenerationMetrics:
    def test_video_duration_incremented(self, sample_enum_values):
        logger = _make_mock_logger()
        payload = {"metadata": {"usage_object": {"duration_seconds": 8.0}}}

        PrometheusLogger._increment_media_generation_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=sample_enum_values,
        )

        logger.litellm_video_duration_seconds_metric.labels().inc.assert_called_once_with(8.0)
        logger.litellm_images_generated_metric.labels.assert_not_called()

    def test_image_count_incremented(self, sample_enum_values):
        logger = _make_mock_logger()
        payload = {
            "metadata": {
                "usage_object": {
                    "prompt_tokens": 18,
                    "completion_tokens": 391,
                    "total_tokens": 409,
                    "output_image_count": 2,
                }
            }
        }

        PrometheusLogger._increment_media_generation_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=sample_enum_values,
        )

        logger.litellm_images_generated_metric.labels().inc.assert_called_once_with(2.0)
        logger.litellm_video_duration_seconds_metric.labels.assert_not_called()

    def test_token_only_usage_is_a_noop(self, sample_enum_values):
        logger = _make_mock_logger()
        payload = {
            "metadata": {
                "usage_object": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            }
        }

        PrometheusLogger._increment_media_generation_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=sample_enum_values,
        )

        for name in MEDIA_GENERATION_METRICS:
            getattr(logger, name).labels.assert_not_called()

    @pytest.mark.parametrize("bad_value", [0, 0.0, None, -4.0, "4", True])
    def test_non_positive_or_non_numeric_values_are_ignored(self, sample_enum_values, bad_value):
        logger = _make_mock_logger()
        payload = {
            "metadata": {
                "usage_object": {
                    "duration_seconds": bad_value,
                    "output_image_count": bad_value,
                }
            }
        }

        PrometheusLogger._increment_media_generation_metrics(
            logger,
            standard_logging_payload=payload,
            enum_values=sample_enum_values,
        )

        for name in MEDIA_GENERATION_METRICS:
            getattr(logger, name).labels.assert_not_called()

    def test_missing_usage_object_is_a_noop(self, sample_enum_values):
        logger = _make_mock_logger()

        for payload in ({"metadata": {}}, {"metadata": None}, {"metadata": {"usage_object": "redacted"}}):
            PrometheusLogger._increment_media_generation_metrics(
                logger,
                standard_logging_payload=payload,
                enum_values=sample_enum_values,
            )

        for name in MEDIA_GENERATION_METRICS:
            getattr(logger, name).labels.assert_not_called()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
