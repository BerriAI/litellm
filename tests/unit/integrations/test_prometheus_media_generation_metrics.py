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





if __name__ == "__main__":
    pytest.main([__file__, "-v"])
