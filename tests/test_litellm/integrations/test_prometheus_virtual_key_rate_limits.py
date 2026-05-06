import sys
from unittest.mock import MagicMock

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass
    yield
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def test_virtual_key_rate_limit_metrics_read_model_per_key_headers():
    prometheus_logger = PrometheusLogger()
    requests_metric = MagicMock()
    tokens_metric = MagicMock()
    requests_labeled_metric = MagicMock()
    tokens_labeled_metric = MagicMock()
    requests_metric.labels.return_value = requests_labeled_metric
    tokens_metric.labels.return_value = tokens_labeled_metric
    prometheus_logger.litellm_remaining_api_key_requests_for_model = requests_metric
    prometheus_logger.litellm_remaining_api_key_tokens_for_model = tokens_metric

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-key",
        user_api_key_alias="test-key",
        kwargs={
            "litellm_params": {"metadata": {"model_group": "gpt-4"}},
            "standard_logging_object": {
                "hidden_params": {
                    "additional_headers": {
                        "x-ratelimit-model_per_key-remaining-requests": "12",
                        "x-ratelimit-model_per_key-remaining-tokens": "345",
                    }
                }
            },
        },
        metadata={"model_group": "gpt-4"},
        model_id="model-id",
    )

    requests_labeled_metric.set.assert_called_once_with(12)
    tokens_labeled_metric.set.assert_called_once_with(345)


def test_virtual_key_rate_limit_metrics_preserve_zero_remaining_values():
    prometheus_logger = PrometheusLogger()
    requests_metric = MagicMock()
    tokens_metric = MagicMock()
    requests_labeled_metric = MagicMock()
    tokens_labeled_metric = MagicMock()
    requests_metric.labels.return_value = requests_labeled_metric
    tokens_metric.labels.return_value = tokens_labeled_metric
    prometheus_logger.litellm_remaining_api_key_requests_for_model = requests_metric
    prometheus_logger.litellm_remaining_api_key_tokens_for_model = tokens_metric

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-key",
        user_api_key_alias="test-key",
        kwargs={"litellm_params": {"metadata": {"model_group": "gpt-4"}}},
        metadata={
            "model_group": "gpt-4",
            "litellm-key-remaining-requests-gpt-4": 0,
            "litellm-key-remaining-tokens-gpt-4": 0,
        },
        model_id="model-id",
    )

    requests_labeled_metric.set.assert_called_once_with(0)
    tokens_labeled_metric.set.assert_called_once_with(0)


def test_virtual_key_rate_limit_metrics_default_to_sys_maxsize_when_missing():
    prometheus_logger = PrometheusLogger()
    requests_metric = MagicMock()
    tokens_metric = MagicMock()
    requests_labeled_metric = MagicMock()
    tokens_labeled_metric = MagicMock()
    requests_metric.labels.return_value = requests_labeled_metric
    tokens_metric.labels.return_value = tokens_labeled_metric
    prometheus_logger.litellm_remaining_api_key_requests_for_model = requests_metric
    prometheus_logger.litellm_remaining_api_key_tokens_for_model = tokens_metric

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-key",
        user_api_key_alias="test-key",
        kwargs={"litellm_params": {"metadata": {"model_group": "gpt-4"}}},
        metadata={"model_group": "gpt-4"},
        model_id="model-id",
    )

    requests_labeled_metric.set.assert_called_once_with(sys.maxsize)
    tokens_labeled_metric.set.assert_called_once_with(sys.maxsize)
