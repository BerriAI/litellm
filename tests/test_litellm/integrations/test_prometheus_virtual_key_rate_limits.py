import sys
from unittest.mock import MagicMock

from litellm.integrations.prometheus import PrometheusLogger


def test_virtual_key_rate_limit_metrics_read_model_per_key_headers():
    model_group = "openrouter/google/gemini-2.0-flash-001"
    requests_gauge = MagicMock()
    tokens_gauge = MagicMock()
    requests_metric = MagicMock()
    tokens_metric = MagicMock()
    requests_metric.labels.return_value = requests_gauge
    tokens_metric.labels.return_value = tokens_gauge

    logger = PrometheusLogger.__new__(PrometheusLogger)
    logger.litellm_remaining_api_key_requests_for_model = requests_metric
    logger.litellm_remaining_api_key_tokens_for_model = tokens_metric

    logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-key",
        user_api_key_alias="alias",
        kwargs={
            "litellm_params": {
                "metadata": {
                    "model_group": model_group,
                },
            },
            "standard_logging_object": {
                "hidden_params": {
                    "additional_headers": {
                        "x-ratelimit-model_per_key-remaining-requests": 123,
                        "x-ratelimit-model_per_key-remaining-tokens": 456,
                    },
                },
            },
        },
        metadata={
            "model_group": model_group,
        },
        model_id="model-id",
    )

    requests_gauge.set.assert_called_once_with(123)
    tokens_gauge.set.assert_called_once_with(456)
    assert requests_gauge.set.call_args.args[0] != sys.maxsize
    assert tokens_gauge.set.call_args.args[0] != sys.maxsize


def test_virtual_key_rate_limit_metrics_prefer_metadata_values():
    model_group = "openrouter/google/gemini-2.0-flash-001"
    requests_gauge = MagicMock()
    tokens_gauge = MagicMock()
    requests_metric = MagicMock()
    tokens_metric = MagicMock()
    requests_metric.labels.return_value = requests_gauge
    tokens_metric.labels.return_value = tokens_gauge

    logger = PrometheusLogger.__new__(PrometheusLogger)
    logger.litellm_remaining_api_key_requests_for_model = requests_metric
    logger.litellm_remaining_api_key_tokens_for_model = tokens_metric

    logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-key",
        user_api_key_alias="alias",
        kwargs={
            "litellm_params": {
                "metadata": {
                    "model_group": model_group,
                },
            },
            "standard_logging_object": {
                "hidden_params": {
                    "additional_headers": {
                        "x-ratelimit-model_per_key-remaining-requests": 123,
                        "x-ratelimit-model_per_key-remaining-tokens": 456,
                    },
                },
            },
        },
        metadata={
            "model_group": model_group,
            f"litellm-key-remaining-requests-{model_group}": 0,
            f"litellm-key-remaining-tokens-{model_group}": 1,
        },
        model_id="model-id",
    )

    requests_gauge.set.assert_called_once_with(0)
    tokens_gauge.set.assert_called_once_with(1)
