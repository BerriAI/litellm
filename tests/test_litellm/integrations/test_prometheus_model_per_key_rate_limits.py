import sys
from unittest.mock import MagicMock, patch

from litellm.integrations.prometheus import PrometheusLogger


def _get_prometheus_logger() -> PrometheusLogger:
    with patch(
        "litellm.integrations.prometheus.PrometheusLogger.__init__",
        return_value=None,
    ):
        logger = PrometheusLogger()
    logger.litellm_remaining_api_key_requests_for_model = MagicMock()
    logger.litellm_remaining_api_key_tokens_for_model = MagicMock()
    return logger


def _get_set_values(logger: PrometheusLogger) -> tuple[int, int]:
    remaining_requests = logger.litellm_remaining_api_key_requests_for_model.labels.return_value.set.call_args[
        0
    ][
        0
    ]
    remaining_tokens = logger.litellm_remaining_api_key_tokens_for_model.labels.return_value.set.call_args[
        0
    ][
        0
    ]
    return remaining_requests, remaining_tokens


def test_virtual_key_rate_limit_metrics_use_model_per_key_headers():
    logger = _get_prometheus_logger()

    model_group = "openrouter/google/gemini-2.0-flash-001"
    kwargs = {
        "litellm_params": {"metadata": {"model_group": model_group}},
        "standard_logging_object": {
            "hidden_params": {
                "additional_headers": {
                    "x-ratelimit-model_per_key-remaining-requests": 17,
                    "x-ratelimit-model_per_key-remaining-tokens": 250,
                }
            }
        },
    }

    logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-key",
        user_api_key_alias="alias",
        kwargs=kwargs,
        metadata=kwargs["litellm_params"]["metadata"],
        model_id="model-id",
    )

    remaining_requests, remaining_tokens = _get_set_values(logger)

    assert remaining_requests == 17
    assert remaining_requests != sys.maxsize
    assert remaining_tokens == 250
    assert remaining_tokens != sys.maxsize


def test_virtual_key_rate_limit_metrics_preserve_zero_remaining_values():
    logger = _get_prometheus_logger()

    model_group = "openrouter/google/gemini-2.0-flash-001"
    kwargs = {
        "litellm_params": {"metadata": {"model_group": model_group}},
        "standard_logging_object": {
            "hidden_params": {
                "additional_headers": {
                    "x-ratelimit-model_per_key-remaining-requests": 0,
                    "x-ratelimit-model_per_key-remaining-tokens": 0,
                }
            }
        },
    }

    logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-key",
        user_api_key_alias="alias",
        kwargs=kwargs,
        metadata=kwargs["litellm_params"]["metadata"],
        model_id="model-id",
    )

    remaining_requests, remaining_tokens = _get_set_values(logger)

    assert remaining_requests == 0
    assert remaining_tokens == 0


def test_virtual_key_rate_limit_metrics_prefer_metadata_values():
    logger = _get_prometheus_logger()

    model_group = "openrouter/google/gemini-2.0-flash-001"
    kwargs = {
        "litellm_params": {
            "metadata": {
                "model_group": model_group,
                f"litellm-key-remaining-requests-{model_group}": 8,
                f"litellm-key-remaining-tokens-{model_group}": 64,
            }
        },
        "standard_logging_object": {
            "hidden_params": {
                "additional_headers": {
                    "x-ratelimit-model_per_key-remaining-requests": 17,
                    "x-ratelimit-model_per_key-remaining-tokens": 250,
                }
            }
        },
    }

    logger._set_virtual_key_rate_limit_metrics(
        user_api_key="hashed-key",
        user_api_key_alias="alias",
        kwargs=kwargs,
        metadata=kwargs["litellm_params"]["metadata"],
        model_id="model-id",
    )

    remaining_requests, remaining_tokens = _get_set_values(logger)

    assert remaining_requests == 8
    assert remaining_tokens == 64
