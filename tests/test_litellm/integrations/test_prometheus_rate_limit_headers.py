import sys
from typing import Dict, Optional

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    yield
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


def _get_metric_value(metric_name: str, labels: Dict[str, Optional[str]]) -> float:
    for metric in REGISTRY.collect():
        for sample in metric.samples:
            if sample.name != metric_name:
                continue
            if all(sample.labels.get(key) == value for key, value in labels.items()):
                return sample.value

    raise AssertionError(f"Unable to find {metric_name} with labels {labels}")


def test_virtual_key_rate_limit_metrics_read_model_per_key_remaining_headers():
    prometheus_logger = PrometheusLogger()
    model_group = "gpt-4o-mini"
    model_id = "model-id-123"
    user_api_key = "hashed-key"
    user_api_key_alias = "test-key"
    metadata = {"model_group": model_group}

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key=user_api_key,
        user_api_key_alias=user_api_key_alias,
        kwargs={
            "litellm_params": {"metadata": metadata},
            "standard_logging_object": {
                "hidden_params": {
                    "additional_headers": {
                        "x-ratelimit-model_per_key-remaining-requests": 17,
                        "x-ratelimit-model_per_key-remaining-tokens": 329,
                    }
                }
            },
        },
        metadata=metadata,
        model_id=model_id,
    )

    labels = {
        "hashed_api_key": user_api_key,
        "api_key_alias": user_api_key_alias,
        "model": model_group,
        "model_id": model_id,
    }

    assert (
        _get_metric_value("litellm_remaining_api_key_requests_for_model", labels) == 17
    )
    assert (
        _get_metric_value("litellm_remaining_api_key_tokens_for_model", labels) == 329
    )
    assert (
        _get_metric_value("litellm_remaining_api_key_requests_for_model", labels)
        != sys.maxsize
    )


def test_virtual_key_rate_limit_metrics_prefer_metadata_over_remaining_headers():
    prometheus_logger = PrometheusLogger()
    model_group = "gpt-4o-mini"
    model_id = "model-id-123"
    user_api_key = "hashed-key"
    user_api_key_alias = "test-key"
    metadata = {
        "model_group": model_group,
        f"litellm-key-remaining-requests-{model_group}": 11,
        f"litellm-key-remaining-tokens-{model_group}": 22,
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key=user_api_key,
        user_api_key_alias=user_api_key_alias,
        kwargs={
            "litellm_params": {"metadata": metadata},
            "standard_logging_object": {
                "hidden_params": {
                    "additional_headers": {
                        "x-ratelimit-model_per_key-remaining-requests": 17,
                        "x-ratelimit-model_per_key-remaining-tokens": 329,
                    }
                }
            },
        },
        metadata=metadata,
        model_id=model_id,
    )

    labels = {
        "hashed_api_key": user_api_key,
        "api_key_alias": user_api_key_alias,
        "model": model_group,
        "model_id": model_id,
    }

    assert (
        _get_metric_value("litellm_remaining_api_key_requests_for_model", labels) == 11
    )
    assert _get_metric_value("litellm_remaining_api_key_tokens_for_model", labels) == 22
