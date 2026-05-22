import logging
import sys

import pytest
from prometheus_client import REGISTRY

import litellm
from litellm.integrations.prometheus import PrometheusLogger


def _clear_prometheus_registry() -> None:
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


def _create_prometheus_logger_with_custom_labels(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        litellm,
        "custom_prometheus_metadata_labels",
        ["metadata.department", "metadata.environment"],
    )
    _clear_prometheus_registry()
    return PrometheusLogger()


def _standard_logging_payload_with_requester_metadata() -> dict:
    return {
        "model_id": "model-123",
        "model_group": "gpt-4o-mini",
        "api_base": "https://api.openai.com",
        "custom_llm_provider": "openai",
        "metadata": {
            "user_api_key_hash": "test-hash",
            "user_api_key_alias": "test-alias",
            "user_api_key_team_id": "test-team",
            "user_api_key_team_alias": "test-team-alias",
            "user_api_key_user_id": "test-user",
            "user_api_key_user_email": "test@example.com",
            "user_api_key_org_id": None,
            "requester_metadata": {
                "department": "engineering",
                "environment": "production",
            },
            "user_api_key_auth_metadata": None,
            "spend_logs_metadata": None,
        },
        "request_tags": [],
        "completion_tokens": 0,
        "total_tokens": 0,
        "response_cost": 0,
    }


def _metric_samples(metric_name: str):
    return [
        sample
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name == metric_name
    ]


@pytest.mark.asyncio
async def test_async_log_failure_event_accepts_custom_metadata_labels(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
):
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    kwargs = {
        "model": "gpt-4o-mini",
        "litellm_params": {
            "metadata": {
                "user_api_key_end_user_id": "test-end-user",
            }
        },
        "standard_logging_object": _standard_logging_payload_with_requester_metadata(),
    }

    with caplog.at_level(logging.ERROR):
        await prometheus_logger.async_log_failure_event(
            kwargs=kwargs,
            response_obj=None,
            start_time=None,
            end_time=None,
        )

    assert "Incorrect label count" not in caplog.text
    samples = _metric_samples("litellm_llm_api_failed_requests_metric_total")
    assert any(
        sample.labels.get("metadata_department") == "engineering"
        and sample.labels.get("metadata_environment") == "production"
        for sample in samples
    )


def test_virtual_key_rate_limit_metrics_accept_custom_metadata_labels(
    monkeypatch: pytest.MonkeyPatch,
):
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    metadata = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 3,
        "litellm-key-remaining-tokens-gpt-4o-mini": 200,
    }
    kwargs = {
        "litellm_params": {
            "metadata": metadata,
        },
        "standard_logging_object": _standard_logging_payload_with_requester_metadata(),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    assert any(
        sample.labels.get("metadata_department") == "engineering"
        and sample.labels.get("metadata_environment") == "production"
        and sample.value == 3
        for sample in samples
    )


def test_virtual_key_rate_limit_metrics_preserve_zero_remaining_values(
    monkeypatch: pytest.MonkeyPatch,
):
    prometheus_logger = _create_prometheus_logger_with_custom_labels(monkeypatch)
    metadata = {
        "model_group": "gpt-4o-mini",
        "litellm-key-remaining-requests-gpt-4o-mini": 0,
        "litellm-key-remaining-tokens-gpt-4o-mini": 0,
    }
    kwargs = {
        "litellm_params": {
            "metadata": metadata,
        },
        "standard_logging_object": _standard_logging_payload_with_requester_metadata(),
    }

    prometheus_logger._set_virtual_key_rate_limit_metrics(
        user_api_key="test-hash",
        user_api_key_alias="test-alias",
        kwargs=kwargs,
        metadata=metadata,
        model_id="model-123",
    )

    request_samples = _metric_samples("litellm_remaining_api_key_requests_for_model")
    token_samples = _metric_samples("litellm_remaining_api_key_tokens_for_model")

    assert any(sample.value == 0 for sample in request_samples)
    assert any(sample.value == 0 for sample in token_samples)
    assert not any(sample.value == sys.maxsize for sample in request_samples)
    assert not any(sample.value == sys.maxsize for sample in token_samples)
