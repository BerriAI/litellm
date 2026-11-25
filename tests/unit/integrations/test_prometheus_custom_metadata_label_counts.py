import logging
import sys
from collections.abc import Awaitable, Callable, Iterator, Mapping
from typing import Final, TypeAlias

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
    return [sample for metric in REGISTRY.collect() for sample in metric.samples if sample.name == metric_name]


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


_FallbackEvent: TypeAlias = Callable[[PrometheusLogger, str, dict[str, object], Exception], Awaitable[None]]
_FALLBACK_METADATA_LABELS: Final = (
    "metadata.caller_agent",
    "metadata.owner",
    "metadata.org.unit",
    "metadata.user_api_key_team_alias",
    "metadata.is_silent_experiment",
)
_FALLBACK_EVENTS: Final = (
    pytest.param(
        PrometheusLogger.log_success_fallback_event, "litellm_deployment_successful_fallbacks_total", id="success"
    ),
    pytest.param(
        PrometheusLogger.log_failure_fallback_event, "litellm_deployment_failed_fallbacks_total", id="failure"
    ),
)


@pytest.fixture
def fallback_logger(monkeypatch: pytest.MonkeyPatch) -> Iterator[PrometheusLogger]:
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", list(_FALLBACK_METADATA_LABELS))
    monkeypatch.setattr(litellm, "custom_prometheus_tags", ["prod"])
    _clear_prometheus_registry()
    yield PrometheusLogger()
    _clear_prometheus_registry()


async def _log_fallback(logger: PrometheusLogger, event: _FallbackEvent, kwargs: Mapping[str, object]) -> None:
    exception: Final = litellm.RateLimitError("boom", llm_provider="openai", model="model-a")
    await event(logger, "model-a", dict(kwargs), exception)


def _only_sample_labels(metric_name: str) -> Mapping[str, str]:
    samples: Final = _metric_samples(metric_name)
    assert [sample.value for sample in samples] == [1.0]
    return samples[0].labels


_UNSET: Final = {
    "metadata_caller_agent": "None",
    "metadata_owner": "None",
    "metadata_org_unit": "None",
    "metadata_user_api_key_team_alias": "None",
    "metadata_is_silent_experiment": "None",
}


def _custom_label_values(labels: Mapping[str, str]) -> Mapping[str, str]:
    return {name: labels[name] for name in _UNSET}


@pytest.mark.parametrize(("event", "metric_name"), _FALLBACK_EVENTS)
@pytest.mark.parametrize(
    ("metadata", "expected"),
    [
        pytest.param(
            {"spend_logs_metadata": {"caller_agent": "triage"}},
            {**_UNSET, "metadata_caller_agent": "triage"},
            id="spend-logs-metadata-header",
        ),
        pytest.param(
            {
                "user_api_key_team_alias": "team-a",
                "requester_metadata": {"owner": "requester"},
                "user_api_key_auth_metadata": {"caller_agent": "auth"},
                "spend_logs_metadata": {"org": {"unit": "search"}},
            },
            {
                **_UNSET,
                "metadata_caller_agent": "auth",
                "metadata_owner": "requester",
                "metadata_org_unit": "search",
                "metadata_user_api_key_team_alias": "team-a",
            },
            id="every-source-and-dotted-key",
        ),
        pytest.param(
            {
                "requester_metadata": {"owner": "requester"},
                "user_api_key_auth_metadata": {"owner": "auth"},
                "spend_logs_metadata": {"owner": "spend"},
            },
            {**_UNSET, "metadata_owner": "spend"},
            id="spend-logs-wins",
        ),
        pytest.param(
            {"requester_metadata": {"owner": "requester"}, "user_api_key_auth_metadata": {"owner": "auth"}},
            {**_UNSET, "metadata_owner": "auth"},
            id="auth-beats-requester",
        ),
        pytest.param({"metadata": {"owner": "sdk"}}, {**_UNSET, "metadata_owner": "sdk"}, id="sdk-nested-metadata"),
        pytest.param({"caller_agent": "raw"}, _UNSET, id="unknown-top-level-key"),
        pytest.param({"spend_logs_metadata": None}, _UNSET, id="null-spend-logs"),
        pytest.param({"spend_logs_metadata": '{"caller_agent": "triage"}'}, _UNSET, id="string-spend-logs"),
        pytest.param({"spend_logs_metadata": ["triage"]}, _UNSET, id="list-spend-logs"),
        pytest.param({"spend_logs_metadata": {"caller_agent": 7}}, _UNSET, id="non-string-value"),
        pytest.param({"spend_logs_metadata": {"org": "flat"}}, _UNSET, id="dotted-key-through-scalar"),
        pytest.param(
            {"is_silent_experiment": True, "spend_logs_metadata": {"is_silent_experiment": "true"}},
            _UNSET,
            id="client-cannot-set-silent-experiment",
        ),
    ],
)
@pytest.mark.asyncio
async def test_fallback_metrics_label_custom_metadata_like_request_metrics(
    fallback_logger: PrometheusLogger,
    event: _FallbackEvent,
    metric_name: str,
    metadata: Mapping[str, object],
    expected: Mapping[str, str],
):
    await _log_fallback(fallback_logger, event, {"model": "model-b", "metadata": metadata})

    assert _custom_label_values(_only_sample_labels(metric_name)) == expected


@pytest.mark.parametrize(("event", "metric_name"), _FALLBACK_EVENTS)
@pytest.mark.asyncio
async def test_fallback_metrics_keep_builtin_labels_and_tags_next_to_custom_metadata(
    fallback_logger: PrometheusLogger, event: _FallbackEvent, metric_name: str
):
    kwargs: Final = {
        "model": "model-b",
        "tags": ["prod"],
        "metadata": {
            "user_api_key_alias": "key-alias",
            "user_api_key_team_id": "team-1",
            "spend_logs_metadata": {"caller_agent": "triage"},
        },
    }

    await _log_fallback(fallback_logger, event, kwargs)

    labels: Final = _only_sample_labels(metric_name)
    assert labels["metadata_caller_agent"] == "triage"
    assert labels["tag_prod"] == "true"
    assert labels["fallback_model"] == "model-b"
    assert labels["api_key_alias"] == "key-alias"
    assert labels["team"] == "team-1"
    assert labels["exception_status"] == "429"


@pytest.mark.parametrize(
    ("metadata_labels", "tags", "expected_custom_labels"),
    [
        (("metadata.department",), (), {"metadata_department": "None"}),
        ((), ("prod",), {"tag_prod": "None"}),
        (("metadata.department",), ("prod",), {"metadata_department": "None", "tag_prod": "None"}),
    ],
    ids=["metadata-only", "tags-only", "metadata-and-tags"],
)
def test_cooldown_counter_increments_with_custom_labels(
    monkeypatch: pytest.MonkeyPatch,
    metadata_labels: tuple[str, ...],
    tags: tuple[str, ...],
    expected_custom_labels: Mapping[str, str],
) -> None:
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", list(metadata_labels))
    monkeypatch.setattr(litellm, "custom_prometheus_tags", list(tags))
    previous_collectors: Final = tuple(REGISTRY._collector_to_names)
    _clear_prometheus_registry()
    try:
        logger: Final = PrometheusLogger()
        deployment_labels: Final = {
            "litellm_model_name": "test-model",
            "model_id": "model-123",
            "api_base": "https://example.com",
            "api_provider": "openai",
            "exception_status": "429",
        }
        logger.increment_deployment_cooled_down(**deployment_labels)

        samples: Final = _metric_samples("litellm_deployment_cooled_down_total")
        assert [(sample.labels, sample.value) for sample in samples] == [
            ({**deployment_labels, **expected_custom_labels}, 1)
        ]
    finally:
        _clear_prometheus_registry()
        for collector in previous_collectors:
            REGISTRY.register(collector)
    assert labels["exception_class"] == "Openai.RateLimitError"


@pytest.mark.parametrize(("event", "metric_name"), _FALLBACK_EVENTS)
@pytest.mark.asyncio
async def test_fallback_metrics_read_litellm_metadata_when_present(
    fallback_logger: PrometheusLogger, event: _FallbackEvent, metric_name: str
):
    kwargs: Final = {
        "model": "model-b",
        "metadata": {"spend_logs_metadata": {"caller_agent": "openai-metadata"}},
        "litellm_metadata": {"spend_logs_metadata": {"caller_agent": "litellm-metadata"}},
    }

    await _log_fallback(fallback_logger, event, kwargs)

    assert _only_sample_labels(metric_name)["metadata_caller_agent"] == "litellm-metadata"


@pytest.mark.parametrize(("event", "metric_name"), _FALLBACK_EVENTS)
@pytest.mark.asyncio
async def test_fallback_metrics_do_not_mutate_the_request_metadata(
    fallback_logger: PrometheusLogger, event: _FallbackEvent, metric_name: str
):
    spend_logs_metadata: Final = {"caller_agent": "triage", "org": {"unit": "search"}}
    metadata: Final = {"requester_metadata": {"owner": "requester"}, "spend_logs_metadata": spend_logs_metadata}

    await _log_fallback(fallback_logger, event, {"model": "model-b", "metadata": metadata})

    assert _only_sample_labels(metric_name)["metadata_caller_agent"] == "triage"
    assert metadata == {
        "requester_metadata": {"owner": "requester"},
        "spend_logs_metadata": {"caller_agent": "triage", "org": {"unit": "search"}},
    }


@pytest.mark.parametrize(("event", "metric_name"), _FALLBACK_EVENTS)
@pytest.mark.asyncio
async def test_fallback_metrics_without_custom_metadata_labels_emit_no_metadata_labels(
    monkeypatch: pytest.MonkeyPatch, event: _FallbackEvent, metric_name: str
):
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", [])
    monkeypatch.setattr(litellm, "custom_prometheus_tags", [])
    _clear_prometheus_registry()
    try:
        logger: Final = PrometheusLogger()
        await _log_fallback(logger, event, {"model": "model-b", "metadata": {"spend_logs_metadata": {"x": "y"}}})

        labels: Final = _only_sample_labels(metric_name)
    finally:
        _clear_prometheus_registry()

    assert not any(name.startswith("metadata_") for name in labels)
    assert labels["fallback_model"] == "model-b"
