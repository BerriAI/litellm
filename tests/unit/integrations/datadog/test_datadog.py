import gzip
import json
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Coroutine, Final
from unittest.mock import AsyncMock, patch

import pytest
from httpx import Request, Response

import litellm
import litellm.integrations.datadog.datadog as datadog_module
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.datadog.datadog_handler import (
    get_datadog_env,
    get_datadog_hostname,
    get_datadog_pod_name,
    get_datadog_service,
    get_datadog_source,
    get_datadog_tags,
)
from litellm.types.integrations.datadog import DatadogInitParams, DatadogPayload, DataDogStatus
from litellm.types.utils import (
    StandardLoggingHiddenParams,
    StandardLoggingMetadata,
    StandardLoggingModelInformation,
    StandardLoggingPayload,
)

STANDARD_START_TIME: Final[datetime] = datetime(2025, 1, 1)
STANDARD_END_TIME: Final[datetime] = datetime(2025, 1, 1, 0, 0, 1)


def _discard_periodic_flush(coroutine: Coroutine[object, object, None]) -> None:
    coroutine.close()


@dataclass(frozen=True, slots=True)
class _DummySpan:
    trace_id: int | None
    span_id: int | None


@dataclass(frozen=True, slots=True)
class _DummyTracer:
    span: _DummySpan | None
    root_span: _DummySpan | None = None

    def current_span(self) -> _DummySpan | None:
        return self.span

    def current_root_span(self) -> _DummySpan | None:
        return self.root_span


def _standard_logging_payload() -> StandardLoggingPayload:
    return StandardLoggingPayload(
        id="test_id",
        trace_id="trace-id",
        session_id="session-id",
        litellm_call_id="call-id",
        call_type="completion",
        stream=False,
        response_cost=0.1,
        cost_breakdown=None,
        autorouter_savings=None,
        autorouter_savings_estimate=None,
        autorouter_baseline_observation=None,
        response_cost_failure_debug_info=None,
        status="success",
        status_fields={},
        custom_llm_provider="openai",
        total_tokens=30,
        prompt_tokens=20,
        completion_tokens=10,
        startTime=1234567890.0,
        endTime=1234567891.0,
        completionStartTime=1234567890.5,
        response_time=1.0,
        model_map_information=StandardLoggingModelInformation(
            model_map_key="gpt-4.1-mini",
            model_map_value=None,
        ),
        model="gpt-4.1-mini",
        model_id="model-123",
        model_group="openai-gpt",
        api_base="https://api.openai.com",
        user_agent=None,
        metadata=StandardLoggingMetadata(
            user_api_key_hash="test_hash",
            user_api_key_org_id=None,
            user_api_key_alias="test_alias",
            user_api_key_team_id="test_team",
            user_api_key_user_id="test_user",
            user_api_key_team_alias="test_team_alias",
            spend_logs_metadata=None,
            requester_ip_address="127.0.0.1",
            requester_metadata=None,
        ),
        cache_hit=False,
        cache_key=None,
        saved_cache_cost=0.0,
        request_tags=[],
        end_user=None,
        requester_ip_address="127.0.0.1",
        messages=[{"role": "user", "content": "Hello, world!"}],
        response={"choices": [{"message": {"content": "Hi there!"}}]},
        error_str=None,
        error_information=None,
        model_parameters={"stream": True},
        hidden_params=StandardLoggingHiddenParams(
            model_id="model-123",
            cache_key=None,
            api_base="https://api.openai.com",
            response_cost="0.1",
            additional_headers=None,
        ),
        guardrail_information=None,
        standard_built_in_tools_params=None,
    )


@pytest.fixture
def datadog_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DD_API_KEY", "test_api_key")
    monkeypatch.setenv("DD_SITE", "test.datadoghq.com")
    monkeypatch.delenv("LITELLM_DD_AGENT_HOST", raising=False)
    monkeypatch.delenv("LITELLM_DD_AGENT_PORT", raising=False)
    monkeypatch.delenv("DD_BASE_URL", raising=False)
    monkeypatch.setattr(litellm, "datadog_params", None)
    monkeypatch.setattr(litellm, "datadog_use_v1", False)


@pytest.fixture
def datadog_logger(datadog_env: None) -> DataDogLogger:
    with patch("asyncio.create_task", side_effect=_discard_periodic_flush):
        logger: Final = DataDogLogger()
    return logger


def test_add_trace_context_uses_current_span(
    datadog_logger: DataDogLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        datadog_module,
        "tracer",
        _DummyTracer(span=_DummySpan(trace_id=123, span_id=456)),
    )
    payload: Final = DatadogPayload(
        ddsource="litellm",
        ddtags="env:test",
        hostname="host",
        message="{}",
        service="svc",
        status="info",
    )

    datadog_logger._add_trace_context_to_payload(payload)

    assert payload["dd.trace_id"] == "123"
    assert payload["dd.span_id"] == "456"


def test_add_trace_context_falls_back_to_root_span(
    datadog_logger: DataDogLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        datadog_module,
        "tracer",
        _DummyTracer(span=None, root_span=_DummySpan(trace_id=789, span_id=None)),
    )
    payload: Final = DatadogPayload(
        ddsource="litellm",
        ddtags="env:test",
        hostname="host",
        message="{}",
        service="svc",
        status="info",
    )

    datadog_logger._add_trace_context_to_payload(payload)

    assert payload["dd.trace_id"] == "789"
    assert "dd.span_id" not in payload


def test_add_trace_context_handles_missing_tracer(
    datadog_logger: DataDogLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(datadog_module, "tracer", object())
    payload: Final = DatadogPayload(
        ddsource="litellm",
        ddtags="env:test",
        hostname="host",
        message="{}",
        service="svc",
        status="info",
    )

    datadog_logger._add_trace_context_to_payload(payload)

    assert "dd.trace_id" not in payload
    assert "dd.span_id" not in payload


def test_add_trace_context_ignores_span_without_trace_id(
    datadog_logger: DataDogLogger,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        datadog_module,
        "tracer",
        _DummyTracer(span=_DummySpan(trace_id=None, span_id=555)),
    )
    payload: Final = DatadogPayload(
        ddsource="litellm",
        ddtags="env:test",
        hostname="host",
        message="{}",
        service="svc",
        status="info",
    )

    datadog_logger._add_trace_context_to_payload(payload)

    assert "dd.trace_id" not in payload
    assert "dd.span_id" not in payload


async def test_datadog_logging_http_request(datadog_logger: DataDogLogger) -> None:
    datadog_logger.batch_size = 5
    standard_logging_payload: Final = {
        **_standard_logging_payload(),
        "call_type": "acompletion",
        "model_parameters": {"temperature": 0.2, "max_tokens": 10},
    }
    response: Final = Response(
        202,
        request=Request("POST", "https://example.com"),
        text="Accepted",
    )
    mock_post: Final = AsyncMock(return_value=response)

    with patch.object(datadog_logger.async_client, "post", new=mock_post):
        for _ in range(5):
            await datadog_logger.async_log_success_event(
                kwargs={"standard_logging_object": standard_logging_payload},
                response_obj=None,
                start_time=STANDARD_START_TIME,
                end_time=STANDARD_END_TIME,
            )

    mock_post.assert_awaited_once()
    request_args: Final = mock_post.await_args
    assert request_args is not None
    assert request_args.kwargs["url"].endswith("/api/v2/logs")
    body: Final = json.loads(gzip.decompress(request_args.kwargs["data"]).decode("utf-8"))
    assert len(body) == 5

    expected_fields: Final = set(DatadogPayload.__annotations__)
    required_fields: Final = {"ddsource", "ddtags", "hostname", "message", "service", "status"}
    for log in body:
        assert required_fields <= set(log)
        assert set(log) <= expected_fields
        assert all(isinstance(value, str) for value in log.values())

    message: Final = json.loads(body[0]["message"])
    assert StandardLoggingPayload.__required_keys__ <= set(message)
    assert message["call_type"] == "acompletion"
    assert message["model"] == "gpt-4.1-mini"
    assert isinstance(message["model_parameters"], dict)
    assert "temperature" in message["model_parameters"]
    assert "max_tokens" in message["model_parameters"]
    assert isinstance(message["response"], dict)
    assert isinstance(message["metadata"], dict)


def test_datadog_payload_environment_variables(datadog_env: None) -> None:
    test_env: Final = {
        "DD_ENV": "test-env",
        "DD_SERVICE": "test-service",
        "DD_VERSION": "1.0.0",
        "DD_SOURCE": "test-source",
        "DD_API_KEY": "fake-key",
        "DD_SITE": "datadoghq.com",
    }
    with (
        patch.dict(os.environ, test_env, clear=True),
        patch("asyncio.create_task", side_effect=_discard_periodic_flush),
    ):
        logger: Final = DataDogLogger()
        standard_payload: Final = _standard_logging_payload()
        datadog_payload: Final = logger.create_datadog_logging_payload(
            kwargs={"standard_logging_object": standard_payload},
            response_obj=None,
            start_time=STANDARD_START_TIME,
            end_time=STANDARD_END_TIME,
        )

    assert datadog_payload["ddsource"] == "test-source"
    assert datadog_payload["service"] == "test-service"
    assert "env:test-env,service:test-service,version:1.0.0,HOSTNAME:" in datadog_payload["ddtags"]


def test_datadog_payload_content_truncation(datadog_logger: DataDogLogger) -> None:
    long_content: Final = "x" * 80_000
    standard_payload: Final = {
        **_standard_logging_payload(),
        "error_str": long_content,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "image_url", "image_url": {"url": long_content, "detail": "low"}}],
            }
        ],
        "response": {"choices": [{"message": {"content": long_content}}]},
    }
    datadog_payload: Final = datadog_logger.create_datadog_logging_payload(
        kwargs={"standard_logging_object": standard_payload},
        response_obj=None,
        start_time=STANDARD_START_TIME,
        end_time=STANDARD_END_TIME,
    )
    message: Final = json.loads(datadog_payload["message"])

    assert len(message["error_str"]) < 10_100
    assert len(str(message["messages"])) < 10_100
    assert len(str(message["response"])) < 10_100


def test_datadog_payload_truncation_leaves_shared_payload_intact(
    datadog_logger: DataDogLogger,
) -> None:
    original_messages: Final = [{"role": "user", "content": "x" * 80_000}]
    standard_payload: Final = {
        **_standard_logging_payload(),
        "messages": original_messages,
    }
    kwargs: Final = {"standard_logging_object": standard_payload}

    datadog_payload: Final = datadog_logger.create_datadog_logging_payload(
        kwargs=kwargs,
        response_obj=None,
        start_time=STANDARD_START_TIME,
        end_time=STANDARD_END_TIME,
    )
    messages: Final = json.loads(datadog_payload["message"])["messages"]

    assert kwargs["standard_logging_object"]["messages"] is original_messages
    assert len(str(messages)) < 10_100


def test_datadog_static_methods() -> None:
    with patch.dict(os.environ, {}, clear=True):
        assert get_datadog_source() == "litellm"
        assert get_datadog_service() == "litellm-server"
        assert get_datadog_hostname() == ""
        assert get_datadog_env() == "unknown"
        assert get_datadog_pod_name() == "unknown"
        assert "env:unknown,service:litellm-server,version:unknown,HOSTNAME:" in ",".join(get_datadog_tags())

    test_env: Final = {
        "DD_SOURCE": "custom-source",
        "DD_SERVICE": "custom-service",
        "HOSTNAME": "test-host",
        "DD_ENV": "production",
        "DD_VERSION": "1.0.0",
        "POD_NAME": "pod-123",
    }
    with patch.dict(os.environ, test_env, clear=True):
        assert get_datadog_source() == "custom-source"
        assert get_datadog_service() == "custom-service"
        assert get_datadog_hostname() == "test-host"
        assert get_datadog_env() == "production"
        assert get_datadog_pod_name() == "pod-123"
        assert ",".join(get_datadog_tags()) == (
            "env:production,service:custom-service,version:1.0.0,HOSTNAME:test-host,POD_NAME:pod-123"
        )


def test_datadog_non_serializable_messages(datadog_logger: DataDogLogger) -> None:
    non_serializable_object: Final = datetime(2025, 1, 1)
    standard_payload: Final = {
        **_standard_logging_payload(),
        "messages": [{"role": "user", "content": non_serializable_object}],
        "response": {"choices": [{"message": {"content": non_serializable_object}}]},
    }
    datadog_payload: Final = datadog_logger.create_datadog_logging_payload(
        kwargs={"standard_logging_object": standard_payload},
        response_obj=None,
        start_time=STANDARD_START_TIME,
        end_time=STANDARD_END_TIME,
    )
    message: Final = json.loads(datadog_payload["message"])

    assert datadog_payload["status"] == DataDogStatus.INFO
    assert isinstance(message["messages"][0]["content"], str)
    assert isinstance(message["response"]["choices"][0]["message"]["content"], str)


def test_get_datadog_tags() -> None:
    test_env: Final = {
        "DD_ENV": "production",
        "DD_SERVICE": "custom-service",
        "DD_VERSION": "1.0.0",
        "HOSTNAME": "test-host",
        "POD_NAME": "pod-123",
    }
    with patch.dict(os.environ, test_env, clear=True):
        base_tags: Final = get_datadog_tags()
        assert "env:production" in base_tags
        assert "service:custom-service" in base_tags
        assert "version:1.0.0" in base_tags
        assert "HOSTNAME:test-host" in base_tags
        assert "POD_NAME:pod-123" in base_tags

        standard_logging_payload: Final = {
            **_standard_logging_payload(),
            "request_tags": ["tag1", "tag2"],
        }
        tags_with_request: Final = get_datadog_tags(standard_logging_payload)
        assert "request_tag:tag1" in tags_with_request
        assert "request_tag:tag2" in tags_with_request

        empty_request_payload: Final = {
            **standard_logging_payload,
            "request_tags": [],
        }
        assert not any(tag.startswith("request_tag:") for tag in get_datadog_tags(empty_request_payload))

        no_request_payload: Final = {
            **standard_logging_payload,
            "request_tags": None,
        }
        assert not any(tag.startswith("request_tag:") for tag in get_datadog_tags(no_request_payload))


def test_datadog_message_redaction(
    datadog_env: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "datadog_params", DatadogInitParams(turn_off_message_logging=True))
    monkeypatch.setattr(litellm, "callbacks", [])
    with patch("asyncio.create_task", side_effect=_discard_periodic_flush):
        logger: Final = DataDogLogger()
    model_call_details: Final = {
        "standard_logging_object": {
            "messages": [{"role": "user", "content": "sensitive request"}],
            "response": {"choices": [{"message": {"content": "sensitive response"}}]},
        }
    }

    redacted_details: Final = logger.redact_standard_logging_payload_from_model_call_details(model_call_details)
    standard_payload: Final = redacted_details["standard_logging_object"]

    assert standard_payload["messages"][0]["content"] == "redacted-by-litellm"
    assert standard_payload["response"]["choices"][0]["message"]["content"] == "redacted-by-litellm"


def test_datadog_agent_configuration(datadog_env: None) -> None:
    test_env: Final = {
        "LITELLM_DD_AGENT_HOST": "localhost",
        "LITELLM_DD_AGENT_PORT": "10518",
    }
    with (
        patch.dict(os.environ, test_env, clear=True),
        patch("asyncio.create_task", side_effect=_discard_periodic_flush),
    ):
        logger: Final = DataDogLogger()

    assert logger.intake_url == "http://localhost:10518/api/v2/logs"
    assert logger.DD_API_KEY is None


def test_datadog_ignores_ddtrace_agent_host(datadog_env: None) -> None:
    test_env: Final = {
        "DD_API_KEY": "fake-api-key",
        "DD_SITE": "us5.datadoghq.com",
        "DD_AGENT_HOST": "10.176.100.40",
        "DD_AGENT_PORT": "8126",
    }
    with (
        patch.dict(os.environ, test_env, clear=True),
        patch("asyncio.create_task", side_effect=_discard_periodic_flush),
    ):
        logger: Final = DataDogLogger()

    assert logger.intake_url == "https://http-intake.logs.us5.datadoghq.com/api/v2/logs"
    assert logger.DD_API_KEY == "fake-api-key"
