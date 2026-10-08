import asyncio
from collections.abc import Awaitable, Coroutine, Mapping
from typing import Final, Literal, TypeAlias

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm import RateLimitError
from litellm.chat_completions import dispatch as chat_dispatch
from litellm.integrations.custom_logger import CustomLogger
from litellm.models.credentials import CredentialItem
from litellm.responses.utils import ResponsesAPIRequestUtils
from litellm.rust_bridge import _native
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import CallTypes, ModelResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_MODEL, MESSAGES_RESPONSE, request_body

pytestmark = pytest.mark.requires_rust_extension
Route: TypeAlias = Literal["chat", "responses"]
_OBJECT: Final = TypeAdapter(dict[str, object])
NativeResult: TypeAlias = (
    ModelResponse
    | ResponsesAPIResponse
    | Coroutine[object, object, ModelResponse]
    | Coroutine[object, object, ResponsesAPIResponse]
)
RESPONSES_MODEL: Final = "openai/gpt-6-sol"
RESPONSES_RESPONSE: Final[dict[str, JsonValue]] = {
    "id": "resp_native",
    "object": "response",
    "created_at": 1,
    "model": RESPONSES_MODEL.removeprefix("openai/"),
    "status": "completed",
    "output": [
        {
            "type": "message",
            "id": "msg_native",
            "role": "assistant",
            "status": "completed",
            "content": [{"type": "output_text", "text": "native response", "annotations": []}],
        }
    ],
    "usage": {"input_tokens": 5, "output_tokens": 4, "total_tokens": 9},
}


@pytest.fixture(params=("chat", "responses"))
def route(request: pytest.FixtureRequest) -> Route:
    return TypeAdapter(Route).validate_python(request.param)


def native_call(
    route: Route, asynchronous: bool, server: RecordingServer, options: Mapping[str, object]
) -> NativeResult:
    server.default_response = ResponseSpec(body=MESSAGES_RESPONSE if route == "chat" else RESPONSES_RESPONSE)
    if route == "chat":
        kwargs: Final = {
            "model": MESSAGES_MODEL,
            "messages": list(MESSAGES),
            "api_key": "test-key",
            "api_base": server.base_url,
            "max_tokens": 32,
            **options,
        }
        request: Final = NativeCall(args=(), kwargs=kwargs, bound=kwargs)
        return (_native.acompletion if asynchronous else _native.completion)(request)
    response_kwargs: Final = {
        "model": RESPONSES_MODEL,
        "input": "hello",
        "api_key": "test-key",
        "api_base": server.base_url,
        "max_output_tokens": 32,
        **options,
    }
    response_request: Final = NativeCall(
        args=(),
        kwargs=response_kwargs,
        bound={
            "model": RESPONSES_MODEL,
            "input": "hello",
            "stream": None,
            "api_key": "test-key",
            "api_base": server.base_url,
            "custom_llm_provider": "openai",
            "extra_headers": None,
            **response_kwargs,
        },
    )
    return (_native.aresponses if asynchronous else _native.responses)(response_request)


async def execute(route: Route, asynchronous: bool, server: RecordingServer, options: Mapping[str, object]) -> object:
    if not asynchronous:
        return await asyncio.to_thread(native_call, route, False, server, options)
    result: Final = native_call(route, True, server, options)
    assert isinstance(result, Awaitable)
    return await result


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_native_inference_returns_public_models_and_logs_once(
    route: Route,
    asynchronous: bool,
    recording_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()
    result: Final = await execute(route, asynchronous, recording_server, {"callbacks": [recorder], "temperature": 0.25})
    assert len(recording_server.requests) == 1
    sent: Final = recording_server.requests[0]
    body: Final = _OBJECT.validate_python(sent.body)
    assert body["temperature"] == 0.25
    assert request_body(_OBJECT.validate_python(recorder.wait_for("log_pre_api_call")[0].kwargs)) == body
    if route == "chat":
        assert isinstance(result, ModelResponse)
        assert result.choices[0].message.content == "Hello from native Messages"
        assert sent.path == "/v1/messages"
    else:
        assert isinstance(result, ResponsesAPIResponse)
        assert result.output_text == "native response"
        assert sent.path == "/responses"
    success: Final = await recorder.wait_for_async("async_log_success_event" if asynchronous else "log_success_event")
    assert len(success) == 1
    if isinstance(result, ModelResponse):
        assert success[0].response is result
    else:
        logged: Final = success[0].response
        assert isinstance(logged, ResponsesAPIResponse)
        assert isinstance(result, ResponsesAPIResponse)
        assert logged.id == result.id
        assert logged.output_text == result.output_text


@pytest.mark.asyncio
async def test_native_inference_pre_call_edits_reach_the_provider(
    route: Route, recording_server: RecordingServer
) -> None:
    class Edit(CustomLogger):
        def log_pre_api_call(self, model: object, messages: object, kwargs: dict[str, object]) -> None:
            request_body(kwargs)["temperature"] = 0.75

    await execute(route, True, recording_server, {"callbacks": [Edit()]})
    assert _OBJECT.validate_python(recording_server.requests[0].body)["temperature"] == 0.75


@pytest.mark.asyncio
@pytest.mark.parametrize("from_credentials", (False, True))
async def test_native_resource_setup_uses_deployment_hook_arguments(
    route: Route,
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
    from_credentials: bool,
) -> None:
    invalid_settings: Final = {"ssl_verify": object()}
    credential: Final = CredentialItem(
        credential_name="resource-settings", credential_info={}, credential_values=invalid_settings
    )
    monkeypatch.setattr(litellm, "credential_list", [credential])

    class Prepare(CustomLogger):
        async def async_pre_call_deployment_hook(
            self, kwargs: dict[str, object], call_type: CallTypes | None
        ) -> dict[str, object]:
            return {
                **kwargs,
                **({"litellm_credential_name": credential.credential_name} if from_credentials else invalid_settings),
            }

    litellm.callbacks.append(Prepare())
    recorder: Final = RecordingLogger()
    recording_server.expected_requests = 0
    with pytest.raises(ValueError, match=r"request\.ssl_verify") as caught:
        await execute(route, True, recording_server, {"callbacks": [recorder]})
    failure: Final = await recorder.wait_for_async("async_log_failure_event")
    assert len(failure) == 1
    assert _OBJECT.validate_python(failure[0].kwargs)["exception"] is caught.value
    assert not recording_server.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_sdk_policy_rejection_precedes_resource_setup_and_is_logged_once(
    route: Route,
    asynchronous: bool,
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "max_budget", 1.0)
    monkeypatch.setattr(litellm, "_current_cost", 2.0)
    monkeypatch.setattr(litellm, "ssl_verify", object())
    recorder: Final = RecordingLogger()
    recording_server.expected_requests = 0
    with pytest.raises(litellm.BudgetExceededError) as caught:
        await execute(route, asynchronous, recording_server, {"callbacks": [recorder]})
    failure: Final = await recorder.wait_for_async("async_log_failure_event" if asynchronous else "log_failure_event")
    assert len(failure) == 1
    assert _OBJECT.validate_python(failure[0].kwargs)["exception"] is caught.value
    assert not recording_server.requests
    assert not any("success" in name for name in recorder.names)


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_native_inference_provider_failure_is_terminal_and_shared_with_callbacks(
    route: Route,
    asynchronous: bool,
    recording_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()
    recording_server.enqueue(
        ResponseSpec(body={"error": {"message": "slow down", "type": "rate_limit_error"}}, status=429)
    )
    with pytest.raises(RateLimitError) as caught:
        await execute(route, asynchronous, recording_server, {"callbacks": [recorder]})
    assert getattr(caught.value, "status_code", None) == 429
    assert len(recording_server.requests) == 1
    failure: Final = await recorder.wait_for_async("async_log_failure_event" if asynchronous else "log_failure_event")
    assert len(failure) == 1
    assert _OBJECT.validate_python(failure[0].kwargs)["exception"] is caught.value
    assert not any("success" in name for name in recorder.names)


@pytest.mark.asyncio
async def test_unstarted_native_inference_has_no_provider_or_callback_effects(
    route: Route,
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    recorder: Final = RecordingLogger()
    recording_server.expected_requests = 0
    monkeypatch.setattr(litellm, "ssl_verify", object())
    monkeypatch.setattr(litellm, "max_budget", 1.0)
    monkeypatch.setattr(litellm, "_current_cost", 2.0)
    pending: Final = native_call(route, True, recording_server, {"callbacks": [recorder]})
    assert asyncio.iscoroutine(pending)
    pending.close()
    assert not recording_server.requests
    assert not recorder.events


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize(
    "extension",
    (
        None,
        False,
        0,
        {"nested": [True, None, {"value": 7}]},
        {
            "format": {
                "type": "json_schema",
                "schema": {
                    "type": "object",
                    "properties": {"answer": {"type": "string"}},
                    "required": ["answer"],
                    "additionalProperties": False,
                },
            },
            "previous_message_id": None,
        },
    ),
)
async def test_native_projection_preserves_provider_extensions(
    route: Route, asynchronous: bool, recording_server: RecordingServer, extension: object
) -> None:
    recorder: Final = RecordingLogger()
    await execute(
        route,
        asynchronous,
        recording_server,
        {
            "provider_extension": extension,
            "extra_body": {"provider_override": extension, "temperature": 0.75},
            "temperature": 0.25,
            "drop_params": True,
            "callbacks": [recorder],
            "litellm_metadata": {"opaque": object()},
        },
    )
    assert len(recording_server.requests) == 1
    body: Final = _OBJECT.validate_python(recording_server.requests[0].body)
    assert body["provider_extension"] == extension
    assert body["provider_override"] == extension
    assert body["temperature"] == 0.75
    assert "extra_body" not in body
    assert "drop_params" not in body
    assert "callbacks" not in body
    assert "litellm_metadata" not in body
    assert "api_key" not in body


@pytest.mark.asyncio
async def test_native_responses_streaming_failure_is_terminal(
    recording_server: RecordingServer,
) -> None:
    recorder: Final = RecordingLogger()
    recording_server.expected_requests = 0
    with pytest.raises(Exception, match="streaming"):
        await execute("responses", True, recording_server, {"stream": True, "callbacks": [recorder]})
    assert not recording_server.requests


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_native_chat_validation_failure_is_terminal(
    asynchronous: bool, recording_server: RecordingServer
) -> None:
    recording_server.expected_requests = 0
    with pytest.raises(Exception, match="chat completions requires at least one message"):
        await execute("chat", asynchronous, recording_server, {"messages": []})
    assert not recording_server.requests


@pytest.mark.asyncio
async def test_native_projection_reads_positional_parameters(route: Route, recording_server: RecordingServer) -> None:
    from litellm.chat_completions.dispatch import (
        _DISPATCH as chat_dispatch,  # pyright: ignore[reportPrivateUsage]  # exercise the request passed to the native boundary
    )
    from litellm.responses.dispatch import (
        _DISPATCH as responses_dispatch,  # pyright: ignore[reportPrivateUsage]  # exercise the request passed to the native boundary
    )

    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE if route == "chat" else RESPONSES_RESPONSE)
    kwargs: Final = {"api_key": "test-key", "api_base": recording_server.base_url}
    if route == "chat":
        args: Final = (MESSAGES_MODEL, list(MESSAGES), 12.0, 0.25)
        request: Final = chat_dispatch.request(args, kwargs)
        assert request is not None
        await asyncio.to_thread(_native.completion, request)
        assert _OBJECT.validate_python(recording_server.requests[0].body)["temperature"] == 0.25
    else:
        response_args: Final = ("hello", RESPONSES_MODEL, None, "Be brief", 16)
        response_request: Final = responses_dispatch.request(response_args, kwargs)
        assert response_request is not None
        await asyncio.to_thread(_native.responses, response_request)
        body: Final = _OBJECT.validate_python(recording_server.requests[0].body)
        assert body["instructions"] == "Be brief"
        assert body["max_output_tokens"] == 16


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("source", ("explicit", "base_url", "global", "provider", "environment", "empty"))
async def test_native_connection_settings_reach_the_provider(
    route: Route,
    asynchronous: bool,
    source: str,
    recording_server: RecordingServer,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key: Final = "selected-key"
    explicit: Final = source in ("explicit", "base_url")
    monkeypatch.setattr(litellm, "api_key", key if source in ("global", "empty") else ("unused" if explicit else None))
    monkeypatch.setattr(litellm, "openai_key", key if source == "provider" else ("unused" if explicit else None))
    monkeypatch.setattr(litellm, "anthropic_key", key if source == "provider" else ("unused" if explicit else None))
    monkeypatch.setattr(
        litellm,
        "api_base",
        None if source == "environment" else ("http://127.0.0.1:1" if explicit else recording_server.base_url),
    )
    monkeypatch.setenv(
        "OPENAI_API_KEY" if route == "responses" else "ANTHROPIC_API_KEY", key if source == "environment" else "unused"
    )
    for name in ("OPENAI_BASE_URL", "OPENAI_API_BASE", "ANTHROPIC_BASE_URL", "ANTHROPIC_API_BASE"):
        monkeypatch.setenv(name, recording_server.base_url if source == "environment" else "http://127.0.0.1:1")
    result: Final = await execute(
        route,
        asynchronous,
        recording_server,
        {
            "api_key": key if explicit else ("" if source == "empty" else None),
            "api_base": recording_server.base_url if source == "explicit" else ("" if source == "empty" else None),
            **({"base_url": recording_server.base_url} if source == "base_url" else {}),
        },
    )
    assert isinstance(result, ModelResponse | ResponsesAPIResponse)
    assert len(recording_server.requests) == 1
    headers: Final = recording_server.requests[0].headers
    assert headers["x-api-key" if route == "chat" else "authorization"] == (key if route == "chat" else f"Bearer {key}")


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
@pytest.mark.parametrize("encoded", (False, True))
async def test_native_responses_decode_continuation_ids(
    asynchronous: bool, encoded: bool, recording_server: RecordingServer
) -> None:
    original: Final = "resp_upstream"
    previous: Final = (
        ResponsesAPIRequestUtils._build_responses_api_response_id("openai", "deployment", original)
        if encoded
        else original
    )
    await execute("responses", asynchronous, recording_server, {"previous_response_id": previous})
    assert _OBJECT.validate_python(recording_server.requests[0].body)["previous_response_id"] == original


@pytest.mark.asyncio
@pytest.mark.parametrize("asynchronous", (False, True))
async def test_native_chat_uses_bound_positional_parameters(
    asynchronous: bool, recording_server: RecordingServer
) -> None:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    arguments: Final = (MESSAGES_MODEL, list(MESSAGES), 12.0, 0.35)
    supplied: Final = {"base_url": recording_server.base_url, "api_key": "test-key", "max_tokens": 32}
    call: Final = chat_dispatch._DISPATCH.request(arguments, supplied)  # pyright: ignore[reportPrivateUsage]  # exercise the native request produced by public binding
    assert call is not None
    result: Final = (
        await _native.acompletion(call) if asynchronous else await asyncio.to_thread(_native.completion, call)
    )
    assert isinstance(result, ModelResponse)
    assert result.choices[0].message.content == "Hello from native Messages"
    assert len(recording_server.requests) == 1
    body: Final = _OBJECT.validate_python(recording_server.requests[0].body)
    assert body["temperature"] == arguments[3]
    assert body["max_tokens"] == supplied["max_tokens"]
