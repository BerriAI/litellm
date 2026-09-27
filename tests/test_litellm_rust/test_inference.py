import asyncio
from collections.abc import Awaitable, Coroutine, Mapping
from typing import Final, Literal, TypeAlias

import pytest
from pydantic import JsonValue, TypeAdapter

from litellm import RateLimitError
from litellm.integrations.custom_logger import CustomLogger
from litellm.rust_bridge import _native
from litellm.rust_bridge.chat_completions.entrypoints import LiteLLMChatCompletionsRequest
from litellm.rust_bridge.responses.entrypoints import LiteLLMResponsesRequest
from litellm.types.llms.openai import ResponsesAPIResponse
from litellm.types.utils import ModelResponse
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import MESSAGES, MESSAGES_MODEL, MESSAGES_RESPONSE, request_body

pytestmark = [pytest.mark.requires_rust_extension, pytest.mark.parametrize("route", ("chat", "responses"))]
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
        request: Final = LiteLLMChatCompletionsRequest(
            MESSAGES_MODEL, list(MESSAGES), None, "test-key", server.base_url, None, None, kwargs
        )
        return (_native.acompletion if asynchronous else _native.completion)(request, (), kwargs)
    response_kwargs: Final = {
        "model": RESPONSES_MODEL,
        "input": "hello",
        "api_key": "test-key",
        "api_base": server.base_url,
        "max_output_tokens": 32,
        **options,
    }
    response_request: Final = LiteLLMResponsesRequest(
        RESPONSES_MODEL, "hello", None, "test-key", server.base_url, "openai", None, response_kwargs
    )
    return (_native.aresponses if asynchronous else _native.responses)(response_request, (), response_kwargs)


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
) -> None:
    recorder: Final = RecordingLogger()
    recording_server.expected_requests = 0
    pending: Final = native_call(route, True, recording_server, {"callbacks": [recorder]})
    assert asyncio.iscoroutine(pending)
    pending.close()
    assert not recording_server.requests
    assert not recorder.events


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "options",
    (
        {"stream": True},
        {"extra_body": {"provider_option": True}},
        {"mock_response": "mock"},
        {"num_retries": 1},
        {"use_chat_completions_api": True},
        {"model_list": []},
    ),
)
async def test_native_inference_declines_unsupported_requests_before_callbacks(
    route: Route,
    recording_server: RecordingServer,
    options: Mapping[str, object],
) -> None:
    recorder: Final = RecordingLogger()
    recording_server.expected_requests = 0
    with pytest.raises(_native.RustBridgeDeclined):
        native_call(route, True, recording_server, {**options, "callbacks": [recorder]})
    assert not recording_server.requests
    assert not recorder.events


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
        await asyncio.to_thread(_native.completion, request, args, kwargs)
        assert _OBJECT.validate_python(recording_server.requests[0].body)["temperature"] == 0.25
    else:
        response_args: Final = ("hello", RESPONSES_MODEL, None, "Be brief", 16)
        response_request: Final = responses_dispatch.request(response_args, kwargs)
        assert response_request is not None
        await asyncio.to_thread(_native.responses, response_request, response_args, kwargs)
        body: Final = _OBJECT.validate_python(recording_server.requests[0].body)
        assert body["instructions"] == "Be brief"
        assert body["max_output_tokens"] == 16
