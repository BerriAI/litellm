import asyncio
import base64
import json
import threading
from collections.abc import AsyncIterator, Iterator
from typing import Final
from unittest.mock import AsyncMock, Mock
from uuid import uuid4

import httpx
import pytest
import pytest_asyncio
import respx

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup
from litellm.litellm_core_utils.logging_worker import GLOBAL_LOGGING_WORKER
from litellm.types.utils import StandardLoggingPayload


def _provider_response(request: httpx.Request) -> httpx.Response:
    path: Final = request.url.path
    if request.headers.get("authorization") == "Bearer bad-key":
        return httpx.Response(401, json={"error": {"message": "bad key"}})
    body: Final = json.loads(request.content or b"{}")
    if body.get("model") == "gpt-4o-audio-preview" and body.get("stream"):
        audio_chunks: Final = (
            {
                "id": "chatcmpl_audio",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "gpt-4o-audio-preview",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "audio": {
                                "id": "audio-response",
                                "data": "Zm9v",
                                "transcript": "hello ",
                            }
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_audio",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "gpt-4o-audio-preview",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "audio": {
                                "data": "YmFy",
                                "transcript": "world",
                                "expires_at": 1700003600,
                            }
                        },
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl_audio",
                "object": "chat.completion.chunk",
                "created": 1700000000,
                "model": "gpt-4o-audio-preview",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        )
        stream_body: Final = b"".join(
            f"data: {json.dumps(chunk)}\n\n".encode() for chunk in audio_chunks
        )
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=stream_body + b"data: [DONE]\n\n",
        )
    if body.get("model") == "gpt-4o-audio-preview":
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl_audio",
                "object": "chat.completion",
                "created": 1700000000,
                "model": "gpt-4o-audio-preview",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "audio": {
                                "id": "audio-response",
                                "data": base64.b64encode(b"foobar").decode("ascii"),
                                "transcript": "hello world",
                                "expires_at": 1700003600,
                            },
                        },
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 2, "completion_tokens": 3, "total_tokens": 5},
            },
        )
    if "filtered request" in json.dumps(body):
        return httpx.Response(
            400,
            json={
                "error": {
                    "code": "ResponsibleAIPolicyViolation",
                    "message": "The response was filtered due to the prompt triggering Azure OpenAI's content management policy",
                    "type": "invalid_request_error",
                    "innererror": {
                        "code": "ResponsibleAIPolicyViolation",
                        "content_filter_result": {
                            "violence": {"filtered": True, "severity": "high"},
                        },
                    },
                }
            },
        )
    if "generativelanguage.googleapis.com" in str(request.url):
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"role": "model", "parts": [{"text": "Hello"}]}, "finishReason": "STOP"}
                ],
                "usageMetadata": {"promptTokenCount": 2, "candidatesTokenCount": 1, "totalTokenCount": 3},
            },
        )
    if "invoke" in path:
        return httpx.Response(
            200,
            json={"embeddings": [[0.1, 0.2, 0.3]], "inputTextTokenCount": 2},
        )
    if "embeddings" in path:
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "model": body.get("model", "embedding-model"),
                "usage": {"prompt_tokens": 2, "total_tokens": 2},
            },
        )
    if "images/generations" in path:
        return httpx.Response(200, json={"created": 1700000000, "data": [{"url": "https://images.test/result"}]})
    if path.endswith("/messages") and body.get("stream"):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'event: message_start\n'
                b'data: {"type":"message_start","message":{"id":"msg_test","type":"message","role":"assistant","content":[],"model":"claude-test","usage":{"input_tokens":2,"output_tokens":0}}}\n\n'
                b'event: content_block_delta\n'
                b'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}\n\n'
                b'event: message_delta\n'
                b'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":1}}\n\n'
                b'event: message_stop\n'
                b'data: {"type":"message_stop"}\n\n'
            ),
        )
    if path.endswith("/completions") and not path.endswith("/chat/completions") and body.get("stream"):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"id":"cmpl_test","object":"text_completion","created":1700000000,"model":"gpt-3.5-turbo","choices":[{"text":"Hello","index":0,"finish_reason":null}]}\n\n'
                b'data: {"id":"cmpl_test","object":"text_completion","created":1700000000,"model":"gpt-3.5-turbo","choices":[{"text":"","index":0,"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )
    if body.get("stream"):
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=(
                b'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1700000000,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"},"finish_reason":null}]}\n\n'
                b'data: {"id":"chatcmpl_test","object":"chat.completion.chunk","created":1700000000,"model":"test-model","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":2,"completion_tokens":1,"total_tokens":3}}\n\n'
                b"data: [DONE]\n\n"
            ),
        )
    if path.endswith("/completions") and not path.endswith("/chat/completions"):
        return httpx.Response(
            200,
            json={
                "id": "cmpl_test",
                "object": "text_completion",
                "model": "text-model",
                "choices": [{"text": "Hello", "index": 0, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        )
    return httpx.Response(
        200,
        headers={"llm_provider-x-request-id": "respx-request-id"},
        json={
            "id": "chatcmpl_test",
            "object": "chat.completion",
            "created": 1700000000,
            "model": body.get("model", "test-model"),
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "Hello"}, "finish_reason": "stop"}
            ],
            "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
        },
    )


class _CallbackCapture(CustomLogger):
    def __init__(self) -> None:
        self.call_id: Final = str(uuid4())
        self.sync_success: Final = Mock()
        self.sync_logged: Final = threading.Semaphore(0)
        self.async_success: Final = AsyncMock()
        self.sync_failure: Final = Mock()
        self.async_failure: Final = AsyncMock()

    def log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.sync_success(kwargs=kwargs, response_obj=response_obj)
        self.sync_logged.release()

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time) -> None:
        await self.async_success(kwargs=kwargs, response_obj=response_obj)

    def log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        self.sync_failure(kwargs=kwargs, response_obj=response_obj)

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time) -> None:
        await self.async_failure(kwargs=kwargs, response_obj=response_obj)


@pytest.fixture
def provider_router(monkeypatch: pytest.MonkeyPatch) -> Iterator[respx.MockRouter]:
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("AZURE_API_KEY", "test-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    with respx.mock(assert_all_called=False, assert_all_mocked=True) as router:
        router.route(url__regex=r"https?://.*").mock(side_effect=_provider_response)
        yield router


@pytest_asyncio.fixture(loop_scope="function")
async def callback_capture(monkeypatch: pytest.MonkeyPatch) -> AsyncIterator[_CallbackCapture]:
    await _drain_logging_worker()
    capture: Final = _CallbackCapture()
    monkeypatch.setattr(litellm, "callbacks", [capture])
    monkeypatch.setattr(litellm, "success_callback", [capture])
    monkeypatch.setattr(litellm, "_async_success_callback", [capture])
    monkeypatch.setattr(litellm, "failure_callback", [capture])
    monkeypatch.setattr(litellm, "_async_failure_callback", [capture])
    try:
        yield capture
    finally:
        await _drain_logging_worker()


async def _drain_logging_worker() -> None:
    for _ in range(10):
        await asyncio.sleep(0)
        await GLOBAL_LOGGING_WORKER.flush()


async def _assert_success_payload(
    capture: _CallbackCapture,
    model: str,
    messages: list[dict[str, str]] | None,
    request_marker: str | None = None,
    call_id: str | None = None,
) -> None:
    await _drain_logging_worker()
    expected_model: Final = model.rsplit("/", maxsplit=1)[-1]
    calls: Final = tuple(
        call
        for call in (*capture.async_success.call_args_list, *capture.sync_success.call_args_list)
        if call.kwargs["kwargs"].get("model") == expected_model
        and (messages is None or call.kwargs["kwargs"].get("messages") == messages)
        and (
            request_marker is None
            or request_marker
            in json.dumps(
                (
                    call.kwargs["kwargs"].get("input"),
                    call.kwargs["kwargs"].get("prompt"),
                    call.kwargs["kwargs"].get("messages"),
                    call.kwargs["kwargs"].get("metadata"),
                ),
                default=str,
            )
        )
        and (call_id is None or call.kwargs["kwargs"].get("litellm_call_id") == call_id)
    )
    assert calls, (
        f"no callback event matched model={expected_model!r}, messages={messages!r}; "
        f"async models={tuple(item.kwargs['kwargs'].get('model') for item in capture.async_success.call_args_list)!r}, "
        f"sync models={tuple(item.kwargs['kwargs'].get('model') for item in capture.sync_success.call_args_list)!r}"
    )
    call: Final = calls[-1]
    event: Final = call.kwargs["kwargs"]
    assert event["model"] == expected_model
    standard: Final = event["standard_logging_object"]
    expected_standard_model: Final = model if model.startswith("bedrock/") else model.rsplit("/", maxsplit=1)[-1]
    assert standard["model"] == expected_standard_model
    if messages is not None:
        assert event["messages"] == messages
        assert standard["messages"] == messages
        assert standard["response"]["choices"][0]["message"]["content"] == "Hello"
        assert standard["prompt_tokens"] == 2
        assert standard["completion_tokens"] == 1
        assert standard["total_tokens"] == 3


def test_amazing_sync_embedding(provider_router: respx.MockRouter, callback_capture: _CallbackCapture) -> None:
    response: Final = litellm.embedding(
        model="text-embedding-3-small",
        input=["sync embedding marker"],
        metadata={"requester_metadata": {"marker": "embedding"}},
    )
    assert callback_capture.sync_logged.acquire(timeout=5)
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    event: Final = callback_capture.sync_success.call_args.kwargs["kwargs"]
    assert event["model"] == "text-embedding-3-small"
    assert event["input"] == ["sync embedding marker"]
    assert event["standard_logging_object"]["response"]["data"][0]["embedding"] == [0.1, 0.2, 0.3]
    assert event["standard_logging_object"]["metadata"]["requester_metadata"] == {"marker": "embedding"}


@pytest.mark.asyncio
async def test_async_embedding_openai(provider_router: respx.MockRouter, callback_capture: _CallbackCapture) -> None:
    response: Final = await litellm.aembedding(model="text-embedding-3-small", input=["openai embedding marker"])
    await _assert_success_payload(callback_capture, "text-embedding-3-small", None, "openai embedding marker")
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    event: Final = callback_capture.async_success.call_args.kwargs["kwargs"]
    assert event["input"] == ["openai embedding marker"]


@pytest.mark.asyncio
async def test_async_embedding_azure(provider_router: respx.MockRouter, callback_capture: _CallbackCapture) -> None:
    response: Final = await litellm.aembedding(
        model="azure/text-embedding-ada-002",
        input=["azure embedding marker"],
        api_base="https://provider.test/v1",
    )
    await _assert_success_payload(callback_capture, "azure/text-embedding-ada-002", None, "azure embedding marker")
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    event: Final = callback_capture.async_success.call_args.kwargs["kwargs"]
    assert event["input"] == ["azure embedding marker"]
    assert event["standard_logging_object"]["response"]["data"][0]["embedding"] == [0.1, 0.2, 0.3]


@pytest.mark.asyncio
async def test_async_embedding_bedrock(provider_router: respx.MockRouter, callback_capture: _CallbackCapture) -> None:
    response: Final = await litellm.aembedding(
        model="bedrock/cohere.embed-english-v3",
        input=["bedrock embedding marker"],
        aws_access_key_id="test",
        aws_secret_access_key="test",
        aws_region_name="us-east-1",
    )
    await _assert_success_payload(callback_capture, "bedrock/cohere.embed-english-v3", None, "bedrock embedding marker")
    assert response.data[0]["embedding"] == [0.1, 0.2, 0.3]
    event: Final = callback_capture.async_success.call_args.kwargs["kwargs"]
    assert event["input"] == ["bedrock embedding marker"]


@pytest.mark.asyncio
async def test_async_text_completion_openai_stream(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    response: Final = await litellm.atext_completion(
        model="gpt-3.5-turbo",
        prompt="text completion marker",
        stream=True,
    )
    tuple([chunk async for chunk in response])
    await _assert_success_payload(callback_capture, "gpt-3.5-turbo", None, "text completion marker")
    call: Final = callback_capture.async_success.call_args or callback_capture.sync_success.call_args
    event: Final = call.kwargs["kwargs"]
    assert event["model"] == "gpt-3.5-turbo"
    assert event["standard_logging_object"]["response"]["choices"][0]["message"]["content"] == "Hello"
    assert event["standard_logging_object"]["total_tokens"] == 3


@pytest.mark.asyncio
async def test_image_generation_openai(provider_router: respx.MockRouter, callback_capture: _CallbackCapture) -> None:
    response: Final = await litellm.aimage_generation(model="openai/gpt-image-1", prompt="image marker")
    assert response.data[0].url == "https://images.test/result"
    await _assert_success_payload(callback_capture, "openai/gpt-image-1", None, "image marker")
    call: Final = callback_capture.async_success.call_args or callback_capture.sync_success.call_args
    event: Final = call.kwargs["kwargs"]
    assert event["standard_logging_object"]["model"] == "gpt-image-1"
    assert event["standard_logging_object"]["response"]["data"][0]["url"] == "https://images.test/result"


def test_logging_standard_payload_failure_call(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    with pytest.raises(litellm.AuthenticationError):
        litellm.completion(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "failure marker"}],
            api_key="bad-key",
        )
    messages: Final = [{"role": "user", "content": "failure marker"}]
    failure_calls: Final = tuple(
        call
        for call in (*callback_capture.sync_failure.call_args_list, *callback_capture.async_failure.call_args_list)
        if call.kwargs["kwargs"].get("model") == "gpt-4o-mini" and call.kwargs["kwargs"].get("messages") == messages
    )
    assert len(failure_calls) == 1
    failure_event: Final = failure_calls[0].kwargs["kwargs"]
    assert failure_event["model"] == "gpt-4o-mini"
    assert failure_event["messages"] == messages
    assert failure_event["standard_logging_object"]["status"] == "failure"


@pytest.mark.asyncio
async def test_standard_logging_payload_stream_usage(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    response: Final = await litellm.acompletion(
        model="anthropic/claude-test",
        messages=[{"role": "user", "content": "usage marker"}],
        stream=True,
    )
    chunks: Final = tuple([chunk async for chunk in response])
    await _assert_success_payload(callback_capture, "anthropic/claude-test", [{"role": "user", "content": "usage marker"}])
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
    event: Final = callback_capture.async_success.call_args.kwargs["kwargs"]
    assert event["standard_logging_object"]["total_tokens"] == 3


@pytest.mark.asyncio
async def test_async_chat_openai_stream(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    messages: Final = [{"role": "user", "content": "async stream marker"}]
    response: Final = await litellm.acompletion(model="gpt-4o-mini", messages=messages, stream=True)
    chunks: Final = tuple([chunk async for chunk in response])
    await _assert_success_payload(callback_capture, "gpt-4o-mini", messages)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"


@pytest.mark.asyncio
async def test_async_custom_handler_completion(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    messages: Final = [{"role": "user", "content": "completion marker"}]
    await litellm.acompletion(model="gpt-4o-mini", messages=messages)
    await _assert_success_payload(callback_capture, "gpt-4o-mini", messages)
    with pytest.raises(litellm.AuthenticationError):
        await litellm.acompletion(model="gpt-4o-mini", messages=messages, api_key="bad-key")
    await _drain_logging_worker()
    failure_calls: Final = tuple(
        call
        for call in callback_capture.async_failure.call_args_list
        if call.kwargs["kwargs"].get("model") == "gpt-4o-mini" and call.kwargs["kwargs"].get("messages") == messages
    )
    assert len(failure_calls) == 1


@pytest.mark.asyncio
async def test_async_custom_handler_embedding(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    response: Final = await litellm.aembedding(model="text-embedding-3-small", input=["custom handler embedding marker"])
    await _assert_success_payload(callback_capture, "text-embedding-3-small", None, "custom handler embedding marker")
    assert response.usage.prompt_tokens == 2


@pytest.mark.asyncio
async def test_async_custom_handler_embedding_optional_param(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    await litellm.aembedding(model="text-embedding-3-small", input=["optional embedding marker"], user="user-123")
    await _assert_success_payload(callback_capture, "text-embedding-3-small", None, "optional embedding marker")
    call: Final = callback_capture.async_success.call_args
    assert call.kwargs["kwargs"]["optional_params"]["user"] == "user-123"
    assert json.loads(provider_router.calls.last.request.content)["user"] == "user-123"


@pytest.mark.asyncio
async def test_async_embedding_failure_reaches_failure_callback(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    with pytest.raises(litellm.AuthenticationError):
        await litellm.aembedding(model="text-embedding-3-small", input=["failed embedding marker"], api_key="bad-key")
    await _drain_logging_worker()
    failure_events: Final = tuple(
        call.kwargs["kwargs"]
        for call in callback_capture.async_failure.call_args_list
        if call.kwargs["kwargs"].get("input") == ["failed embedding marker"]
    )
    assert len(failure_events) == 1
    assert failure_events[0]["model"] == "text-embedding-3-small"
    assert isinstance(failure_events[0]["exception"], litellm.AuthenticationError)
    assert callback_capture.async_success.call_args_list == []


@pytest.mark.asyncio
async def test_async_custom_handler_stream(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    response: Final = await litellm.acompletion(
        model="azure/gpt-4o-mini",
        messages=[{"role": "user", "content": "azure stream marker"}],
        stream=True,
        api_base="https://provider.test/v1",
    )
    chunks: Final = tuple([chunk async for chunk in response])
    await _assert_success_payload(
        callback_capture,
        "azure/gpt-4o-mini",
        [{"role": "user", "content": "azure stream marker"}],
    )
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"


def test_chat_openai_stream(provider_router: respx.MockRouter, callback_capture: _CallbackCapture) -> None:
    messages: Final = [{"role": "user", "content": "sync stream marker"}]
    response: Final = litellm.completion(model="gpt-4o-mini", messages=messages, stream=True)
    chunks: Final = tuple(response)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
    assert callback_capture.sync_logged.acquire(timeout=5)
    event: Final = callback_capture.sync_success.call_args.kwargs["kwargs"]
    assert event["model"] == "gpt-4o-mini"
    assert event["messages"] == messages
    assert event["standard_logging_object"]["response"]["choices"][0]["message"]["content"] == "Hello"


@pytest.mark.asyncio
async def test_chat_azure_stream(provider_router: respx.MockRouter, callback_capture: _CallbackCapture) -> None:
    messages: Final = [{"role": "user", "content": "sync azure marker"}]
    response: Final = await litellm.acompletion(
        model="azure/gpt-4o-mini",
        messages=messages,
        stream=True,
        api_base="https://provider.test/v1",
    )
    chunks: Final = tuple([chunk async for chunk in response])
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "Hello"
    await _assert_success_payload(callback_capture, "azure/gpt-4o-mini", messages)


@pytest.mark.asyncio
async def test_async_chat_azure_stream(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    response: Final = await litellm.acompletion(
        model="azure/gpt-4o-mini",
        messages=[{"role": "user", "content": "async azure marker"}],
        stream=True,
        api_base="https://provider.test/v1",
    )
    tuple([chunk async for chunk in response])
    await _assert_success_payload(callback_capture, "azure/gpt-4o-mini", [{"role": "user", "content": "async azure marker"}])


@pytest.mark.asyncio
async def test_async_chat_openai_stream_options(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    response: Final = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "stream usage marker"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    chunks: Final = tuple([chunk async for chunk in response])
    await _assert_success_payload(callback_capture, "gpt-4o-mini", [{"role": "user", "content": "stream usage marker"}])
    assert chunks[-1].usage.total_tokens == 3


@pytest.mark.asyncio
async def test_standard_logging_payload(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    messages: Final = [{"role": "user", "content": "standard marker"}]
    response: Final = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=messages,
        metadata={"requester_metadata": {"marker": "standard"}},
    )
    await _assert_success_payload(callback_capture, "gpt-4o-mini", messages)
    event: Final = callback_capture.async_success.call_args.kwargs["kwargs"]
    assert event["model"] == "gpt-4o-mini"
    assert event["messages"] == messages
    assert event["standard_logging_object"]["response"]["choices"][0]["message"]["content"] == "Hello"
    assert event["standard_logging_object"]["metadata"]["requester_metadata"] == {"marker": "standard"}
    assert response.choices[0].message.content == "Hello"
    standard: Final = event["standard_logging_object"]
    assert sorted(StandardLoggingPayload.__required_keys__ - standard.keys()) == []
    assert json.loads(json.dumps(standard))["id"] == standard["id"]
    assert standard["response_cost"] > 0


@pytest.mark.asyncio
async def test_turn_off_message_logging(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture, monkeypatch: pytest.MonkeyPatch
) -> None:
    marker: Final = "secret callback message"
    monkeypatch.setattr(litellm, "turn_off_message_logging", True)
    await litellm.acompletion(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": marker}],
        litellm_call_id=callback_capture.call_id,
    )
    await _assert_success_payload(callback_capture, "gpt-4o-mini", None, call_id=callback_capture.call_id)
    call: Final = callback_capture.async_success.call_args or callback_capture.sync_success.call_args
    event: Final = call.kwargs["kwargs"]
    assert event["standard_logging_object"]["messages"] == [{"role": "user", "content": "redacted-by-litellm"}]
    assert event["standard_logging_object"]["response"]["choices"][0]["message"]["content"] == "redacted-by-litellm"


@pytest.mark.asyncio
@pytest.mark.parametrize("turn_off_message_logging", [False, True])
async def test_logging_async_cache_hit_sync_call(
    provider_router: respx.MockRouter,
    callback_capture: _CallbackCapture,
    monkeypatch: pytest.MonkeyPatch,
    turn_off_message_logging: bool,
) -> None:
    monkeypatch.setattr(litellm, "cache", litellm.Cache(type="local"))
    monkeypatch.setattr(litellm, "turn_off_message_logging", turn_off_message_logging)
    messages: Final = [{"role": "user", "content": "cache marker"}]
    first: Final = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        stream=True,
        litellm_call_id=callback_capture.call_id,
    )
    first_chunks: Final = tuple([chunk async for chunk in first])
    await _drain_logging_worker()
    second: Final = await litellm.acompletion(
        model="gpt-4o-mini",
        messages=messages,
        caching=True,
        stream=True,
        litellm_call_id=callback_capture.call_id,
    )
    second_chunks: Final = tuple([chunk async for chunk in second])
    await _drain_logging_worker()
    assert "".join(chunk.choices[0].delta.content or "" for chunk in first_chunks) == "Hello"
    assert "".join(chunk.choices[0].delta.content or "" for chunk in second_chunks) == "Hello"
    assert provider_router.calls.call_count == 1
    await _assert_success_payload(callback_capture, "gpt-4o-mini", None, call_id=callback_capture.call_id)
    call: Final = callback_capture.async_success.call_args or callback_capture.sync_success.call_args
    event: Final = call.kwargs["kwargs"]
    standard: Final = event["standard_logging_object"]
    assert standard["cache_hit"] is True
    assert standard["response_cost"] == 0
    assert standard["saved_cache_cost"] > 0
    if turn_off_message_logging:
        assert standard["messages"] == [{"role": "user", "content": "redacted-by-litellm"}]
        assert standard["response"]["choices"][0]["message"]["content"] == "redacted-by-litellm"


@pytest.mark.parametrize("turn_off_message_logging", [False, True])
def test_logging_cache_hit_sync_stream_call(
    provider_router: respx.MockRouter,
    callback_capture: _CallbackCapture,
    monkeypatch: pytest.MonkeyPatch,
    turn_off_message_logging: bool,
) -> None:
    monkeypatch.setattr(litellm, "cache", litellm.Cache(type="local"))
    monkeypatch.setattr(litellm, "turn_off_message_logging", turn_off_message_logging)
    messages: Final = [{"role": "user", "content": "sync cache marker"}]
    first_chunks: Final = tuple(
        litellm.completion(model="gpt-4o-mini", messages=messages, caching=True, stream=True)
    )
    assert callback_capture.sync_logged.acquire(timeout=5)
    second_chunks: Final = tuple(
        litellm.completion(model="gpt-4o-mini", messages=messages, caching=True, stream=True)
    )
    assert callback_capture.sync_logged.acquire(timeout=5)
    assert "".join(chunk.choices[0].delta.content or "" for chunk in first_chunks) == "Hello"
    assert "".join(chunk.choices[0].delta.content or "" for chunk in second_chunks) == "Hello"
    assert provider_router.calls.call_count == 1
    standard: Final = callback_capture.sync_success.call_args.kwargs["kwargs"]["standard_logging_object"]
    assert standard["cache_hit"] is True
    assert standard["response_cost"] == 0
    assert standard["saved_cache_cost"] > 0
    if turn_off_message_logging:
        assert standard["messages"] == [{"role": "user", "content": "redacted-by-litellm"}]
        assert standard["response"]["choices"][0]["message"]["content"] == "redacted-by-litellm"


@pytest.mark.asyncio
async def test_logging_standard_payload_llm_headers(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    messages: Final = [{"role": "user", "content": "headers marker"}]
    await litellm.acompletion(model="gpt-4o-mini", messages=messages)
    await _assert_success_payload(callback_capture, "gpt-4o-mini", messages)
    event: Final = callback_capture.async_success.call_args.kwargs["kwargs"]
    assert event["model"] == "gpt-4o-mini"
    assert event["messages"] == [{"role": "user", "content": "headers marker"}]
    hidden_params: Final = event["standard_logging_object"]["hidden_params"]
    assert hidden_params["additional_headers"]["llm_provider-x-request-id"] == "respx-request-id"


@pytest.mark.asyncio
async def test_logging_key_masking_gemini(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    hidden_params: Final = StandardLoggingPayloadSetup.get_hidden_params({"api_key": "secret-gemini-key"})
    assert "secret-gemini-key" not in json.dumps(hidden_params)
    messages: Final = [{"role": "user", "content": "key masking marker"}]
    await litellm.acompletion(
        model="gemini/gemini-1.5-flash",
        messages=messages,
        api_key="secret-gemini-key",
    )
    await _assert_success_payload(callback_capture, "gemini/gemini-1.5-flash", messages)
    event: Final = callback_capture.async_success.call_args.kwargs["kwargs"]
    assert "secret-gemini-key" not in json.dumps(event["standard_logging_object"])
    assert event["standard_logging_object"]["model"] == "gemini-1.5-flash"


@pytest.mark.parametrize("turn_off_message_logging", [False, True])
@pytest.mark.parametrize("stream", [True, False])
@pytest.mark.asyncio
async def test_standard_logging_payload_audio(
    provider_router: respx.MockRouter,
    callback_capture: _CallbackCapture,
    monkeypatch: pytest.MonkeyPatch,
    stream: bool,
    turn_off_message_logging: bool,
) -> None:
    monkeypatch.setattr(litellm, "turn_off_message_logging", turn_off_message_logging)
    messages: Final = [{"role": "user", "content": "response in 1 word - yes or no"}]
    response: Final = await litellm.acompletion(
        model="openai/gpt-4o-audio-preview",
        modalities=["text", "audio"],
        audio={"voice": "alloy", "format": "pcm16"},
        messages=messages,
        stream=stream,
        litellm_call_id=callback_capture.call_id,
    )
    if stream:
        _ = [chunk async for chunk in response]
    assert json.loads(provider_router.calls[0].request.content)["model"] == "gpt-4o-audio-preview"
    await _assert_success_payload(
        callback_capture, "gpt-4o-audio-preview", None, call_id=callback_capture.call_id
    )
    call: Final = callback_capture.async_success.call_args or callback_capture.sync_success.call_args
    standard: Final = call.kwargs["kwargs"]["standard_logging_object"]
    assert standard["model"] == "gpt-4o-audio-preview"
    message: Final = standard["response"]["choices"][0]["message"]
    if turn_off_message_logging:
        assert standard["messages"] == [{"role": "user", "content": "redacted-by-litellm"}]
        assert message.get("audio") is None
        assert "hello world" not in json.dumps(standard)
        return
    audio: Final = message["audio"]
    assert audio["id"] == "audio-response"
    assert audio["data"] == base64.b64encode(b"foobar").decode("ascii")
    assert audio["transcript"] == "hello world"
    assert audio["expires_at"] == 1700003600


def test_completion_azure_stream_moderation_failure(
    provider_router: respx.MockRouter, callback_capture: _CallbackCapture
) -> None:
    messages: Final = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "filtered request"},
    ]
    with pytest.raises(litellm.ContentPolicyViolationError):
        litellm.completion(
            model="azure/gpt-4o-mini",
            messages=messages,
            stream=True,
            api_base="https://provider.test/v1",
            api_key="test-key",
        )
    failure_calls: Final = tuple(
        call
        for call in (*callback_capture.sync_failure.call_args_list, *callback_capture.async_failure.call_args_list)
        if call.kwargs["kwargs"].get("model") == "gpt-4o-mini" and call.kwargs["kwargs"].get("messages") == messages
    )
    assert len(failure_calls) == 1
    failure_event: Final = failure_calls[0].kwargs["kwargs"]
    assert failure_event["model"] == "gpt-4o-mini"
    assert failure_event["messages"] == messages
    assert failure_event["standard_logging_object"]["status"] == "failure"
