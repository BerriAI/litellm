import json
from collections.abc import AsyncIterator
from typing import Final, cast

import pytest

import litellm
from tests.test_litellm_rust.contracts import MESSAGES, MESSAGES_EVENTS, MESSAGES_MODEL, MESSAGES_RESPONSE
from tests.test_litellm_rust.recording_server import RecordingServer, ResponseSpec

pytestmark = pytest.mark.requires_rust_extension


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


async def call_messages(server: RecordingServer, **kwargs: object):
    return await litellm.anthropic.messages.acreate(
        model=MESSAGES_MODEL,
        messages=MESSAGES,
        max_tokens=64,
        api_key="test-key",
        api_base=server.base_url,
        **kwargs,
    )


def assert_native_request(server: RecordingServer) -> None:
    assert len(server.requests) == 1
    assert "accept-encoding" not in server.requests[0].headers


def assert_native_response(response: object) -> None:
    assert isinstance(response, dict)
    assert response["_hidden_params"]["additional_headers"] == {"x-litellm-rust": "true"}


@pytest.mark.asyncio
async def test_messages_sends_expected_provider_request(messages_server: RecordingServer) -> None:
    response: Final = await call_messages(messages_server)

    assert response["content"] == [{"type": "text", "text": "Hello from native Messages"}]
    assert_native_response(response)
    assert_native_request(messages_server)
    request: Final = messages_server.requests[0]
    assert request.path == "/v1/messages"
    assert request.headers["x-api-key"] == "test-key"
    assert request.headers["anthropic-version"] == "2023-06-01"
    assert request.body == {
        "model": "claude-sonnet-4-5-20250929",
        "messages": MESSAGES,
        "max_tokens": 64,
    }


@pytest.mark.asyncio
async def test_messages_sends_custom_headers(messages_server: RecordingServer) -> None:
    response: Final = await call_messages(messages_server, extra_headers={"x-trace-id": "trace-1"})

    assert_native_response(response)
    assert messages_server.requests[0].headers["x-trace-id"] == "trace-1"


@pytest.mark.asyncio
async def test_messages_resolves_provider_credentials(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key")

    response: Final = await litellm.anthropic.messages.acreate(
        model=MESSAGES_MODEL,
        messages=MESSAGES,
        max_tokens=64,
        api_base=messages_server.base_url,
    )

    assert_native_response(response)
    assert messages_server.requests[0].headers["x-api-key"] == "environment-key"


@pytest.mark.asyncio
async def test_messages_explicit_credentials_override_defaults(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "environment-key")

    response: Final = await call_messages(messages_server)

    assert_native_response(response)
    assert messages_server.requests[0].headers["x-api-key"] == "test-key"


@pytest.mark.asyncio
async def test_azure_messages_uses_foundry_endpoint_and_credentials(messages_server: RecordingServer) -> None:
    response: Final = await litellm.anthropic.messages.acreate(
        model="azure_ai/claude-opus-4.5",
        messages=MESSAGES,
        max_tokens=64,
        api_key="azure-key",
        api_base=f"{messages_server.base_url}/anthropic",
    )

    assert_native_response(response)
    assert_native_request(messages_server)
    assert messages_server.requests[0].path == "/anthropic/v1/messages"
    assert messages_server.requests[0].headers["x-api-key"] == "azure-key"


@pytest.mark.asyncio
async def test_messages_stream_yields_anthropic_events(messages_server: RecordingServer) -> None:
    messages_server.enqueue(ResponseSpec(body=None, events=MESSAGES_EVENTS))
    stream: Final = cast(AsyncIterator[bytes], await call_messages(messages_server, stream=True))
    payload: Final = b"".join([chunk async for chunk in stream])

    assert_native_request(messages_server)
    assert messages_server.requests[0].body["stream"] is True
    assert b"event: message_start" in payload
    assert b"event: content_block_delta" in payload
    assert b"Hello from native Messages" in payload
    assert b"event: message_stop" in payload
    message_delta: Final = next(
        json.loads(block.split(b"data: ", 1)[1])
        for block in payload.split(b"\n\n")
        if block.startswith(b"event: message_delta")
    )
    assert message_delta["usage"] == {"input_tokens": 5, "output_tokens": 4}


@pytest.mark.asyncio
@pytest.mark.parametrize("provider", ["anthropic", "azure_ai"])
async def test_messages_delivers_before_upstream_finishes(messages_server: RecordingServer, provider: str) -> None:
    import asyncio
    import threading

    from tests.test_litellm_rust.callback_recorder import RecordingLogger

    release: Final = threading.Event()
    recorder: Final = RecordingLogger()
    chunks: Final = tuple(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode() for event, data in MESSAGES_EVENTS)
    messages_server.enqueue(ResponseSpec(body=None, chunks=chunks, release=release))
    try:
        stream: Final = await asyncio.wait_for(
            litellm.anthropic.messages.acreate(
                model=MESSAGES_MODEL.split("/", 1)[-1],
                custom_llm_provider=provider,
                messages=MESSAGES,
                max_tokens=64,
                api_key="test-key",
                api_base=messages_server.base_url,
                stream=True,
                callbacks=[recorder],
            ),
            timeout=5,
        )
        first: Final = await asyncio.wait_for(anext(stream), timeout=2)
        assert b"message_start" in first
        assert not release.is_set()
        assert "async_log_success_event" not in recorder.names
        assert messages_server.requests[0].body["stream"] is True
        assert stream._hidden_params["additional_headers"]["x-litellm-rust"] == "true"
        assert messages_server.requests[0].path == (
            "/v1/messages" if provider == "anthropic" else "/anthropic/v1/messages"
        )
        release.set()
        remaining: Final = b"".join([chunk async for chunk in stream])
        assert first + remaining == b"".join(chunks)
        await stream.aclose()
        events: Final = await recorder.wait_for_async("async_log_success_event")
        assert len(events) == 1
        assert "async_log_failure_event" not in recorder.names
        assert events[0].response.choices[0].message.content == "Hello from native Messages"
        assert events[0].response.usage.completion_tokens == 4
        assert len(messages_server.requests) == 1
    finally:
        release.set()


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", ["truncated", "provider_error", "http_error"])
async def test_messages_stream_failures_do_not_replay(messages_server: RecordingServer, ending: str) -> None:
    from tests.test_litellm_rust.callback_recorder import RecordingLogger

    recorder: Final = RecordingLogger()
    events: Final = (
        MESSAGES_EVENTS[:-1]
        if ending == "truncated"
        else (
            *MESSAGES_EVENTS[:1],
            ("error", {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}),
        )
    )
    messages_server.enqueue(
        ResponseSpec(
            body={"error": "rejected"},
            status=429 if ending == "http_error" else 200,
            events=() if ending == "http_error" else events,
        )
    )

    async def consume() -> None:
        stream: Final = await call_messages(messages_server, stream=True, callbacks=[recorder])
        async for _ in stream:
            pass

    with pytest.raises(litellm.APIError):
        await consume()
    await recorder.wait_for_async("async_log_failure_event")
    assert recorder.names.count("async_log_failure_event") == 1
    assert "async_log_success_event" not in recorder.names
    assert len(messages_server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("abandon", ["close", "cancel", "drop", "exhaust"])
async def test_messages_stream_abandonment_releases_roots(messages_server: RecordingServer, abandon: str) -> None:
    import asyncio
    import gc
    import threading
    import weakref

    from tests.test_litellm_rust.callback_recorder import RecordingLogger

    class NamesOnlyLogger(RecordingLogger):
        def _record(self, name: str, kwargs: object = None, response: object = None) -> None:
            super()._record(name)

    class Root:
        pass

    root = Root()
    reference: Final = weakref.ref(root)
    release: Final = threading.Event()
    recorder: Final = NamesOnlyLogger()
    chunks: Final = tuple(f"event: {event}\ndata: {json.dumps(data)}\n\n".encode() for event, data in MESSAGES_EVENTS)
    messages_server.enqueue(ResponseSpec(body=None, chunks=chunks, release=release))
    try:
        stream = await call_messages(messages_server, stream=True, callbacks=[recorder], metadata={"retained": root})
        del root
        assert reference() is not None
        assert b"message_start" in await anext(stream)
        if abandon == "exhaust":
            release.set()
            async for _ in stream:
                pass
            await recorder.wait_for_async("async_log_success_event")
            gc.collect()
            assert reference() is None
            assert recorder.names.count("async_log_success_event") == 1
            return
        if abandon == "cancel":
            task: Final = asyncio.create_task(anext(stream))
            await asyncio.sleep(0.05)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
        elif abandon == "close":
            await stream.aclose()
            await stream.aclose()
        del stream
        gc.collect()
        await recorder.wait_for_async("async_log_failure_event")
        assert recorder.names.count("async_log_failure_event") == 1
        assert "async_log_success_event" not in recorder.names
        assert len(messages_server.requests) == 1
        gc.collect()
        assert reference() is None
    finally:
        release.set()
