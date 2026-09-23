import asyncio
from collections.abc import AsyncIterator, Iterator
from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.integrations.custom_logger import CustomLogger
from litellm.rust_bridge.catalog import CacheRule
from litellm.rust_bridge.configuration import Rollout
from litellm.rust_bridge.response_cache import resolve_response_cache
from tests.test_litellm_rust.support.callback_recorder import RecordingLogger, drain_logging
from tests.test_litellm_rust.support.recording_server import RecordingServer, ResponseSpec
from tests.test_litellm_rust.support.requests import (
    MESSAGES,
    MESSAGES_EVENTS,
    MESSAGES_MODEL,
    MESSAGES_RESPONSE,
    request_body,
)

pytestmark = pytest.mark.requires_rust_extension

STREAM: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS)


@pytest.fixture
def messages_server(recording_server: RecordingServer) -> RecordingServer:
    recording_server.default_response = ResponseSpec(body=MESSAGES_RESPONSE)
    return recording_server


def arguments(server: RecordingServer, **kwargs: object) -> dict[str, object]:
    return {
        "model": MESSAGES_MODEL,
        "messages": [dict(message) for message in MESSAGES],
        "max_tokens": 64,
        "api_key": "test-key",
        "api_base": server.base_url,
        **kwargs,
    }


def assert_served_natively(server: RecordingServer) -> None:
    assert len(server.requests) == 1
    assert not server.requests[0].headers.get("user-agent", "").startswith("python-httpx")


@pytest.mark.asyncio
async def test_native_messages_callbacks_see_the_provider_request_and_the_public_response(
    messages_server: RecordingServer,
) -> None:
    await drain_logging()
    recorder: Final = RecordingLogger()

    response: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, callbacks=[recorder], litellm_call_id="messages-success")
    )

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]
    sent: Final = messages_server.requests[0]
    assert sent.path == "/v1/messages"
    assert sent.body == {"model": "claude-sonnet-5", "messages": list(MESSAGES), "max_tokens": 64, "stream": False}
    pre_call: Final = recorder.wait_for("log_pre_api_call")
    assert request_body(pre_call[0].kwargs) == sent.body
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].call_type == "anthropic_messages"
    assert success[0].kwargs["litellm_call_id"] == "messages-success"
    assert success[0].response.choices[0].message.content == "Hello from native Messages"


@pytest.mark.asyncio
async def test_native_messages_pre_call_body_edit_reaches_the_provider(messages_server: RecordingServer) -> None:
    class Edit(CustomLogger):
        def log_pre_api_call(self, model, messages, kwargs):
            request_body(kwargs)["temperature"] = 0.25

    await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[Edit()]))

    assert messages_server.requests[0].body["temperature"] == 0.25


@pytest.mark.parametrize(
    "context_management",
    (
        {"edits": [{"type": "compact_20260112"}]},
        [{"type": "compaction", "compact_threshold": 2048}],
    ),
)
@pytest.mark.asyncio
async def test_native_messages_matches_python_request_cleanup_and_beta_headers(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, context_management: object
) -> None:
    messages_server.expected_requests = 2
    messages: Final = [
        {
            "role": "assistant",
            "content": [
                {"type": "text", "text": "  "},
                {"type": "tool_use", "id": "functions.Bash:0", "name": "Bash", "input": {}},
                {"type": "text", "text": "hello", "provider_specific_fields": {"source": "other"}},
            ],
        }
    ]
    kwargs: Final = arguments(
        messages_server,
        messages=messages,
        metadata={"user_id": "user", "internal": "private"},
        system="Keep it short",
        stop_sequences=["END"],
        temperature=1,
        tools=[{"name": "Bash", "input_schema": {"type": "object"}}],
        tool_choice={"type": "auto"},
        context_management=context_management,
        extra_headers={"x-request-id": "parity"},
        provider_specific_header=[
            {"custom_llm_provider": "azure_ai", "extra_headers": {"x-ignored": "other"}},
            {"custom_llm_provider": "anthropic, bedrock", "extra_headers": {"x-scope": "selected"}},
        ],
    )

    monkeypatch.setenv("LITELLM_RUST", "0")
    await litellm.anthropic.messages.acreate(**kwargs)
    monkeypatch.setenv("LITELLM_RUST", "1")
    await litellm.anthropic.messages.acreate(**kwargs)

    python_request, rust_request = messages_server.requests
    assert python_request.body == rust_request.body
    assert python_request.headers["anthropic-beta"] == rust_request.headers["anthropic-beta"]
    assert rust_request.headers["x-request-id"] == "parity"
    assert rust_request.headers["x-scope"] == "selected"
    assert "x-ignored" not in rust_request.headers
    assert not rust_request.headers.get("user-agent", "").startswith("python-httpx")


@pytest.mark.asyncio
async def test_native_messages_stream_preserves_python_sse_event_order(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages_server.expected_requests = 2
    messages_server.enqueue(STREAM)
    messages_server.enqueue(STREAM)

    monkeypatch.setenv("LITELLM_RUST", "0")
    python_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    python_events: Final = b"".join([chunk async for chunk in python_stream])

    monkeypatch.setenv("LITELLM_RUST", "1")
    rust_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    rust_events: Final = b"".join([chunk async for chunk in rust_stream])

    assert python_events == rust_events == sse_payload()
    assert messages_server.requests[0].body == messages_server.requests[1].body


@pytest.mark.asyncio
async def test_native_messages_stream_preserves_incomplete_provider_response(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages_server.expected_requests = 2
    incomplete: Final = ResponseSpec(body=None, events=MESSAGES_EVENTS[:-1])
    messages_server.enqueue(incomplete)
    messages_server.enqueue(incomplete)

    monkeypatch.setenv("LITELLM_RUST", "0")
    python_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    python_events: Final = b"".join([chunk async for chunk in python_stream])

    monkeypatch.setenv("LITELLM_RUST", "1")
    rust_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    rust_events: Final = b"".join([chunk async for chunk in rust_stream])

    assert python_events == rust_events == b"".join(incomplete.payloads())
    assert b"event: message_stop" not in rust_events


@pytest.mark.asyncio
async def test_native_messages_provider_error_reaches_caller_and_failure_callbacks_as_one_public_error(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(
        ResponseSpec(body={"type": "error", "error": {"type": "invalid_request_error", "message": "bad"}}, status=400)
    )
    observed: Final = []

    class Observe(CustomLogger):
        def log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("sync", kwargs["exception"]))

        async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
            observed.append(("async", kwargs["exception"]))

    with pytest.raises(litellm.BadRequestError) as raised:
        await litellm.anthropic.messages.acreate(**arguments(messages_server, callbacks=[Observe()]))

    assert_served_natively(messages_server)
    assert [phase for phase, _ in observed] == ["sync", "async"]
    assert all(error is raised.value for _, error in observed)


@pytest.mark.asyncio
async def test_native_messages_retries_invalid_thinking_signature_with_clean_history(
    messages_server: RecordingServer,
) -> None:
    messages_server.expected_requests = 2
    messages_server.enqueue(
        ResponseSpec(
            body={"type": "error", "error": {"type": "invalid_request_error", "message": "Invalid signature in thinking block"}},
            status=400,
        )
    )
    recorder: Final = RecordingLogger()
    messages: Final = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "previous", "signature": "expired"},
                {"type": "text", "text": "answer"},
            ],
        },
        {"role": "user", "content": "continue"},
    ]

    response: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, messages=messages, callbacks=[recorder])
    )

    assert response["content"] == MESSAGES_RESPONSE["content"]
    assert [block["type"] for block in messages_server.requests[0].body["messages"][0]["content"]] == [
        "thinking", "text"
    ]
    assert messages_server.requests[1].body["messages"][0]["content"] == [{"type": "text", "text": "answer"}]
    assert len(recorder.wait_for("log_pre_api_call")) == 1
    assert len(await recorder.wait_for_async("async_log_success_event")) == 1


@pytest.mark.asyncio
async def test_provider_changing_hook_runs_once_before_native_request(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    messages_server.expected_requests = 0

    class Reroute(CustomLogger):
        calls = 0

        async def async_pre_request_hook(self, model, messages, kwargs):
            self.calls += 1
            return {**kwargs, "litellm_params": {"custom_llm_provider": "openai"}, "mock_response": "rerouted"}

    hook: Final = Reroute()
    monkeypatch.setattr(litellm, "callbacks", [hook])

    response: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert response["content"][0]["text"] == "rerouted"
    assert hook.calls == 1
    assert messages_server.requests == []


@pytest.mark.parametrize("native_cache", (False, True))
@pytest.mark.asyncio
async def test_native_messages_share_the_configured_response_cache(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, native_cache: bool
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    if native_cache:
        cache._native_cache = resolve_response_cache(  # pyright: ignore[reportPrivateUsage]  # select the native store for this cache integration test
            cache,
            rules=(CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.LOCAL})),),
        )
        assert cache._native_cache is not None  # pyright: ignore[reportPrivateUsage]  # confirm the selected backend
    monkeypatch.setattr(litellm, "cache", cache)

    first: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))
    await asyncio.sleep(0.05)
    second: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert first == second
    assert_served_natively(messages_server)


@pytest.mark.asyncio
async def test_native_messages_no_cache_replaces_stored_response(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(litellm, "cache", Cache(type=LiteLLMCacheType.LOCAL))
    messages_server.expected_requests = 2
    replacement: Final = {
        **MESSAGES_RESPONSE,
        "id": "msg_replacement",
        "content": [{"type": "text", "text": "replacement"}],
    }
    messages_server.enqueue(ResponseSpec(body=MESSAGES_RESPONSE))
    messages_server.enqueue(ResponseSpec(body=replacement))

    first: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))
    await asyncio.sleep(0.05)
    refreshed: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, cache={"no-cache": True})
    )
    await asyncio.sleep(0.05)
    cached: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server))

    assert first["id"] != refreshed["id"]
    assert cached == refreshed
    assert len(messages_server.requests) == 2


@pytest.mark.parametrize("native_cache", (False, True))
@pytest.mark.asyncio
async def test_native_messages_stream_is_replayed_from_the_configured_cache(
    messages_server: RecordingServer, monkeypatch: pytest.MonkeyPatch, native_cache: bool
) -> None:
    cache: Final = Cache(type=LiteLLMCacheType.LOCAL)
    if native_cache:
        cache._native_cache = resolve_response_cache(  # pyright: ignore[reportPrivateUsage]  # select the native store for this cache integration test
            cache,
            rules=(CacheRule(Rollout.RUST_OPT_OUT, backends=frozenset({LiteLLMCacheType.LOCAL})),),
        )
        assert cache._native_cache is not None  # pyright: ignore[reportPrivateUsage]  # confirm the selected backend
    monkeypatch.setattr(litellm, "cache", cache)
    messages_server.enqueue(STREAM)

    first_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    first: Final = b"".join([chunk async for chunk in first_stream])
    second_stream: Final = await litellm.anthropic.messages.acreate(**arguments(messages_server, stream=True))
    second: Final = b"".join([chunk async for chunk in second_stream])

    assert first == second == sse_payload()
    assert second_stream._hidden_params["cache_hit"] is True
    assert_served_natively(messages_server)


def sse_payload() -> bytes:
    return b"".join(STREAM.payloads())


@pytest.mark.asyncio
async def test_native_messages_stream_relays_provider_events_and_logs_success_once_after_the_last_chunk(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    first: Final = await anext(stream)
    await drain_logging()
    assert "async_log_success_event" not in recorder.names
    rest: Final = [chunk async for chunk in stream]

    assert first + b"".join(rest) == sse_payload()
    assert_served_natively(messages_server)
    assert messages_server.requests[0].body["stream"] is True
    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    assert success[0].kwargs["stream"] is True
    assert success[0].kwargs["completion_start_time"] is not None
    assert "log_failure_event" not in recorder.names


@pytest.mark.asyncio
async def test_native_messages_stream_closed_early_logs_success_once_for_what_was_delivered(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = await litellm.anthropic.messages.acreate(
        **arguments(messages_server, stream=True, callbacks=[recorder])
    )
    assert isinstance(stream, AsyncIterator)
    await anext(stream)
    await stream.aclose()

    success: Final = await recorder.wait_for_async("async_log_success_event")
    assert len(success) == 1
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


def test_native_sync_messages_stream_relays_provider_events_and_logs_success_once(
    messages_server: RecordingServer,
) -> None:
    messages_server.enqueue(STREAM)
    recorder: Final = RecordingLogger()

    stream: Final = litellm.anthropic.messages.create(**arguments(messages_server, stream=True, callbacks=[recorder]))
    assert isinstance(stream, Iterator)

    assert b"".join(stream) == sse_payload()
    assert_served_natively(messages_server)
    assert len(recorder.wait_for("async_log_success_event")) == 1


def test_native_sync_messages_returns_the_provider_message(messages_server: RecordingServer) -> None:
    recorder: Final = RecordingLogger()

    response: Final = litellm.anthropic.messages.create(**arguments(messages_server, callbacks=[recorder]))

    assert_served_natively(messages_server)
    assert response["content"] == MESSAGES_RESPONSE["content"]
    assert len(recorder.wait_for("log_success_event")) == 1
