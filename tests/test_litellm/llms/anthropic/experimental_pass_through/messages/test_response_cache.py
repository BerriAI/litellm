import asyncio
import os
import sys
from typing import Any, AsyncIterator, Dict, List

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

import litellm
from litellm.caching.caching import Cache, LiteLLMCacheType
from litellm.llms.anthropic.experimental_pass_through.messages import handler

STREAM_EVENTS: List[bytes] = [
    b'event: message_start\ndata: {"type": "message_start", "message": {"id": "msg_stream_1", "type": "message", '
    b'"role": "assistant", "model": "claude-sonnet-4-5", "content": [], "stop_reason": null, '
    b'"usage": {"input_tokens": 10, "output_tokens": 0}}}\n\n',
    b'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, '
    b'"content_block": {"type": "text", "text": ""}}\n\n',
    b'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, '
    b'"delta": {"type": "text_delta", "text": "ALPHA"}}\n\n',
    b'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}\n\n',
    b'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, '
    b'"usage": {"output_tokens": 3}}\n\n',
    b'event: message_stop\ndata: {"type": "message_stop"}\n\n',
]


def _anthropic_response(message_id: str, text: str) -> Dict[str, Any]:
    return {
        "id": message_id,
        "type": "message",
        "role": "assistant",
        "model": "claude-sonnet-4-5",
        "content": [{"type": "text", "text": text}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 10, "output_tokens": 3},
    }


class _CountingHandler:
    """Stands in for the provider dispatch so cache hits are observable as skipped calls."""

    def __init__(self, results: List[Any]) -> None:
        self.results = results
        self.calls: List[Dict[str, Any]] = []

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.results[min(len(self.calls) - 1, len(self.results) - 1)]


async def _byte_stream(chunks: List[bytes]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


async def _collect(stream: AsyncIterator[bytes]) -> List[bytes]:
    return [chunk async for chunk in stream]


@pytest.fixture
def local_cache():
    previous_cache = litellm.cache
    litellm.cache = Cache(type=LiteLLMCacheType.LOCAL)
    yield litellm.cache
    litellm.cache = previous_cache


@pytest.fixture
def request_kwargs() -> Dict[str, Any]:
    return {
        "model": "anthropic/claude-sonnet-4-5",
        "custom_llm_provider": "anthropic",
        "api_key": "fake-key",
        "max_tokens": 64,
        "messages": [{"role": "user", "content": "which greek letter?"}],
    }


@pytest.mark.asyncio
async def test_non_streaming_request_is_served_from_cache(local_cache, request_kwargs, monkeypatch):
    fake_handler = _CountingHandler([_anthropic_response("msg_1", "ALPHA"), _anthropic_response("msg_2", "BETA")])
    monkeypatch.setattr(handler, "anthropic_messages_handler", fake_handler)

    first = await litellm.anthropic_messages(**request_kwargs)
    await asyncio.sleep(0)
    second = await litellm.anthropic_messages(**request_kwargs)

    assert len(fake_handler.calls) == 1
    assert first == second
    assert second["content"][0]["text"] == "ALPHA"


@pytest.mark.asyncio
async def test_cache_key_separates_different_system_prompts(local_cache, request_kwargs, monkeypatch):
    """`system` has no OpenAI equivalent; if it is dropped from the cache key the
    second request is answered with the first system prompt's response."""
    fake_handler = _CountingHandler([_anthropic_response("msg_1", "ALPHA"), _anthropic_response("msg_2", "BETA")])
    monkeypatch.setattr(handler, "anthropic_messages_handler", fake_handler)

    first = await litellm.anthropic_messages(**request_kwargs, system="Always answer ALPHA")
    await asyncio.sleep(0)
    second = await litellm.anthropic_messages(**request_kwargs, system="Always answer BETA")

    assert len(fake_handler.calls) == 2
    assert first["content"][0]["text"] == "ALPHA"
    assert second["content"][0]["text"] == "BETA"


@pytest.mark.parametrize("anthropic_param", [{"top_k": 5}, {"stop_sequences": ["STOP"]}])
@pytest.mark.asyncio
async def test_cache_key_separates_anthropic_native_params(local_cache, request_kwargs, monkeypatch, anthropic_param):
    fake_handler = _CountingHandler([_anthropic_response("msg_1", "ALPHA"), _anthropic_response("msg_2", "BETA")])
    monkeypatch.setattr(handler, "anthropic_messages_handler", fake_handler)

    await litellm.anthropic_messages(**request_kwargs)
    await asyncio.sleep(0)
    await litellm.anthropic_messages(**request_kwargs, **anthropic_param)

    assert len(fake_handler.calls) == 2


@pytest.mark.asyncio
async def test_streaming_request_is_replayed_from_cache(local_cache, request_kwargs, monkeypatch):
    fake_handler = _CountingHandler([_byte_stream(STREAM_EVENTS), _byte_stream([b"event: never_used\n\n"])])
    monkeypatch.setattr(handler, "anthropic_messages_handler", fake_handler)

    first = await _collect(await litellm.anthropic_messages(**request_kwargs, stream=True))
    second_stream = await litellm.anthropic_messages(**request_kwargs, stream=True)
    second = await _collect(second_stream)

    assert len(fake_handler.calls) == 1
    assert first == STREAM_EVENTS
    assert second == STREAM_EVENTS
    assert second_stream._hidden_params["cache_hit"] is True


@pytest.mark.asyncio
async def test_streaming_cache_is_not_shared_with_non_streaming(local_cache, request_kwargs, monkeypatch):
    fake_handler = _CountingHandler([_byte_stream(STREAM_EVENTS), _anthropic_response("msg_2", "ALPHA")])
    monkeypatch.setattr(handler, "anthropic_messages_handler", fake_handler)

    await _collect(await litellm.anthropic_messages(**request_kwargs, stream=True))
    non_streaming = await litellm.anthropic_messages(**request_kwargs)

    assert len(fake_handler.calls) == 2
    assert non_streaming["content"][0]["text"] == "ALPHA"


@pytest.mark.asyncio
async def test_failed_stream_is_not_cached(local_cache, request_kwargs, monkeypatch):
    error_events = STREAM_EVENTS[:3] + [
        b'event: error\ndata: {"type": "error", "error": {"type": "overloaded_error", "message": "overloaded"}}\n\n'
    ]
    fake_handler = _CountingHandler([_byte_stream(error_events), _byte_stream(STREAM_EVENTS)])
    monkeypatch.setattr(handler, "anthropic_messages_handler", fake_handler)

    failed = await _collect(await litellm.anthropic_messages(**request_kwargs, stream=True))
    replayed = await _collect(await litellm.anthropic_messages(**request_kwargs, stream=True))

    assert failed == error_events
    assert len(fake_handler.calls) == 2
    assert replayed == STREAM_EVENTS


@pytest.mark.asyncio
async def test_abandoned_stream_is_not_cached(local_cache, request_kwargs, monkeypatch):
    fake_handler = _CountingHandler([_byte_stream(STREAM_EVENTS), _byte_stream(STREAM_EVENTS)])
    monkeypatch.setattr(handler, "anthropic_messages_handler", fake_handler)

    partial_stream = await litellm.anthropic_messages(**request_kwargs, stream=True)
    await partial_stream.__anext__()
    await partial_stream.aclose()

    replayed = await _collect(await litellm.anthropic_messages(**request_kwargs, stream=True))

    assert len(fake_handler.calls) == 2
    assert replayed == STREAM_EVENTS

@pytest.mark.asyncio
async def test_cached_stream_replay_logs_once_when_polled_after_exhaustion():
    from unittest.mock import AsyncMock, MagicMock, patch

    from litellm.llms.anthropic.experimental_pass_through.messages.response_cache import (
        CachedAnthropicMessagesStreamIterator,
    )
    from litellm.proxy.pass_through_endpoints.streaming_handler import (
        PassThroughStreamingHandler,
    )

    logging_obj = MagicMock()
    logging_obj.model_call_details = {}
    iterator = CachedAnthropicMessagesStreamIterator(
        events=[event.decode("utf-8") for event in STREAM_EVENTS],
        litellm_logging_obj=logging_obj,
        request_body={"model": "claude-sonnet-4-5"},
    )

    with patch.object(
        PassThroughStreamingHandler,
        "_route_streaming_logging_to_handler",
        new=AsyncMock(),
    ) as mock_route:
        assert await _collect(iterator) == STREAM_EVENTS
        for _ in range(2):
            with pytest.raises(StopAsyncIteration):
                await iterator.__anext__()
        await asyncio.sleep(0)

    mock_route.assert_called_once()
