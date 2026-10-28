"""
Tests for AnthropicSSEStream, the object translate_completion_output_params_streaming
hands to the proxy for /v1/messages streaming. It must emit the same SSE bytes as
the wrapper's async_anthropic_sse_wrapper, propagate aclose into it, and expose the
wrapper's chunks/messages/model so disconnect-time partial billing can read them.
"""

from typing import Final
from unittest.mock import MagicMock

import httpx
import pytest

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.anthropic.messages.adapters.streaming_iterator import (
    AnthropicSSEStream,
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, StreamingChoices


def _make_chunk(delta: Delta, finish_reason: str | None = None) -> MagicMock:
    chunk = MagicMock()
    chunk.choices = [StreamingChoices(finish_reason=finish_reason, index=0, delta=delta, logprobs=None)]
    chunk.usage = None
    chunk._hidden_params = {}
    return chunk


class _AsyncStream:
    def __init__(self, items: list[MagicMock]):
        self._it = iter(items)
        self.chunks = list(items)
        self.messages: list[dict] = [{"role": "user", "content": "hi"}]

    def __aiter__(self):
        return self

    async def __anext__(self):
        try:
            return next(self._it)
        except StopIteration:
            raise StopAsyncIteration


def _streamed_events() -> AnthropicSSEStream:
    upstream: Final = _AsyncStream(
        [
            _make_chunk(Delta(content="Once")),
            _make_chunk(Delta(content=" upon"), finish_reason="stop"),
        ]
    )
    wrapper: Final = AnthropicStreamWrapper(completion_stream=upstream, model="gpt-4o-mini")
    wrapper._message_id = "msg_test"
    return AnthropicSSEStream(wrapper)


@pytest.mark.asyncio
async def test_sse_stream_yields_identical_bytes_to_the_wrappers_sse_wrapper():
    upstream_a: Final = _AsyncStream(
        [_make_chunk(Delta(content="Once")), _make_chunk(Delta(content=" upon"), finish_reason="stop")]
    )
    wrapper_a: Final = AnthropicStreamWrapper(completion_stream=upstream_a, model="gpt-4o-mini")
    wrapper_a._message_id = "msg_test"
    expected: Final = [event async for event in wrapper_a.async_anthropic_sse_wrapper()]

    actual: Final = [event async for event in _streamed_events()]

    assert actual == expected


@pytest.mark.asyncio
async def test_sse_stream_aclose_ends_the_wrapped_stream():
    stream: Final = _streamed_events()

    first: Final = await stream.__anext__()
    assert first.startswith(b"event: message_start")
    await stream.aclose()
    with pytest.raises(StopAsyncIteration):
        await stream.__anext__()


def test_sse_stream_exposes_chunks_messages_and_model():
    stream: Final = _streamed_events()

    assert stream.model == "gpt-4o-mini"
    assert stream.messages == [{"role": "user", "content": "hi"}]
    chunks: Final = stream.chunks
    assert isinstance(chunks, list) and len(chunks) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("started", [False, True])
async def test_sse_stream_close_releases_http_response_before_upstream_body(started: bool) -> None:
    response: Final = httpx.Response(200, stream=httpx.ByteStream(b"unread upstream body"))
    upstream: Final = CustomStreamWrapper(
        completion_stream=response,
        model="test-model",
        custom_llm_provider="openai",
        logging_obj=Logging(
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="acompletion",
            start_time=1.0,
            litellm_call_id="disconnect-test",
            function_id="test",
        ),
    )
    stream: Final = AnthropicSSEStream(AnthropicStreamWrapper(completion_stream=upstream, model="test-model"))
    if started:
        assert (await anext(stream)).startswith(b"event: message_start")
    assert not response.is_stream_consumed
    assert not response.is_closed

    await stream.aclose()
    await stream.aclose()

    assert response.is_closed
    assert not response.is_stream_consumed
    with pytest.raises(StopAsyncIteration):
        await anext(stream)


class _CloseableAsyncStream(_AsyncStream):
    def __init__(self, items: list[MagicMock]) -> None:
        super().__init__(items)
        self.close_calls = 0

    async def aclose(self) -> None:
        self.close_calls += 1


@pytest.mark.asyncio
@pytest.mark.parametrize("finish_stream", [False, True])
async def test_sse_stream_releases_upstream_once_after_content(finish_stream: bool) -> None:
    upstream: Final = _CloseableAsyncStream(
        [_make_chunk(Delta(content="hello")), _make_chunk(Delta(), finish_reason="stop")]
    )
    stream: Final = AnthropicSSEStream(AnthropicStreamWrapper(completion_stream=upstream, model="test-model"))
    events: Final[list[bytes]] = []
    async for event in stream:
        events.append(event)
        if not finish_stream and b"event: content_block_delta" in event:
            break
    assert any(b'"text":"hello"' in event.replace(b" ", b"") for event in events)
    if finish_stream:
        assert events[-1].startswith(b"event: message_stop")

    await stream.aclose()
    await stream.aclose()

    assert upstream.close_calls == 1
