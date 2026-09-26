"""
Tests for AnthropicSSEStream, the object translate_completion_output_params_streaming
hands to the proxy for /v1/messages streaming. It must emit the same SSE bytes as
the wrapper's async_anthropic_sse_wrapper, propagate aclose into it, and expose the
wrapper's chunks/messages/model so disconnect-time partial billing can read them.
"""

from typing import Final
from unittest.mock import MagicMock

import pytest

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
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
