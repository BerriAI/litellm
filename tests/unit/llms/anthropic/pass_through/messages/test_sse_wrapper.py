import pytest

from litellm.llms.anthropic.pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


# Create a simple test
class MockCompletionStream:
    def __init__(self):
        self.responses = [
            ModelResponseStream(
                choices=[StreamingChoices(delta=Delta(content="Hello"), index=0, finish_reason=None)],
            ),
            ModelResponseStream(
                choices=[StreamingChoices(delta=Delta(content=" World"), index=0, finish_reason=None)],
            ),
            ModelResponseStream(
                choices=[StreamingChoices(delta=Delta(content=""), index=0, finish_reason="stop")],
            ),
        ]
        self.index = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.index >= len(self.responses):
            raise StopIteration
        response = self.responses[self.index]
        self.index += 1
        return response


def test_anthropic_sse_wrapper_format():
    """Test that the SSE wrapper produces proper event and data formatting"""
    wrapper = AnthropicStreamWrapper(completion_stream=MockCompletionStream(), model="claude-3")

    # Get the first chunk from the SSE wrapper
    first_chunk = next(wrapper.anthropic_sse_wrapper())

    # Verify it's bytes
    assert isinstance(first_chunk, bytes)

    # Decode and check format
    chunk_str = first_chunk.decode("utf-8")

    # Should have event line and data line
    lines = chunk_str.split("\n")
    assert len(lines) >= 3  # event line, data line, empty line (+ possibly more)
    assert lines[0].startswith("event: ")
    assert lines[1].startswith("data: ")
    assert lines[2] == ""  # Empty line to end the SSE chunk


def test_anthropic_sse_wrapper_event_types():
    """Test that different chunk types produce correct event types"""
    wrapper = AnthropicStreamWrapper(completion_stream=MockCompletionStream(), model="claude-3")

    chunks = []
    for chunk in wrapper.anthropic_sse_wrapper():
        chunks.append(chunk.decode("utf-8"))
        if len(chunks) >= 3:  # Get first few chunks
            break

    # First chunk should be message_start
    assert "event: message_start" in chunks[0]
    assert '"type": "message_start"' in chunks[0]

    # Second chunk should be content_block_start
    assert "event: content_block_start" in chunks[1]
    assert '"type": "content_block_start"' in chunks[1]

    # Third chunk should be content_block_delta
    assert "event: content_block_delta" in chunks[2]
    assert '"type": "content_block_delta"' in chunks[2]


@pytest.mark.asyncio
async def test_async_anthropic_sse_wrapper():
    """Test the async version of the SSE wrapper"""

    class AsyncMockCompletionStream:
        def __init__(self):
            self.responses = [
                ModelResponseStream(
                    choices=[StreamingChoices(delta=Delta(content="Hello"), index=0, finish_reason=None)],
                ),
                ModelResponseStream(
                    choices=[StreamingChoices(delta=Delta(content=" World"), index=0, finish_reason=None)],
                ),
            ]
            self.index = 0

        def __aiter__(self):
            return self

        async def __anext__(self):
            if self.index >= len(self.responses):
                raise StopAsyncIteration
            response = self.responses[self.index]
            self.index += 1
            return response

    wrapper = AnthropicStreamWrapper(completion_stream=AsyncMockCompletionStream(), model="claude-3")

    # Get the first chunk from the async SSE wrapper
    first_chunk = None
    async for chunk in wrapper.async_anthropic_sse_wrapper():
        first_chunk = chunk
        break

    # Verify it's bytes and properly formatted
    assert first_chunk is not None
    assert isinstance(first_chunk, bytes)

    chunk_str = first_chunk.decode("utf-8")
    assert "event: message_start" in chunk_str
    assert '"type": "message_start"' in chunk_str


class _FailingAsyncCompletionStream:
    """Async stream that emits one content chunk, then the provider dies."""

    def __init__(self):
        self.responses = [
            ModelResponseStream(
                choices=[StreamingChoices(delta=Delta(content="Hello"), index=0, finish_reason=None)],
            ),
        ]
        self.index = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index < len(self.responses):
            response = self.responses[self.index]
            self.index += 1
            return response
        raise Exception("Server disconnected")


class _MockLoggingObj:
    """Stand-in for LiteLLMLoggingObj covering the mid-stream failure surface."""

    def __init__(self):
        self._on_detached_stream_failure = None
        self.failure_handler_calls = []

    def record_streamed_anthropic_message_id(self, message_id):
        pass

    async def async_failure_handler(self, exception, traceback_exception, **kwargs):
        self.failure_handler_calls.append(exception)


@pytest.mark.asyncio
async def test_async_sse_wrapper_mid_stream_error_propagates_when_proxy_managed():
    """A proxy-managed stream re-raises the mid-stream error so the proxy's
    streaming boundary runs post_call_failure_hook (failure spend row + failure
    callbacks) and serializes the error frame itself. No frame is emitted here."""
    logging_obj = _MockLoggingObj()
    logging_obj._on_detached_stream_failure = lambda exc: None  # armed by the proxy

    wrapper = AnthropicStreamWrapper(
        completion_stream=_FailingAsyncCompletionStream(),
        model="claude-3",
        litellm_logging_obj=logging_obj,
    )

    received = []

    async def _drain():
        async for chunk in wrapper.async_anthropic_sse_wrapper():
            received.append(chunk)

    with pytest.raises(Exception, match="Server disconnected"):
        await _drain()

    # message_start was delivered; no adapter-level error frame followed
    assert any(b"message_start" in chunk for chunk in received)
    assert not any(b"event: error" in chunk for chunk in received)
    assert logging_obj.failure_handler_calls == []


@pytest.mark.asyncio
async def test_async_sse_wrapper_mid_stream_error_emits_frame_when_standalone():
    """Standalone (SDK) consumption has no proxy boundary downstream, so the
    mid-stream error still becomes a client-facing Anthropic error frame."""
    wrapper = AnthropicStreamWrapper(completion_stream=_FailingAsyncCompletionStream(), model="claude-3")

    received = []
    async for chunk in wrapper.async_anthropic_sse_wrapper():
        received.append(chunk)

    assert any(b"message_start" in chunk for chunk in received)
    error_frames = [chunk for chunk in received if b"event: error" in chunk]
    assert len(error_frames) == 1
    assert b"Server disconnected" in error_frames[0]


@pytest.mark.asyncio
async def test_async_sse_wrapper_runs_failure_handler_when_standalone():
    """Standalone consumption runs the logging object's failure handler before
    yielding the error frame, so failure logging fires without a proxy."""
    logging_obj = _MockLoggingObj()
    wrapper = AnthropicStreamWrapper(
        completion_stream=_FailingAsyncCompletionStream(),
        model="claude-3",
        litellm_logging_obj=logging_obj,
    )

    received = []
    async for chunk in wrapper.async_anthropic_sse_wrapper():
        received.append(chunk)

    assert len(logging_obj.failure_handler_calls) == 1
    assert "Server disconnected" in str(logging_obj.failure_handler_calls[0])
    assert any(b"event: error" in chunk for chunk in received)
