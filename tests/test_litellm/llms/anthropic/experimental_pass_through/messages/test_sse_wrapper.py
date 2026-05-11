import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, ModelResponseStream, StreamingChoices


# Create a simple test
class MockCompletionStream:
    def __init__(self):
        self.responses = [
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content="Hello"), index=0, finish_reason=None
                    )
                ],
            ),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content=" World"), index=0, finish_reason=None
                    )
                ],
            ),
            ModelResponseStream(
                choices=[
                    StreamingChoices(
                        delta=Delta(content=""), index=0, finish_reason="stop"
                    )
                ],
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
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(), model="claude-3"
    )

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
    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStream(), model="claude-3"
    )

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
                    choices=[
                        StreamingChoices(
                            delta=Delta(content="Hello"), index=0, finish_reason=None
                        )
                    ],
                ),
                ModelResponseStream(
                    choices=[
                        StreamingChoices(
                            delta=Delta(content=" World"), index=0, finish_reason=None
                        )
                    ],
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

    wrapper = AnthropicStreamWrapper(
        completion_stream=AsyncMockCompletionStream(), model="claude-3"
    )

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


def test_managed_messages_streaming_logging_ignores_openrouter_control_events():
    from unittest.mock import Mock

    from litellm.proxy.pass_through_endpoints.llm_provider_handlers.anthropic_passthrough_logging_handler import (
        AnthropicPassthroughLoggingHandler,
    )
    from litellm.types.utils import ModelResponse

    litellm_logging_obj = Mock()
    openrouter_chunks = [
        ": OPENROUTER PROCESSING",
        "data: : OPENROUTER PROCESSING",
        "event: message_start",
        'data: {"type":"message_start","message":{"id":"gen-test","type":"message","role":"assistant","model":"anthropic/claude-4.5-haiku-20251001","content":[],"stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":0,"output_tokens":0}}}',
        "event: content_block_start",
        'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}',
        "event: content_block_delta",
        'data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}',
        "event: content_block_stop",
        'data: {"type":"content_block_stop","index":0}',
        "event: message_delta",
        'data: {"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"input_tokens":8,"output_tokens":1,"cache_creation_input_tokens":0,"cache_read_input_tokens":0,"cost":0.000001}}',
        "event: message_stop",
        'data: {"type":"message_stop"}',
        "event: data",
        "data: [DONE]",
    ]

    result = AnthropicPassthroughLoggingHandler._build_complete_streaming_response(
        all_chunks=openrouter_chunks,
        model="claude-haiku-4-5-20251001-MODELFALLBACK",
        litellm_logging_obj=litellm_logging_obj,
    )

    assert isinstance(result, ModelResponse)
    assert result.usage.prompt_tokens == 8
    assert result.usage.completion_tokens == 1
