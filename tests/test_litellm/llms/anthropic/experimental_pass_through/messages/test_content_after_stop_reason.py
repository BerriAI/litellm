"""
Test for AnthropicStreamWrapper handling content blocks that exist after message_delta with stop_reason and usage.

This tests the scenario where a streaming response includes:
1. Initial content blocks
2. A message_delta chunk with stop_reason and usage
3. Additional content blocks after the stop_reason

The wrapper should properly handle this by:
- Holding the stop_reason chunk until usage is available
- Merging usage into the stop_reason chunk
- Properly managing content_block_stop/start events for subsequent content
"""

import os
import sys
from typing import List

import pytest

sys.path.insert(0, os.path.abspath("../../../../.."))

from litellm.llms.anthropic.experimental_pass_through.adapters.streaming_iterator import (
    AnthropicStreamWrapper,
)
from litellm.types.utils import Delta, ModelResponse, StreamingChoices, Usage


class MockCompletionStreamWithContentAfterStopReason:
    """Mock stream that simulates content blocks existing after message_delta with stop_reason and usage."""

    def __init__(self):
        self.responses = [
            # Initial text content
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(content="Hello"), index=0, finish_reason=None
                    )
                ],
            ),
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(content=" world"), index=0, finish_reason=None
                    )
                ],
            ),
            # Message delta with stop_reason AND usage (this is how it actually comes from the API)
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(content=""), index=0, finish_reason="stop"
                    )
                ],
                usage=Usage(prompt_tokens=230, completion_tokens=65, total_tokens=295),
            ),
            # Additional content after the stop_reason - this simulates the scenario
            # where there might be additional content blocks after the main response
            ModelResponse(
                stream=True,
                choices=[
                    StreamingChoices(
                        delta=Delta(content=" Additional content"),
                        index=0,
                        finish_reason=None,
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

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self.index >= len(self.responses):
            raise StopAsyncIteration
        response = self.responses[self.index]
        self.index += 1
        return response


def test_anthropic_stream_wrapper_content_after_stop_reason():
    """Test that AnthropicStreamWrapper properly handles content blocks after message_delta with stop_reason."""

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamWithContentAfterStopReason(),
        model="claude-3",
    )

    chunks = []
    chunk_types = []

    # Collect all chunks
    for chunk in wrapper:
        chunks.append(chunk)
        chunk_types.append(chunk.get("type"))

    # Verify the expected sequence of chunk types
    expected_types = [
        "message_start",  # Initial message start
        "content_block_start",  # Start of first content block
        "content_block_delta",  # "Hello"
        "content_block_delta",  # " world"
        "content_block_stop",  # End of first content block due to stop_reason
        "message_delta",  # Stop reason with merged usage
        "message_stop",  # Final message stop
    ]

    print(f"Actual chunk types: {chunk_types}")
    print(f"Expected chunk types: {expected_types}")

    # Verify we have the expected number of chunks
    assert len(chunk_types) >= len(
        expected_types
    ), f"Expected at least {len(expected_types)} chunks, got {len(chunk_types)}"

    # Verify key chunk types are present
    assert "message_start" in chunk_types
    assert "content_block_start" in chunk_types
    assert "content_block_delta" in chunk_types
    assert "content_block_stop" in chunk_types
    assert "message_delta" in chunk_types
    assert "message_stop" in chunk_types

    # Find the message_delta chunk with stop_reason
    message_delta_chunk = None
    for chunk in chunks:
        if chunk.get("type") == "message_delta":
            message_delta_chunk = chunk
            break

    assert message_delta_chunk is not None, "message_delta chunk not found"

    # Verify that the message_delta chunk has both stop_reason and usage
    delta = message_delta_chunk.get("delta", {})
    usage = message_delta_chunk.get("usage", {})

    assert (
        delta.get("stop_reason") == "end_turn"
    ), f"Expected stop_reason 'end_turn', got {delta.get('stop_reason')}"
    assert (
        usage.get("input_tokens") == 230
    ), f"Expected input_tokens 230, got {usage.get('input_tokens')}"
    assert (
        usage.get("output_tokens") == 65
    ), f"Expected output_tokens 65, got {usage.get('output_tokens')}"

    # Verify content_block_stop comes before message_delta
    content_block_stop_index = None
    message_delta_index = None

    for i, chunk_type in enumerate(chunk_types):
        if chunk_type == "content_block_stop" and content_block_stop_index is None:
            content_block_stop_index = i
        elif chunk_type == "message_delta":
            message_delta_index = i

    assert content_block_stop_index is not None, "content_block_stop not found"
    assert message_delta_index is not None, "message_delta not found"
    assert (
        content_block_stop_index < message_delta_index
    ), "content_block_stop should come before message_delta"


@pytest.mark.asyncio
async def test_async_anthropic_stream_wrapper_content_after_stop_reason():
    """Test async version of AnthropicStreamWrapper handling content blocks after message_delta with stop_reason."""

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamWithContentAfterStopReason(),
        model="claude-3",
    )

    chunks = []
    chunk_types = []

    # Collect all chunks asynchronously
    async for chunk in wrapper:
        chunks.append(chunk)
        chunk_types.append(chunk.get("type"))

    print(f"Async - Actual chunk types: {chunk_types}")

    # Verify key chunk types are present
    assert "message_start" in chunk_types
    assert "content_block_start" in chunk_types
    assert "content_block_delta" in chunk_types
    assert "content_block_stop" in chunk_types
    assert "message_delta" in chunk_types
    assert "message_stop" in chunk_types

    # Find the message_delta chunk with stop_reason
    message_delta_chunk = None
    for chunk in chunks:
        if chunk.get("type") == "message_delta":
            message_delta_chunk = chunk
            break

    assert message_delta_chunk is not None, "message_delta chunk not found"

    # Verify that the message_delta chunk has both stop_reason and usage
    delta = message_delta_chunk.get("delta", {})
    usage = message_delta_chunk.get("usage", {})

    assert (
        delta.get("stop_reason") == "end_turn"
    ), f"Expected stop_reason 'end_turn', got {delta.get('stop_reason')}"
    assert (
        usage.get("input_tokens") == 230
    ), f"Expected input_tokens 230, got {usage.get('input_tokens')}"
    assert (
        usage.get("output_tokens") == 65
    ), f"Expected output_tokens 65, got {usage.get('output_tokens')}"


def test_usage_merging_behavior():
    """Test that usage information is properly merged with stop_reason chunk."""

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamWithContentAfterStopReason(),
        model="claude-3",
    )

    # Process chunks and look specifically for the usage merging behavior
    chunks = []
    for chunk in wrapper:
        chunks.append(chunk)
        # If this is a message_delta with stop_reason, verify it has usage
        if (
            chunk.get("type") == "message_delta"
            and chunk.get("delta", {}).get("stop_reason") is not None
        ):

            usage = chunk.get("usage", {})
            assert (
                usage.get("input_tokens") is not None
            ), "Usage should be merged with stop_reason chunk"
            assert (
                usage.get("output_tokens") is not None
            ), "Usage should be merged with stop_reason chunk"
            break


def test_sse_wrapper_with_content_after_stop_reason():
    """Test SSE wrapper formatting for the content after stop_reason scenario."""

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamWithContentAfterStopReason(),
        model="claude-3",
    )

    # Get SSE formatted chunks
    sse_chunks = []
    for chunk in wrapper.anthropic_sse_wrapper():
        sse_chunks.append(chunk)
        if len(sse_chunks) >= 10:  # Limit to avoid infinite loops in tests
            break

    # Verify all chunks are properly formatted as bytes
    for chunk in sse_chunks:
        assert isinstance(chunk, bytes), "SSE chunks should be bytes"

        # Decode and verify SSE format
        chunk_str = chunk.decode("utf-8")
        lines = chunk_str.split("\n")

        # Should have event and data lines
        assert any(
            line.startswith("event: ") for line in lines
        ), f"Missing event line in: {chunk_str}"
        assert any(
            line.startswith("data: ") for line in lines
        ), f"Missing data line in: {chunk_str}"


@pytest.mark.asyncio
async def test_async_sse_wrapper_with_content_after_stop_reason():
    """Test async SSE wrapper formatting for the content after stop_reason scenario."""

    wrapper = AnthropicStreamWrapper(
        completion_stream=MockCompletionStreamWithContentAfterStopReason(),
        model="claude-3",
    )

    # Get SSE formatted chunks asynchronously
    sse_chunks = []
    async for chunk in wrapper.async_anthropic_sse_wrapper():
        sse_chunks.append(chunk)
        if len(sse_chunks) >= 10:  # Limit to avoid infinite loops in tests
            break

    # Verify all chunks are properly formatted as bytes
    for chunk in sse_chunks:
        assert isinstance(chunk, bytes), "Async SSE chunks should be bytes"

        # Decode and verify SSE format
        chunk_str = chunk.decode("utf-8")
        lines = chunk_str.split("\n")

        # Should have event and data lines
        assert any(
            line.startswith("event: ") for line in lines
        ), f"Missing event line in: {chunk_str}"
        assert any(
            line.startswith("data: ") for line in lines
        ), f"Missing data line in: {chunk_str}"


if __name__ == "__main__":
    # Run a quick test
    test_anthropic_stream_wrapper_content_after_stop_reason()
    print("✅ Sync test passed")

    import asyncio

    asyncio.run(test_async_anthropic_stream_wrapper_content_after_stop_reason())
    print("✅ Async test passed")

    test_usage_merging_behavior()
    print("✅ Usage merging test passed")

    test_sse_wrapper_with_content_after_stop_reason()
    print("✅ SSE wrapper test passed")

    asyncio.run(test_async_sse_wrapper_with_content_after_stop_reason())
    print("✅ Async SSE wrapper test passed")

    print("🎉 All tests passed!")
