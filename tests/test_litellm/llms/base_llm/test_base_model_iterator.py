"""
Tests for BaseModelResponseIterator - specifically testing that empty SSE lines are filtered
and non-string objects (e.g. Pydantic BaseModel events from the Responses API) pass through.
"""

import pytest
from pydantic import BaseModel

from litellm.llms.base_llm.base_model_iterator import BaseModelResponseIterator
from litellm.types.utils import GenericStreamingChunk


class TestBaseModelResponseIterator:
    """Test cases for BaseModelResponseIterator empty line filtering"""

    def test_filter_empty_sse_lines_sync(self):
        """
        Test that empty SSE lines (common between SSE events) are filtered out
        and don't produce empty chunks.

        This fixes the bug where providers using BaseLLMHTTPHandler (like xAI)
        would return extra empty chunks when streaming with include_usage=True.

        Related: GitHub Issue #17136
        """
        # Simulate SSE stream with empty lines between events (normal SSE format)
        sse_lines = [
            'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}',
            "",  # Empty line (SSE separator)
            'data: {"id":"1","choices":[{"delta":{"content":" World"}}]}',
            "",  # Empty line (SSE separator)
            'data: {"id":"1","choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
            "",  # Empty line (SSE separator)
            "data: [DONE]",
            "",  # Empty line after DONE
        ]

        iterator = BaseModelResponseIterator(
            streaming_response=iter(sse_lines), sync_stream=True
        )

        chunks = list(iterator)

        # Should have 4 chunks: 2 content + 1 usage + 1 DONE
        # Empty lines should be filtered out
        assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"

        # Verify no empty/None chunks were included
        # The base iterator returns ModelResponseStream objects
        for i, chunk in enumerate(chunks):
            assert chunk is not None, f"Chunk {i} should not be None"

    def test_filter_whitespace_only_lines_sync(self):
        """Test that lines with only whitespace are also filtered"""
        sse_lines = [
            'data: {"id":"1","choices":[{"delta":{"content":"Hi"}}]}',
            "   ",  # Whitespace only
            "\t",  # Tab only
            "data: [DONE]",
        ]

        iterator = BaseModelResponseIterator(
            streaming_response=iter(sse_lines), sync_stream=True
        )

        chunks = list(iterator)

        # Should have 2 chunks: 1 content + 1 DONE
        assert len(chunks) == 2, f"Expected 2 chunks, got {len(chunks)}"

    def test_valid_chunks_not_filtered_sync(self):
        """Test that valid data chunks are not filtered"""
        sse_lines = [
            'data: {"id":"1","choices":[{"delta":{"content":"A"}}]}',
            'data: {"id":"1","choices":[{"delta":{"content":"B"}}]}',
            'data: {"id":"1","choices":[{"delta":{"content":"C"}}]}',
            "data: [DONE]",
        ]

        iterator = BaseModelResponseIterator(
            streaming_response=iter(sse_lines), sync_stream=True
        )

        chunks = list(iterator)

        # All 4 chunks should be present
        assert len(chunks) == 4, f"Expected 4 chunks, got {len(chunks)}"


@pytest.mark.asyncio
async def test_filter_empty_sse_lines_async():
    """
    Test async version: empty SSE lines should be filtered out
    """

    async def async_sse_generator():
        lines = [
            'data: {"id":"1","choices":[{"delta":{"content":"Hello"}}]}',
            "",  # Empty line
            'data: {"id":"1","choices":[{"delta":{"content":" World"}}]}',
            "",  # Empty line
            "data: [DONE]",
            "",  # Empty line
        ]
        for line in lines:
            yield line

    iterator = BaseModelResponseIterator(
        streaming_response=async_sse_generator(), sync_stream=False
    )

    chunks = []
    async for chunk in iterator:
        chunks.append(chunk)

    # Should have 3 chunks: 2 content + 1 DONE
    assert len(chunks) == 3, f"Expected 3 chunks, got {len(chunks)}"


class FakeResponseEvent(BaseModel):
    """Simulates a Pydantic BaseModel event like ResponseCreatedEvent from the OpenAI SDK."""

    type: str = "response.created"
    data: dict = {}


class TestBaseModelResponseIteratorNonStringChunks:
    """
    Test that non-string objects (e.g. Pydantic BaseModel events from the
    Responses API) are not dropped by the empty-line filter.

    Without the isinstance(str_line, str) guard, calling .strip() on a
    BaseModel raises AttributeError: 'FakeResponseEvent' object has no
    attribute 'strip'.
    """

    def test_pydantic_basemodel_chunk_passes_through_sync(self):
        """Non-string chunks must not be dropped or cause AttributeError."""
        event = FakeResponseEvent(type="response.created", data={"id": "resp_1"})

        class TestIterator(BaseModelResponseIterator):
            def _handle_string_chunk(self, str_line):
                # Just return the object wrapped in a GenericStreamingChunk
                return GenericStreamingChunk(
                    text=str(str_line),
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=0,
                    tool_use=None,
                )

        iterator = TestIterator(
            streaming_response=iter([event]),
            sync_stream=True,
        )

        chunks = list(iterator)
        assert len(chunks) == 1
        assert "response.created" in chunks[0]["text"]

    def test_mixed_string_and_pydantic_chunks_sync(self):
        """Mix of empty strings, valid SSE, and Pydantic objects."""
        event = FakeResponseEvent(type="response.done", data={})

        class TestIterator(BaseModelResponseIterator):
            def _handle_string_chunk(self, str_line):
                return GenericStreamingChunk(
                    text=str(str_line),
                    is_finished=False,
                    finish_reason="",
                    usage=None,
                    index=0,
                    tool_use=None,
                )

        items = [
            "",  # empty string — should be skipped
            event,  # Pydantic object — must pass through
            "   ",  # whitespace — should be skipped
            "data: [DONE]",  # valid SSE
        ]

        iterator = TestIterator(
            streaming_response=iter(items),
            sync_stream=True,
        )

        chunks = list(iterator)
        # 2 chunks: the Pydantic event + [DONE]
        assert len(chunks) == 2


@pytest.mark.asyncio
async def test_pydantic_basemodel_chunk_passes_through_async():
    """Async variant: non-string chunks must not be dropped."""
    event = FakeResponseEvent(type="response.created", data={"id": "resp_1"})

    class TestIterator(BaseModelResponseIterator):
        def _handle_string_chunk(self, str_line):
            return GenericStreamingChunk(
                text=str(str_line),
                is_finished=False,
                finish_reason="",
                usage=None,
                index=0,
                tool_use=None,
            )

    async def async_gen():
        yield event

    iterator = TestIterator(
        streaming_response=async_gen(),
        sync_stream=False,
    )

    chunks = []
    async for chunk in iterator:
        chunks.append(chunk)

    assert len(chunks) == 1
    assert "response.created" in chunks[0]["text"]


@pytest.mark.asyncio
async def test_aclose_closes_attached_http_response():
    """Regression for BerriAI/litellm#30244: CustomStreamWrapper.aclose() can
    only release the upstream provider connection if the iterator exposes
    aclose() and it reaches the underlying HTTP response. Without this, a
    client disconnect leaves backends like vLLM generating into a dead pipe."""
    from unittest.mock import AsyncMock, MagicMock

    async def async_gen():
        yield "data: {}"

    iterator = BaseModelResponseIterator(
        streaming_response=async_gen(), sync_stream=False
    )
    http_response = MagicMock()
    http_response.aclose = AsyncMock()
    iterator.http_response = http_response

    await iterator.aclose()

    http_response.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_aclose_is_noop_without_http_response():
    async def async_gen():
        yield "data: {}"

    iterator = BaseModelResponseIterator(
        streaming_response=async_gen(), sync_stream=False
    )

    await iterator.aclose()


class ContentEchoIterator(BaseModelResponseIterator):
    def chunk_parser(self, chunk: dict) -> GenericStreamingChunk:
        choices = chunk.get("choices") or []
        content = choices[0].get("delta", {}).get("content", "") if choices else ""
        return GenericStreamingChunk(
            text=content or "",
            is_finished=False,
            finish_reason="",
            usage=None,
            index=0,
            tool_use=None,
        )


class TestDoneMarkerExactMatch:
    def test_content_containing_done_substring_is_not_treated_as_stream_end(self):
        lines = [
            'data: {"id":"1","choices":[{"delta":{"content":"foo [DONE] bar"}}]}',
            "data: [DONE]",
        ]
        iterator = ContentEchoIterator(streaming_response=iter(lines), sync_stream=True)

        chunks = list(iterator)

        assert len(chunks) == 2
        assert chunks[0]["text"] == "foo [DONE] bar"
        assert chunks[0]["is_finished"] is False
        assert chunks[1]["is_finished"] is True
        assert chunks[1]["finish_reason"] == "stop"

    def test_done_line_variants_still_terminate(self):
        for line in ["data: [DONE]", "data:[DONE]", "[DONE]", "data:  [DONE]"]:
            iterator = ContentEchoIterator(
                streaming_response=iter([line]), sync_stream=True
            )
            chunks = list(iterator)
            assert len(chunks) == 1
            assert chunks[0]["is_finished"] is True


class TestUnicodeLineSeparatorSplitRecovery:
    def test_line_split_at_unicode_separator_is_rejoined(self):
        lines = [
            'data: {"id":"1","choices":[{"delta":{"content":"foo',
            'bar"}}]}',
            "data: [DONE]",
        ]
        iterator = ContentEchoIterator(streaming_response=iter(lines), sync_stream=True)

        chunks = list(iterator)

        streamed_text = "".join(chunk["text"] for chunk in chunks)
        assert streamed_text == "foobar"
        assert all(chunk["is_finished"] is False for chunk in chunks[:-1])
        assert chunks[-1]["is_finished"] is True

    def test_line_split_into_three_fragments_is_rejoined(self):
        lines = [
            'data: {"id":"1","choices":[{"delta":{"content":"a',
            "b",
            'c"}}]}',
            "data: [DONE]",
        ]
        iterator = ContentEchoIterator(streaming_response=iter(lines), sync_stream=True)

        chunks = list(iterator)

        assert "".join(chunk["text"] for chunk in chunks) == "abc"

    def test_pending_fragment_does_not_corrupt_next_valid_line(self):
        lines = [
            'data: {"broken',
            'data: {"id":"1","choices":[{"delta":{"content":"ok"}}]}',
            "data: [DONE]",
        ]
        iterator = ContentEchoIterator(streaming_response=iter(lines), sync_stream=True)

        chunks = list(iterator)

        assert "".join(chunk["text"] for chunk in chunks) == "ok"
        assert chunks[-1]["is_finished"] is True

    @pytest.mark.asyncio
    async def test_line_split_at_unicode_separator_is_rejoined_async(self):
        async def async_gen():
            for line in [
                'data: {"id":"1","choices":[{"delta":{"content":"foo',
                'bar"}}]}',
                "data: [DONE]",
            ]:
                yield line

        iterator = ContentEchoIterator(
            streaming_response=async_gen(), sync_stream=False
        )

        chunks = [chunk async for chunk in iterator]

        assert "".join(chunk["text"] for chunk in chunks) == "foobar"
