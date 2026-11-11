"""
Tests for BaseModelResponseIterator - specifically testing that empty SSE lines are filtered
and non-string objects (e.g. Pydantic BaseModel events from the Responses API) pass through.
"""

import json
from collections.abc import AsyncIterator, Mapping
from itertools import chain
from typing import Final

import httpx
import pytest
from pydantic import BaseModel

from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.llms.base_llm.base_model_iterator import (
    BaseModelResponseIterator,
    MockResponseIterator,
    convert_model_response_to_streaming,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIChatCompletionStreamingHandler
from litellm.llms.openai.common_utils import OpenAIError
from litellm.types.utils import (
    ChoiceLogprobs,
    Choices,
    GenericStreamingChunk,
    Message,
    ModelResponse,
    ModelResponseStream,
)


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


_PREFIXED_BYTE_FRAMES: Final = (
    b"event: completion\ndata: "
    + json.dumps(
        {
            "id": "chatcmpl-bytes",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "gpt-5.5",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": "Hello"}, "finish_reason": None}],
        }
    ).encode(),
    b"data: [DONE]",
)


async def _aiter_frames(frames: tuple[bytes, ...]) -> AsyncIterator[bytes]:
    for frame in frames:
        yield frame


async def _collect_prefixed_byte_frames(sync_mode: bool) -> list[object]:
    if sync_mode:
        return list(
            OpenAIChatCompletionStreamingHandler(streaming_response=iter(_PREFIXED_BYTE_FRAMES), sync_stream=True)
        )
    return [
        chunk
        async for chunk in OpenAIChatCompletionStreamingHandler(
            streaming_response=_aiter_frames(_PREFIXED_BYTE_FRAMES), sync_stream=False
        )
    ]


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_bytes_chunk_with_text_before_data_prefix_is_parsed(sync_mode: bool) -> None:
    chunks: Final = await _collect_prefixed_byte_frames(sync_mode)

    assert len(chunks) == 2
    assert chunks[0].id == "chatcmpl-bytes"
    assert chunks[0].choices[0].delta.content == "Hello"
    assert chunks[1]["is_finished"] is True


def test_close_releases_the_attached_http_response():
    response: Final = httpx.Response(200, stream=httpx.ByteStream(b"data: [DONE]\n\n"))
    iterator: Final = BaseModelResponseIterator(streaming_response=iter(()), sync_stream=True)
    iterator.http_response = response
    iterator.close()
    assert response.is_closed


def test_fake_stream_chunk_indexes_thinking_blocks_by_position_without_touching_the_response():
    """The OpenAI SDK's `.stream()` accumulator merges streamed `thinking_blocks` entries by `index`, so a fake stream
    numbers them the way a real stream does, redacted blocks included"""
    thinking_blocks: Final = [
        {"type": "thinking", "thinking": "Step 1.", "signature": "sig-a"},
        {"type": "redacted_thinking", "data": "opaque"},
        {"type": "thinking", "thinking": "Step 2.", "signature": "sig-b"},
    ]
    response: Final = ModelResponse(
        choices=[Choices(message=Message(content="Paris", thinking_blocks=thinking_blocks), finish_reason="stop")]
    )

    chunk: Final = convert_model_response_to_streaming(response)

    assert chunk.choices[0].delta.thinking_blocks == [
        {"type": "thinking", "thinking": "Step 1.", "signature": "sig-a", "index": 0},
        {"type": "redacted_thinking", "data": "opaque", "index": 1},
        {"type": "thinking", "thinking": "Step 2.", "signature": "sig-b", "index": 2},
    ]
    assert [block.get("index") for block in response.choices[0].message.thinking_blocks] == [None, None, None]


def _sse_chunk(delta: Mapping[str, object], finish_reason: str | None = None) -> str:
    return "data: " + json.dumps(
        {
            "id": "chatcmpl-done",
            "object": "chat.completion.chunk",
            "created": 1700000000,
            "model": "mistral-small-latest",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        }
    )


async def _aiter_lines(lines: tuple[str, ...]) -> AsyncIterator[str]:
    for line in lines:
        yield line


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.asyncio
async def test_done_inside_streamed_json_is_content_not_the_end_of_stream(sync_mode: bool) -> None:
    lines: Final = (
        _sse_chunk({"role": "assistant", "reasoning_content": "Mark it [DONE] once tests pass"}),
        _sse_chunk({"content": "Checklist: "}),
        _sse_chunk({"content": "- [DONE] write tests"}),
        _sse_chunk({"content": " - [ ] ship it"}),
        _sse_chunk(
            {
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "set_status", "arguments": ""},
                    }
                ]
            }
        ),
        _sse_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '{"status": '}}]}),
        _sse_chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"[DONE]"'}}]}),
        _sse_chunk({"tool_calls": [{"index": 0, "function": {"arguments": ', "n": 1}'}}]}),
        _sse_chunk({}, "tool_calls"),
        "data: [DONE]",
    )
    wrapper: Final = CustomStreamWrapper(
        completion_stream=OpenAIChatCompletionStreamingHandler(
            streaming_response=iter(lines) if sync_mode else _aiter_lines(lines), sync_stream=sync_mode
        ),
        model="mistral/mistral-small-latest",
        custom_llm_provider="mistral",
        logging_obj=Logging(
            model="mistral/mistral-small-latest",
            messages=[{"role": "user", "content": "hi"}],
            stream=True,
            call_type="completion",
            start_time=1_790_000_000.0,
            litellm_call_id="call-id",
            function_id="function-id",
        ),
    )

    chunks: Final = list(wrapper) if sync_mode else [chunk async for chunk in wrapper]

    deltas: Final = [chunk.choices[0].delta for chunk in chunks if chunk.choices]
    tool_calls: Final = tuple(chain.from_iterable(delta.tool_calls or () for delta in deltas))
    assert "".join(getattr(delta, "reasoning_content", None) or "" for delta in deltas) == (
        "Mark it [DONE] once tests pass"
    )
    assert "".join(delta.content or "" for delta in deltas) == "Checklist: - [DONE] write tests - [ ] ship it"
    assert json.loads("".join(tool_call.function.arguments or "" for tool_call in tool_calls)) == {
        "status": "[DONE]",
        "n": 1,
    }
    assert [chunk.choices[0].finish_reason for chunk in chunks if chunk.choices and chunk.choices[0].finish_reason] == [
        "tool_calls"
    ]


@pytest.mark.parametrize("line", ["data: [DONE]", "data:[DONE]", "[DONE]", "data:  [DONE]", "data: [DONE]  "])
def test_done_marker_ends_the_stream(line: str) -> None:
    chunks: Final = list(BaseModelResponseIterator(streaming_response=iter((line,)), sync_stream=True))

    assert chunks == [
        GenericStreamingChunk(text="", is_finished=True, finish_reason="stop", usage=None, index=0, tool_use=None)
    ]


def test_error_frame_mentioning_done_raises_instead_of_ending_the_stream() -> None:
    line: Final = 'data: {"error": {"message": "prompt too long [DONE]", "code": 400}}'

    with pytest.raises(OpenAIError, match=r"prompt too long \[DONE\]") as error:
        list(OpenAIChatCompletionStreamingHandler(streaming_response=iter((line,)), sync_stream=True))

    assert error.value.status_code == 400


@pytest.mark.parametrize("sync_mode", [True, False])
@pytest.mark.parametrize("preserve_logprobs", [True, False])
@pytest.mark.asyncio
async def test_fake_stream_preserves_each_choices_logprobs(sync_mode: bool, preserve_logprobs: bool) -> None:
    response: Final = ModelResponse(
        id="chatcmpl-logprobs",
        created=1_790_000_000,
        model="test-model",
        choices=[
            Choices(
                index=2,
                message=Message(content="Yes"),
                logprobs={
                    "content": [
                        {
                            "token": "Yes",
                            "logprob": -0.25,
                            "bytes": [89, 101, 115],
                            "top_logprobs": [
                                {"token": "Yes", "logprob": -0.25, "bytes": [89, 101, 115]},
                                {"token": "No", "logprob": -1.75, "bytes": [78, 111]},
                            ],
                        }
                    ],
                    "refusal": [],
                },
            ),
            Choices(
                index=7,
                message=Message(content=None, refusal="Cannot help"),
                logprobs={
                    "content": [],
                    "refusal": [
                        {"token": "Cannot", "logprob": -0.5, "bytes": [67, 97, 110, 110, 111, 116], "top_logprobs": []}
                    ],
                },
            ),
            Choices(index=9, message=Message(content="No probabilities requested")),
        ],
    )
    original: Final = response.model_dump_json()
    iterator: Final = (
        MockResponseIterator(response, preserve_logprobs=True) if preserve_logprobs else MockResponseIterator(response)
    )
    chunks: Final = list(iterator) if sync_mode else [chunk async for chunk in iterator]

    assert len(chunks) == 1
    chunk: Final = chunks[0]
    assert isinstance(chunk, ModelResponseStream)
    assert [choice.index for choice in chunk.choices] == [2, 7, 9]
    for logprobs, source_logprobs in zip(
        (choice.logprobs for choice in chunk.choices[:2]),
        (choice.logprobs for choice in response.choices[:2]),
        strict=True,
    ):
        if preserve_logprobs:
            assert isinstance(logprobs, ChoiceLogprobs)
            assert isinstance(source_logprobs, ChoiceLogprobs)
            assert logprobs.model_dump() == source_logprobs.model_dump()
        else:
            assert logprobs is None
    assert chunk.choices[2].logprobs is None
    assert chunk.choices[0].delta.content == "Yes"
    assert getattr(chunk.choices[1].delta, "refusal", None) == "Cannot help"
    assert response.model_dump_json() == original


def test_response_replay_drops_original_token_probabilities_by_default() -> None:
    response: Final = ModelResponse(
        choices=[
            Choices(
                message=Message(content="[REDACTED]"),
                logprobs={
                    "content": [{"token": "original-secret", "logprob": -0.25, "bytes": None, "top_logprobs": []}],
                    "refusal": [],
                },
            )
        ]
    )

    for chunk in (convert_model_response_to_streaming(response), next(MockResponseIterator(response))):
        assert chunk.choices[0].delta.content == "[REDACTED]"
        assert chunk.choices[0].logprobs is None
        assert "original-secret" not in chunk.model_dump_json()
