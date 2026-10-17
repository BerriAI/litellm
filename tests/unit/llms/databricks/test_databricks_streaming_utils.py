"""
Regression test for the databricks streaming chunk parser.

OpenAI-compatible servers (e.g. Vertex AI Model Garden vLLM endpoints) send a final
usage-only chunk with an empty `choices` list when `stream_options.include_usage` is
set. `chunk_parser` previously did `choices[0]` unconditionally, raising
`IndexError` -> `MidStreamFallbackError` and crashing the stream.
"""

import json
from typing import Final

import httpx
import pytest
import respx

import litellm
from litellm.llms.databricks.streaming_utils import ModelResponseIterator


def test_chunk_parser_handles_empty_choices_usage_chunk():
    """A usage-only final chunk (empty choices) must not raise IndexError."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    usage_only_chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28},
    }

    result = iterator.chunk_parser(chunk=usage_only_chunk)

    assert result["text"] == ""
    assert result["is_finished"] is False
    assert result["usage"] is not None
    assert result["usage"]["prompt_tokens"] == 20
    assert result["usage"]["completion_tokens"] == 8


def test_chunk_parser_empty_choices_without_usage():
    """An empty-choices chunk with no usage block returns usage=None, no error."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert result["text"] == ""
    assert result["usage"] is None


def test_chunk_parser_normal_content_chunk_still_works():
    """A regular content chunk is unaffected by the empty-choices guard."""
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [{"index": 0, "delta": {"content": "hi"}, "finish_reason": None}],
    }

    result = iterator.chunk_parser(chunk=chunk)

    assert result["text"] == "hi"


def test_chunk_parser_keeps_cached_tokens_on_usage_only_chunk():
    iterator = ModelResponseIterator(streaming_response=None, sync_stream=True)
    usage_only_chunk = {
        "id": "chatcmpl-x",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": [],
        "usage": {
            "prompt_tokens": 340,
            "completion_tokens": 1,
            "total_tokens": 341,
            "prompt_tokens_details": {"cached_tokens": 320},
        },
    }

    result = iterator.chunk_parser(chunk=usage_only_chunk)

    assert result["usage"] is not None
    assert result["usage"]["prompt_tokens_details"] is not None
    assert result["usage"]["prompt_tokens_details"]["cached_tokens"] == 320


def _sse_event(body: dict[str, object]) -> bytes:
    return f"data: {json.dumps(body)}\n\n".encode()


@pytest.mark.respx(assert_all_called=True)
def test_stream_usage_after_finish_chunk_reaches_the_caller(respx_mock: respx.MockRouter) -> None:
    chunk_header: Final = {"id": "chatcmpl-x", "object": "chat.completion.chunk", "created": 1, "model": "sabia-3"}
    stream_body: Final = b"".join(
        (
            _sse_event({**chunk_header, "choices": [{"index": 0, "delta": {"role": "assistant", "content": "144"}}]}),
            _sse_event({**chunk_header, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}),
            _sse_event(
                {
                    **chunk_header,
                    "choices": [],
                    "usage": {
                        "prompt_tokens": 340,
                        "completion_tokens": 1,
                        "total_tokens": 341,
                        "prompt_tokens_details": {"cached_tokens": 320},
                    },
                }
            ),
            b"data: [DONE]\n\n",
        )
    )
    respx_mock.post("http://upstream.test/chat/completions").mock(
        return_value=httpx.Response(200, headers={"content-type": "text/event-stream"}, content=stream_body)
    )

    stream: Final = litellm.completion(
        model="maritalk/sabia-3",
        api_base="http://upstream.test",
        api_key="sk-test",
        messages=[{"role": "user", "content": "What is 12*12?"}],
        stream=True,
        stream_options={"include_usage": True},
    )
    usages: Final = [chunk.usage for chunk in stream if getattr(chunk, "usage", None)]

    assert [(usage.prompt_tokens, usage.prompt_tokens_details.cached_tokens) for usage in usages] == [(340, 320)]


def test_chunk_parser_keeps_usage_on_the_finish_chunk():
    iterator: Final = ModelResponseIterator(streaming_response=None, sync_stream=True)

    result: Final = iterator.chunk_parser(
        chunk={
            "id": "chatcmpl-x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "m",
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 340, "completion_tokens": 1, "total_tokens": 341},
        }
    )

    assert result["is_finished"] is True
    assert result["usage"] is not None
    assert result["usage"]["prompt_tokens"] == 340


_SSE_LINES_WITH_BLANK_SEPARATORS: Final = (
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{"content":"144"},"finish_reason":null}]}',
    "",
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m",'
    '"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}',
    "",
    "data: {not json",
    "",
    'data: {"id":"c","object":"chat.completion.chunk","created":1,"model":"m","choices":[],'
    '"usage":{"prompt_tokens":340,"completion_tokens":1,"total_tokens":341}}',
    "",
    "data: [DONE]",
    "",
)


@pytest.mark.asyncio
async def test_async_stream_skips_blank_sse_lines_and_keeps_the_usage_chunk():
    async def lines():
        for line in _SSE_LINES_WITH_BLANK_SEPARATORS:
            yield line

    chunks: Final = [chunk async for chunk in ModelResponseIterator(streaming_response=lines(), sync_stream=False)]

    assert [(chunk["text"], chunk["finish_reason"], (chunk["usage"] or {}).get("prompt_tokens")) for chunk in chunks] == [
        ("144", "", None),
        ("", "stop", None),
        ("", "", None),
        ("", "", 340),
    ]


class _BrokenTransport(ValueError):
    pass


def test_sync_stream_transport_error_surfaces_as_runtime_error():
    def lines():
        yield _SSE_LINES_WITH_BLANK_SEPARATORS[0]
        raise _BrokenTransport("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        list(ModelResponseIterator(streaming_response=lines(), sync_stream=True))


@pytest.mark.asyncio
async def test_async_stream_transport_error_surfaces_as_runtime_error():
    async def lines():
        yield _SSE_LINES_WITH_BLANK_SEPARATORS[0]
        raise _BrokenTransport("connection reset")

    with pytest.raises(RuntimeError, match="connection reset"):
        [chunk async for chunk in ModelResponseIterator(streaming_response=lines(), sync_stream=False)]


@pytest.mark.asyncio
async def test_async_stream_without_done_marker_ends_after_the_last_chunk():
    async def lines():
        for line in _SSE_LINES_WITH_BLANK_SEPARATORS[:4]:
            yield line

    chunks: Final = [chunk async for chunk in ModelResponseIterator(streaming_response=lines(), sync_stream=False)]

    assert [(chunk["text"], chunk["finish_reason"]) for chunk in chunks] == [("144", ""), ("", "stop")]
