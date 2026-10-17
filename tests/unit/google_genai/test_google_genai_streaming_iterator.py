import json
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest

from litellm.google_genai.streaming_iterator import (
    AsyncGoogleGenAIGenerateContentStreamingIterator,
    GoogleGenAIGenerateContentStreamingIterator,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType


@pytest.mark.parametrize(
    "custom_llm_provider, expected_endpoint_type",
    [("gemini", EndpointType.GEMINI), ("vertex_ai", EndpointType.VERTEX_AI)],
)
@pytest.mark.parametrize(
    "iterator_cls",
    [
        AsyncGoogleGenAIGenerateContentStreamingIterator,
        GoogleGenAIGenerateContentStreamingIterator,
    ],
)
def test_streaming_logging_targets_the_provider_that_served_the_request(
    iterator_cls: type,
    custom_llm_provider: str,
    expected_endpoint_type: EndpointType,
):
    """Routing every google stream through the vertex handler bills gemini/* at vertex_ai/ rates."""
    iterator = iterator_cls(
        response=MagicMock(),
        model="gemini-3.1-flash-image",
        logging_obj=MagicMock(spec=LiteLLMLoggingObj),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider=custom_llm_provider,
    )

    assert iterator.endpoint_type is expected_endpoint_type


def _large_inline_data_event() -> str:
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {
                            "inlineData": {
                                "mimeType": "image/jpeg",
                                "data": "A" * 20000,
                            }
                        }
                    ]
                }
            }
        ]
    }
    return f"data: {json.dumps(payload)}"


@pytest.mark.asyncio
async def test_async_streaming_iterator_yields_complete_sse_events():
    """Large inlineData must not be split across byte-chunk boundaries."""
    mock_response = MagicMock()

    async def _aiter_lines():
        yield _large_inline_data_event()

    mock_response.aiter_lines = _aiter_lines

    iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=mock_response,
        model="gemini-3.1-flash-image-preview",
        logging_obj=MagicMock(spec=LiteLLMLoggingObj),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = await iterator.__anext__()
    assert chunk.startswith(b"data: ")
    assert chunk.endswith(b"\n\n")
    assert (
        json.loads(chunk[len(b"data: ") : -2])["candidates"][0]["content"]["parts"][0]["inlineData"]["mimeType"]
        == "image/jpeg"
    )


def test_sync_streaming_iterator_yields_complete_sse_events():
    mock_response = MagicMock()
    mock_response.iter_lines.return_value = iter([_large_inline_data_event()])

    iterator = GoogleGenAIGenerateContentStreamingIterator(
        response=mock_response,
        model="gemini-3.1-flash-image-preview",
        logging_obj=MagicMock(spec=LiteLLMLoggingObj),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = next(iterator)
    assert chunk.startswith(b"data: ")
    assert chunk.endswith(b"\n\n")
    assert json.loads(chunk[len(b"data: ") : -2])["candidates"][0]["content"]["parts"][0]["inlineData"][
        "data"
    ].startswith("A")


@pytest.mark.asyncio
async def test_async_streaming_iterator_preserves_multi_field_sse_event():
    mock_response = MagicMock()

    async def _aiter_lines():
        yield "event: message"
        yield 'data: {"text":"hi"}'
        yield ""

    mock_response.aiter_lines = _aiter_lines

    iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=mock_response,
        model="gemini-test",
        logging_obj=MagicMock(spec=LiteLLMLoggingObj),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = await iterator.__anext__()
    assert chunk == b'event: message\ndata: {"text":"hi"}\n\n'


@pytest.mark.asyncio
async def test_async_streaming_iterator_forwards_sse_comment_events():
    mock_response = MagicMock()

    async def _aiter_lines():
        yield ": keepalive"
        yield ""

    mock_response.aiter_lines = _aiter_lines

    iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=mock_response,
        model="gemini-test",
        logging_obj=MagicMock(spec=LiteLLMLoggingObj),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="gemini",
    )

    chunk = await iterator.__anext__()
    assert chunk == b": keepalive\n\n"


def _async_response_with_lines(lines: list[str], drop_after: Exception | None = None):
    """An httpx.Response stand-in whose line iterator yields `lines`, then raises `drop_after`."""
    mock_response = MagicMock()

    async def _aiter_lines():
        for line in lines:
            yield line
        if drop_after is not None:
            raise drop_after

    mock_response.aiter_lines = _aiter_lines
    return mock_response


def _async_iterator(mock_response) -> AsyncGoogleGenAIGenerateContentStreamingIterator:
    return AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=mock_response,
        model="gemini-3.8-flash",
        logging_obj=MagicMock(spec=LiteLLMLoggingObj),
        generate_content_provider_config=MagicMock(),
        litellm_metadata={},
        custom_llm_provider="vertex_ai",
    )


@pytest.mark.asyncio
async def test_priming_surfaces_a_stream_that_drops_before_its_first_event():
    """A drop on the first read fails the call that opened the stream, not the consumer that reads it later."""
    mock_response = _async_response_with_lines([], drop_after=httpx.ReadError("Connection closed."))

    iterator = _async_iterator(mock_response)

    with pytest.raises(httpx.ReadError):
        await iterator.prime_first_chunk()


@pytest.mark.asyncio
async def test_priming_replays_the_first_event_without_losing_or_duplicating_it():
    mock_response = _async_response_with_lines(['data: {"first": 1}', "", 'data: {"second": 2}', ""])

    iterator = _async_iterator(mock_response)
    await iterator.prime_first_chunk()

    assert await iterator.__anext__() == b'data: {"first": 1}\n\n'
    assert await iterator.__anext__() == b'data: {"second": 2}\n\n'
    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()


@pytest.mark.asyncio
async def test_priming_an_empty_stream_ends_the_iterator_and_still_logs():
    """A stream that ends before its first event is a valid (empty) answer, not a failure."""
    mock_response = _async_response_with_lines([])

    iterator = _async_iterator(mock_response)
    iterator._handle_async_streaming_logging = AsyncMock()

    await iterator.prime_first_chunk()

    with pytest.raises(StopAsyncIteration):
        await iterator.__anext__()

    iterator._handle_async_streaming_logging.assert_awaited_once()


@pytest.mark.asyncio
async def test_priming_twice_does_not_consume_a_second_event():
    mock_response = _async_response_with_lines(['data: {"first": 1}', "", 'data: {"second": 2}', ""])

    iterator = _async_iterator(mock_response)
    await iterator.prime_first_chunk()
    await iterator.prime_first_chunk()

    assert await iterator.__anext__() == b'data: {"first": 1}\n\n'
    assert await iterator.__anext__() == b'data: {"second": 2}\n\n'
