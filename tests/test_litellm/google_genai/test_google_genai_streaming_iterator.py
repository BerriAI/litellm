import json
from unittest.mock import MagicMock

import pytest

from litellm.google_genai.streaming_iterator import (
    AsyncGoogleGenAIGenerateContentStreamingIterator,
    GoogleGenAIGenerateContentStreamingIterator,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


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
        json.loads(chunk[len(b"data: ") : -2])["candidates"][0]["content"]["parts"][0][
            "inlineData"
        ]["mimeType"]
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
    assert json.loads(chunk[len(b"data: ") : -2])["candidates"][0]["content"]["parts"][
        0
    ]["inlineData"]["data"].startswith("A")


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
