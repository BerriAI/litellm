import json
from unittest.mock import AsyncMock

import pytest

from litellm.google_genai.adapters.transformation import GoogleGenAIStreamWrapper
from litellm.google_genai.streaming_iterator import (
    AsyncGoogleGenAIGenerateContentStreamingIterator,
    GoogleGenAIGenerateContentStreamingIterator,
)


def test_google_genai_sse_wrapper_skips_done_sentinel():
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "OK"}],
                }
            }
        ]
    }

    wrapper = GoogleGenAIStreamWrapper(
        iter(
            [
                payload,
                b"data: [DONE]\n\n",
            ]
        )
    )

    chunks = list(wrapper.google_genai_sse_wrapper())

    assert chunks == [f"data: {json.dumps(payload)}\n\n".encode()]


@pytest.mark.asyncio
async def test_async_google_genai_sse_wrapper_skips_done_sentinel():
    payload = {
        "candidates": [
            {
                "content": {
                    "role": "model",
                    "parts": [{"text": "OK"}],
                }
            }
        ]
    }

    async def completion_stream():
        yield payload
        yield b"data: [DONE]\n\n"

    wrapper = GoogleGenAIStreamWrapper(completion_stream())
    chunks = [chunk async for chunk in wrapper.async_google_genai_sse_wrapper()]

    assert chunks == [f"data: {json.dumps(payload)}\n\n".encode()]


def test_google_genai_provider_stream_iterator_skips_done_sentinel():
    payload = b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"OK"}]}}]}\n\n'

    class DummyResponse:
        def iter_bytes(self):
            yield payload
            yield b"data: [DONE]\n\n"

    iterator = GoogleGenAIGenerateContentStreamingIterator(
        response=DummyResponse(),
        model="gemini-2.5-pro",
        logging_obj=None,
        generate_content_provider_config=None,
        litellm_metadata={},
        custom_llm_provider="vertex_ai",
        request_body={},
    )

    assert list(iterator) == [payload]


@pytest.mark.asyncio
async def test_async_google_genai_provider_stream_iterator_skips_done_sentinel():
    payload = b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"OK"}]}}]}\n\n'

    class DummyResponse:
        async def aiter_bytes(self):
            yield payload
            yield b"data: [DONE]\n\n"

    iterator = AsyncGoogleGenAIGenerateContentStreamingIterator(
        response=DummyResponse(),
        model="gemini-2.5-pro",
        logging_obj=None,
        generate_content_provider_config=None,
        litellm_metadata={},
        custom_llm_provider="vertex_ai",
        request_body={},
    )
    iterator._handle_async_streaming_logging = AsyncMock()

    chunks = [chunk async for chunk in iterator]

    assert chunks == [payload]
    iterator._handle_async_streaming_logging.assert_awaited_once()
