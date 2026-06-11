"""
Regression tests for https://github.com/BerriAI/litellm/issues/28777

The streamGenerateContent endpoint must return raw JSON bytes from Gemini,
not SSE-wrapped 'data: {...}' format. The google-genai SDK parses raw JSON
and raises UnknownApiResponseError on the SSE 'data: ' prefix.
"""

import os
import sys
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

GEMINI_STREAM_CHUNKS = [
    b'[{"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]},"finishReason":"STOP"}]}]\n',
    b'[{"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]},"finishReason":"STOP"}]}]\n',
]


class FakeGeminiStreamingIterator:
    """Mimics AsyncGoogleGenAIGenerateContentStreamingIterator: yields raw bytes,
    has _hidden_params dict."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self._index = 0
        self._hidden_params = {"additional_headers": {}}

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._index >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._index]
        self._index += 1
        return chunk


def test_streaming_response_detected_for_async_iterator():
    """_is_streaming_response must recognize the Gemini streaming iterator."""
    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    processor = ProxyBaseLLMRequestProcessing(data={"model": "test", "stream": True})
    fake_iter = FakeGeminiStreamingIterator(GEMINI_STREAM_CHUNKS)
    assert processor._is_streaming_response(fake_iter) is True


@pytest.mark.asyncio
async def test_agenerate_content_stream_returns_raw_streaming_response():
    """base_process_llm_request with route_type='agenerate_content_stream' must
    return a StreamingResponse that passes raw bytes through — no SSE wrapping."""
    from starlette.responses import StreamingResponse

    from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing

    fake_response = FakeGeminiStreamingIterator(GEMINI_STREAM_CHUNKS)

    processor = ProxyBaseLLMRequestProcessing(
        data={"model": "gemini-2.0-flash", "stream": True}
    )

    fake_logging_obj = MagicMock()
    fake_logging_obj.litellm_call_id = "test-call-id"
    fake_logging_obj._on_deferred_stream_complete = None
    fake_logging_obj._defer_async_logging = False

    fake_proxy_logging = AsyncMock()
    fake_proxy_logging.during_call_hook = AsyncMock()
    fake_proxy_logging.post_call_response_headers_hook = AsyncMock(return_value={})
    fake_proxy_logging.update_request_status = AsyncMock()

    fake_fastapi_response = MagicMock()
    fake_fastapi_response.headers = {}

    fake_user_api_key_dict = MagicMock()
    fake_user_api_key_dict.token = "sk-test"
    fake_user_api_key_dict.key_alias = None
    fake_user_api_key_dict.allowed_model_region = ""

    async def fake_route_request(**kwargs):
        """Simulates route_request returning a coroutine whose result is the
        streaming iterator (matching the real call chain: await route_request()
        returns a coroutine, then asyncio.gather awaits it)."""

        async def _inner():
            return fake_response

        return _inner()

    with (
        patch.object(
            processor,
            "common_processing_pre_call_logic",
            new_callable=AsyncMock,
            return_value=(processor.data, fake_logging_obj),
        ),
        patch.object(
            processor,
            "_has_post_call_guardrails",
            return_value=False,
        ),
        patch(
            "litellm.proxy.common_request_processing.route_request",
            new=fake_route_request,
        ),
    ):
        result = await processor.base_process_llm_request(
            request=MagicMock(),
            fastapi_response=fake_fastapi_response,
            user_api_key_dict=fake_user_api_key_dict,
            route_type="agenerate_content_stream",
            proxy_logging_obj=fake_proxy_logging,
            llm_router=None,
            general_settings={},
            proxy_config=MagicMock(),
            select_data_generator=None,
            model="gemini-2.0-flash",
        )

    assert isinstance(
        result, StreamingResponse
    ), f"Expected StreamingResponse, got {type(result).__name__}"

    collected = b""
    async for chunk in result.body_iterator:
        if isinstance(chunk, str):
            collected += chunk.encode()
        else:
            collected += chunk

    assert (
        b"data: " not in collected
    ), f"Raw Gemini bytes must not be SSE-wrapped. Got: {collected[:120]!r}"
    assert b'"candidates"' in collected
    for expected_chunk in GEMINI_STREAM_CHUNKS:
        assert expected_chunk in collected
