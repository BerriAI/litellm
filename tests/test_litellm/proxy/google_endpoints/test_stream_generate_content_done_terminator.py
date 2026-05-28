"""
LIT-3411 regression tests.

Google native `streamGenerateContent` SSE responses must NOT include the
OpenAI-style `data: [DONE]` terminator. Native Vertex/Gemini SDKs (e.g.
the Vertex Java SDK) parse the SSE body as JSON and throw
`JsonParseException` on the literal token `DONE` when this terminator is
appended.

These tests pin two contracts:

1. The SSE generator (`async_data_generator` in `proxy_server.py`) skips
   the `data: [DONE]\\n\\n` tail when `request_data` carries the
   `_litellm_skip_sse_done_terminator=True` flag.
2. The Google `:streamGenerateContent` endpoint handler sets that flag
   on the request body it hands to `ProxyBaseLLMRequestProcessing`.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest


class _FakeGeminiAsyncStream:
    """Async iterator that yields raw Gemini-format SSE bytes — the shape
    `AsyncGoogleGenAIGenerateContentStreamingIterator` hands downstream
    for native `streamGenerateContent`."""

    def __init__(self):
        self._chunks = [
            b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":"Hello"}]}}]}\n\n',
            b'data: {"candidates":[{"content":{"role":"model","parts":[{"text":" world"}]}}],'
            b'"usageMetadata":{"totalTokenCount":4}}\n\n',
        ]
        self._i = 0

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._i >= len(self._chunks):
            raise StopAsyncIteration
        chunk = self._chunks[self._i]
        self._i += 1
        return chunk

    async def aclose(self):
        return None


async def _drain(gen):
    out = []
    async for chunk in gen:
        if isinstance(chunk, bytes):
            chunk = chunk.decode("utf-8", errors="replace")
        out.append(chunk)
    return out


# ---------------------------------------------------------------------------
# 1. async_data_generator gating — the SSE writer respects the flag
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_data_generator_emits_done_by_default():
    """OpenAI-style streaming still gets the `[DONE]` terminator. Pins
    the existing contract for chat-completions and every caller that
    does NOT set the new flag."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import async_data_generator

    gen = async_data_generator(
        response=_FakeGeminiAsyncStream(),
        user_api_key_dict=UserAPIKeyAuth(token="t"),
        request_data={"model": "gpt-4o", "stream": True},
    )
    chunks = await _drain(gen)
    body = "".join(chunks)
    assert "data: [DONE]\n\n" in body
    assert chunks[-1] == "data: [DONE]\n\n"


@pytest.mark.asyncio
async def test_async_data_generator_skips_done_when_flag_set():
    """LIT-3411: when `_litellm_skip_sse_done_terminator=True` is on
    `request_data`, the SSE generator must NOT append `data: [DONE]`."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import async_data_generator

    gen = async_data_generator(
        response=_FakeGeminiAsyncStream(),
        user_api_key_dict=UserAPIKeyAuth(token="t"),
        request_data={
            "model": "gemini/gemini-2.0-flash",
            "stream": True,
            "_litellm_skip_sse_done_terminator": True,
        },
    )
    chunks = await _drain(gen)
    body = "".join(chunks)
    assert "[DONE]" not in body, (
        "Google-native streamGenerateContent must not emit `data: [DONE]`; "
        "Vertex Java SDK throws JsonParseException on the literal `DONE`."
    )
    # All chunks should be real Gemini SSE frames, in order.
    assert all(c.startswith("data: {") and c.endswith("\n\n") for c in chunks)


@pytest.mark.asyncio
async def test_async_data_generator_skips_done_when_flag_set_falsy_off():
    """An explicit `False` flag still emits `[DONE]` — the gate is a
    truthy check only."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import async_data_generator

    gen = async_data_generator(
        response=_FakeGeminiAsyncStream(),
        user_api_key_dict=UserAPIKeyAuth(token="t"),
        request_data={
            "model": "gpt-4o",
            "stream": True,
            "_litellm_skip_sse_done_terminator": False,
        },
    )
    body = "".join(await _drain(gen))
    assert body.endswith("data: [DONE]\n\n")


# ---------------------------------------------------------------------------
# 2. The Google streamGenerateContent endpoint sets the flag — behavioral
# ---------------------------------------------------------------------------


def _build_test_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy.google_endpoints.endpoints import router as google_router

    app = FastAPI()
    app.include_router(google_router)
    return TestClient(app)


def test_stream_generate_content_endpoint_sets_skip_done_flag():
    """LIT-3411 behavioral wiring test.

    The Google `:streamGenerateContent` endpoint handler must put
    `_litellm_skip_sse_done_terminator=True` on the request body it
    passes to `ProxyBaseLLMRequestProcessing`. We patch the processor
    and inspect the `data` it was constructed with — this verifies the
    *runtime behavior* of the endpoint, not its source layout, so the
    test still passes under whitespace/structural refactors and still
    fails if a reviewer accidentally inverts the conditional or drops
    the flag-set."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with (
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.base_process_llm_request",
            new_callable=AsyncMock,
            return_value={"ok": True},
        ),
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.__init__",
            return_value=None,
        ) as mock_init,
    ):
        resp = client.post(
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )
        assert resp.status_code == 200, resp.text
        mock_init.assert_called_once()
        data = mock_init.call_args.kwargs["data"]
        # Existing contract: stream is forced True on this route.
        assert data["stream"] is True
        # New LIT-3411 contract: the [DONE]-suppression flag is set.
        assert data.get("_litellm_skip_sse_done_terminator") is True, (
            "Google-native streamGenerateContent endpoint must mark "
            "request_data with `_litellm_skip_sse_done_terminator=True` "
            "so async_data_generator does not append the OpenAI-style "
            "`data: [DONE]` terminator (LIT-3411)."
        )


def test_generate_content_endpoint_does_not_set_skip_done_flag():
    """The non-streaming `:generateContent` route returns a single JSON
    body and never goes through `async_data_generator`. It must NOT set
    the suppression flag — this guards against accidental coupling
    creeping into the non-streaming handler."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")

    with (
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.base_process_llm_request",
            new_callable=AsyncMock,
            return_value={"ok": True},
        ),
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.__init__",
            return_value=None,
        ) as mock_init,
    ):
        resp = client.post(
            "/v1beta/models/gemini-2.0-flash:generateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]},
        )
        assert resp.status_code == 200, resp.text
        data = mock_init.call_args.kwargs["data"]
        assert (
            "_litellm_skip_sse_done_terminator" not in data
        ), "Non-streaming generateContent must not set the SSE done-terminator flag"
