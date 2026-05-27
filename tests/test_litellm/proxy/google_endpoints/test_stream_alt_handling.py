"""
Tests for client `?alt=` handling on the native
`/v1beta/models/{model}:streamGenerateContent` endpoint.

Background: LiteLLM streams from upstream Gemini with `alt=sse` so it can
parse chunks server-side. When the client (e.g. the google-genai Python SDK)
does not pass `?alt=sse`, Gemini's documented contract returns raw JSON, and
the SDK relies on that. Without re-framing, LiteLLM would forward
`data: {...}` SSE lines to the SDK and JSON parsing would fail with
`google.genai.errors.UnknownApiResponseError: Failed to parse response as JSON`.

These tests exercise:
1. The endpoint captures `?alt=` and propagates it via `litellm_metadata`,
   defaulting to "json" when the client omits it (Gemini's documented default).
2. `async_data_generator` strips SSE framing AND omits the `data: [DONE]`
   sentinel when the client did not request SSE; both are kept otherwise.
"""

import os
import sys
from unittest.mock import AsyncMock, patch

import pytest

sys.path.insert(0, os.path.abspath("../../.."))


def _build_test_client():
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    from litellm.proxy.google_endpoints.endpoints import router as google_router

    app = FastAPI()
    app.include_router(google_router)
    return TestClient(app)


def _patch_base_process(return_value=None):
    if return_value is None:
        return_value = {"test": "response"}
    return patch(
        "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.base_process_llm_request",
        new_callable=AsyncMock,
        return_value=return_value,
    )


def test_stream_endpoint_records_alt_sse_in_metadata():
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")
    with (
        _patch_base_process(),
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.__init__",
            return_value=None,
        ) as mock_init,
    ):
        response = client.post(
            "/v1beta/models/test-model:streamGenerateContent?alt=sse",
            json={"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
        )
        assert response.status_code == 200
        init_kwargs = mock_init.call_args.kwargs
        meta = init_kwargs["data"].get("litellm_metadata", {})
        assert meta.get("client_requested_stream_format") == "sse"


def test_stream_endpoint_records_alt_json_in_metadata():
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")
    with (
        _patch_base_process(),
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.__init__",
            return_value=None,
        ) as mock_init,
    ):
        response = client.post(
            "/v1beta/models/test-model:streamGenerateContent?alt=json",
            json={"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
        )
        assert response.status_code == 200
        init_kwargs = mock_init.call_args.kwargs
        meta = init_kwargs["data"].get("litellm_metadata", {})
        assert meta.get("client_requested_stream_format") == "json"


def test_stream_endpoint_no_alt_defaults_to_json():
    """When the client omits ?alt= (the google-genai SDK default), the proxy
    must record "json" as the client-requested format. Per the Gemini REST
    API contract, omitting `alt` means "return JSON" -- it is not a
    "no preference, use server default" signal. This is the actual reported
    bug: the SDK omits alt, expects JSON, but LiteLLM was returning SSE."""
    try:
        client = _build_test_client()
    except ImportError as e:
        pytest.skip(f"Skipping test due to missing dependency: {e}")
    with (
        _patch_base_process(),
        patch(
            "litellm.proxy.google_endpoints.endpoints.ProxyBaseLLMRequestProcessing.__init__",
            return_value=None,
        ) as mock_init,
    ):
        response = client.post(
            "/v1beta/models/test-model:streamGenerateContent",
            json={"contents": [{"role": "user", "parts": [{"text": "Hi"}]}]},
        )
        assert response.status_code == 200
        init_kwargs = mock_init.call_args.kwargs
        meta = init_kwargs["data"].get("litellm_metadata", {})
        assert meta.get("client_requested_stream_format") == "json"


class _FakeAsyncIterator:
    """Yields a sequence of pre-prepared bytes chunks, then stops."""

    def __init__(self, chunks):
        self._chunks = list(chunks)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._chunks:
            raise StopAsyncIteration
        return self._chunks.pop(0)

    async def aclose(self):
        return None


SSE_FROM_GEMINI = [
    b'data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}\n\n',
    b'data: {"candidates":[{"content":{"parts":[{"text":" world"}],"role":"model"}}]}\n\n',
]


def _make_request_data(alt=None):
    d = {"model": "gemini-pro", "stream": True}
    if alt is not None:
        d.setdefault("litellm_metadata", {})["client_requested_stream_format"] = alt
    return d


@pytest.mark.asyncio
async def test_strip_sse_when_client_did_not_request_sse():
    """alt=json: strip SSE framing, omit [DONE] sentinel, emit JSON only."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import async_data_generator

    request_data = _make_request_data(alt="json")
    iterator = _FakeAsyncIterator(SSE_FROM_GEMINI)
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u")

    out = []
    async for piece in async_data_generator(
        response=iterator,
        user_api_key_dict=user_api_key_dict,
        request_data=request_data,
    ):
        out.append(piece)

    body = "".join(out)
    assert "data: " not in body, f"unexpected SSE prefix: {body!r}"
    assert "[DONE]" not in body, f"unexpected [DONE]: {body!r}"
    import json

    lines = [ln for ln in body.split("\n") if ln.strip()]
    assert len(lines) == 2, f"expected 2 JSON lines, got {lines!r}"
    parsed = [json.loads(ln) for ln in lines]
    assert parsed[0]["candidates"][0]["content"]["parts"][0]["text"] == "Hello"
    assert parsed[1]["candidates"][0]["content"]["parts"][0]["text"] == " world"


@pytest.mark.asyncio
async def test_keep_sse_when_client_requested_sse():
    """alt=sse: preserve SSE framing and `data: [DONE]` sentinel."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import async_data_generator

    request_data = _make_request_data(alt="sse")
    iterator = _FakeAsyncIterator(SSE_FROM_GEMINI)
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u")

    out = []
    async for piece in async_data_generator(
        response=iterator,
        user_api_key_dict=user_api_key_dict,
        request_data=request_data,
    ):
        out.append(piece)

    body = "".join(out)
    assert 'data: {"candidates"' in body
    assert '"text":"Hello"' in body
    assert '"text":" world"' in body
    assert "data: [DONE]" in body


@pytest.mark.asyncio
async def test_no_alt_metadata_keeps_existing_sse_behavior():
    """No alt metadata at all (defensive path for non-google routes): keep
    SSE framing and [DONE] sentinel -- fully backward-compatible."""
    from litellm.proxy._types import UserAPIKeyAuth
    from litellm.proxy.proxy_server import async_data_generator

    request_data = _make_request_data(alt=None)
    iterator = _FakeAsyncIterator(SSE_FROM_GEMINI)
    user_api_key_dict = UserAPIKeyAuth(api_key="sk-test", user_id="u")

    out = []
    async for piece in async_data_generator(
        response=iterator,
        user_api_key_dict=user_api_key_dict,
        request_data=request_data,
    ):
        out.append(piece)

    body = "".join(out)
    assert 'data: {"candidates"' in body
    assert "data: [DONE]" in body
