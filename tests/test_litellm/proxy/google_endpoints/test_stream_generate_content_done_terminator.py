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
2. `common_request_processing.py` sets that flag whenever the streaming
   `select_data_generator` branch fires with
   `route_type == "agenerate_content_stream"` (the Google-native route).
"""

import asyncio
import inspect

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
# 1. async_data_generator gating
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_data_generator_emits_done_by_default():
    """Sanity check: OpenAI-style streaming still gets the `[DONE]`
    terminator. This pins the existing behavior for chat-completions and
    every other caller that does NOT set the new flag."""
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
    """An explicit `False` flag still emits `[DONE]` — the gate is
    truthy-check only."""
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
# 2. common_request_processing.py wires the flag for the right route_type
# ---------------------------------------------------------------------------


def test_common_request_processing_sets_flag_for_generate_content_stream():
    """Static check that common_request_processing.py installs the flag
    for `route_type == "agenerate_content_stream"` immediately before
    calling `select_data_generator`. Tests this by reading the source of
    `base_process_llm_request` so we catch regressions where the flag
    move out of the streaming branch."""
    from litellm.proxy import common_request_processing

    src = inspect.getsource(
        common_request_processing.ProxyBaseLLMRequestProcessing.base_process_llm_request
    )

    assert (
        '"_litellm_skip_sse_done_terminator"' in src
    ), "_litellm_skip_sse_done_terminator flag must be set in base_process_llm_request"
    assert (
        'route_type == "agenerate_content_stream"' in src
    ), "Flag must be gated on the Google-native streamGenerateContent route_type"

    # The flag-set must live IN the `elif select_data_generator:` branch
    # — that's the streaming-passthrough path. We verify the flag-set
    # appears between the elif and the select_data_generator(...) call.
    idx_elif = src.index("elif select_data_generator:")
    idx_call = src.index("selected_data_generator = select_data_generator(", idx_elif)
    flag_set_idx = src.index('"_litellm_skip_sse_done_terminator"')
    assert idx_elif < flag_set_idx < idx_call, (
        "Flag must be set inside the `elif select_data_generator:` branch, "
        "before the generator is constructed."
    )
