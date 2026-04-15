"""
Tests for the cache-hit replay generators in
``litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response``.

These generators are used by ``LLMCachingHandler._convert_cached_stream_response``
to replay a cached non-streaming ``ModelResponse`` as a stream when the
incoming request has ``stream=True``. The fix in this test file ensures the
replay yields multiple word-shaped chunks instead of a single one-shot
content frame, restoring per-token cadence on cache hits (case
2026-04-13-pramod-streaming-buffered-subsequent-requests).
"""

import pytest

from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _split_assembled_content_for_replay,
    convert_to_streaming_response,
    convert_to_streaming_response_async,
)
from litellm.types.utils import ModelResponseStream


def _async_payload(content="Hello world! How are you?"):
    return {
        "id": "chatcmpl-test-async",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 5,
            "completion_tokens": 7,
            "total_tokens": 12,
        },
    }


async def _collect_async(payload):
    return [chunk async for chunk in convert_to_streaming_response_async(payload)]


# ---------- helper ----------


@pytest.mark.parametrize(
    "text",
    [
        "Hello world!",
        "Sure! Here's a list of 25 fruits:\n\n1. Apple\n2. Banana\n3. Orange\n",
        " leading whitespace matters",
        "Hi",
        "你好世界",
    ],
)
def test_split_is_lossless(text):
    assert "".join(_split_assembled_content_for_replay(text)) == text


def test_split_returns_empty_for_none_and_empty():
    assert _split_assembled_content_for_replay(None) == []
    assert _split_assembled_content_for_replay("") == []


# ---------- async generator ----------


@pytest.mark.asyncio
async def test_async_yields_multiple_content_chunks_with_lossless_join():
    text = "Sure! Here's a list of fruits: apple, banana, orange."
    chunks = await _collect_async(_async_payload(content=text))
    assert len(chunks) > 1
    assert all(isinstance(c, ModelResponseStream) for c in chunks)
    reassembled = "".join((c.choices[0].delta.content or "") for c in chunks)
    assert reassembled == text


@pytest.mark.asyncio
async def test_async_finish_reason_only_on_last_chunk():
    chunks = await _collect_async(_async_payload())
    finish_reasons = [c.choices[0].finish_reason for c in chunks]
    assert finish_reasons[-1] == "stop"
    assert all(fr is None for fr in finish_reasons[:-1])


@pytest.mark.asyncio
async def test_async_role_only_on_first_chunk():
    chunks = await _collect_async(_async_payload())
    assert chunks[0].choices[0].delta.role == "assistant"
    for c in chunks[1:]:
        assert c.choices[0].delta.role is None


@pytest.mark.asyncio
async def test_async_usage_attached_to_last_chunk_only():
    chunks = await _collect_async(_async_payload())
    usage_frames = [c for c in chunks if getattr(c, "usage", None) is not None]
    assert len(usage_frames) == 1
    assert usage_frames[0] is chunks[-1]
    assert usage_frames[0].usage.completion_tokens == 7
    assert usage_frames[0].usage.prompt_tokens == 5
    assert usage_frames[0].usage.total_tokens == 12
    assert usage_frames[0].choices[0].finish_reason == "stop"


@pytest.mark.asyncio
async def test_async_empty_content_yields_single_finish_frame():
    # tool-calls-only-style response: short-circuit to the single-yield path.
    payload = _async_payload(content=None)
    payload["usage"] = None
    chunks = await _collect_async(payload)
    assert len(chunks) == 1
    assert chunks[0].choices[0].delta.content is None
    assert chunks[0].choices[0].finish_reason == "stop"
    assert chunks[0].choices[0].delta.role == "assistant"
    assert getattr(chunks[0], "usage", None) is None


@pytest.mark.asyncio
async def test_async_metadata_propagated_to_every_chunk():
    chunks = await _collect_async(_async_payload())
    for c in chunks:
        assert c.id == "chatcmpl-test-async"
        assert c.model == "gpt-4o-mini"
        assert c.system_fingerprint == "fp_test"
        assert c.created == 1700000000


# ---------- sync generator (parity smoke test) ----------


def test_sync_multi_chunk_and_lossless_join():
    text = "Sure! Here's a list of fruits: apple, banana, orange."
    payload = {
        "id": "chatcmpl-test-sync",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "system_fingerprint": "fp_test",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": text},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 12},
    }
    chunks = list(convert_to_streaming_response(payload))
    assert len(chunks) > 1
    reassembled = "".join((c.choices[0].delta.content or "") for c in chunks)
    assert reassembled == text
    # Sync path must also honor the "usage on last chunk" invariant.
    assert chunks[-1].choices[0].finish_reason == "stop"
    assert getattr(chunks[-1], "usage", None) is not None
    assert chunks[-1].usage.total_tokens == 12
