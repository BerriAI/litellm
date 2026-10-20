"""
Tests for the cache-hit replay generators in
``litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response``.

These generators are used by ``LLMCachingHandler._convert_cached_stream_response``
to replay a cached non-streaming ``ModelResponse`` as a stream when the
incoming request has ``stream=True``. The fix in this test file ensures the
replay yields multiple word-shaped chunks instead of a single one-shot
content frame, restoring per-token cadence on cache hits.
"""

from typing import Final

import pytest

import litellm
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


def test_split_returns_empty_for_whitespace_only():
    # Must short-circuit before the regex: findall backtracks quadratically
    # on all-whitespace input.
    assert _split_assembled_content_for_replay("   ") == []
    assert _split_assembled_content_for_replay(" \n\t" * 10000) == []


# ---------- async generator ----------


@pytest.mark.asyncio
async def test_async_yields_multiple_content_chunks_with_lossless_join():
    text = "Sure! Here's a list of fruits: apple, banana, orange."
    chunks = await _collect_async(_async_payload(content=text))
    assert len(chunks) > 1
    assert all(isinstance(c, ModelResponseStream) for c in chunks)
    reassembled = "".join((c.choices[0].delta.content or "") for c in chunks)
    assert reassembled == text


@pytest.mark.parametrize("content", ["", "Hello world! How are you?"], ids=["single_frame", "sliced"])
@pytest.mark.asyncio
async def test_async_and_sync_replays_carry_the_cached_provider_specific_fields(content):
    provider_specific_fields = {"stop_reason": "refusal", "stop_details": {"category": "cyber"}}

    def payload():
        cached = _async_payload(content=content)
        cached["choices"][0]["message"]["provider_specific_fields"] = provider_specific_fields
        return cached

    async_chunks = await _collect_async(payload())
    sync_chunks = list(convert_to_streaming_response(payload()))

    assert [c.choices[0].delta.provider_specific_fields for c in async_chunks] == [
        c.choices[0].delta.provider_specific_fields for c in sync_chunks
    ]
    assert async_chunks[0].choices[0].delta.provider_specific_fields == provider_specific_fields


@pytest.mark.parametrize("provider_fields", [{"thought_signatures": ["sig"]}, None], ids=["provider_fields", "text"])
@pytest.mark.parametrize("replay", ["async", "sync"])
@pytest.mark.asyncio
async def test_a_replayed_stream_keeps_the_cached_completion_tokens_details(replay, provider_fields):
    cached = _async_payload()
    cached["choices"][0]["message"]["provider_specific_fields"] = provider_fields
    cached["usage"]["completion_tokens_details"] = {"reasoning_tokens": 4, "text_tokens": 3}

    chunks = await _collect_async(cached) if replay == "async" else list(convert_to_streaming_response(cached))
    rebuilt = litellm.stream_chunk_builder(chunks=chunks)

    assert rebuilt.choices[0].message.get("provider_specific_fields") == provider_fields
    assert rebuilt.usage.completion_tokens_details.reasoning_tokens == 4
    assert rebuilt.usage.completion_tokens_details.text_tokens == 3


@pytest.mark.parametrize("cached_fields", [7, [1, 2], "", "x" * 5000], ids=["int", "list", "empty", "long"])
@pytest.mark.asyncio
async def test_the_async_replay_ignores_cached_provider_fields_that_are_not_an_object(cached_fields):
    cached = _async_payload()
    cached["choices"][0]["message"]["provider_specific_fields"] = cached_fields

    chunks = await _collect_async(cached)

    assert "".join(c.choices[0].delta.content or "" for c in chunks) == "Hello world! How are you?"
    assert all(c.choices[0].delta.provider_specific_fields is None for c in chunks)


@pytest.mark.parametrize(
    "details",
    [{"reasoning_tokens": "abc"}, {"reasoning_tokens": 1.5}, {"reasoning_tokens": {"a": 1}}],
    ids=["text", "fraction", "object"],
)
@pytest.mark.parametrize("replay", ["async", "sync"])
@pytest.mark.asyncio
async def test_a_replayed_stream_skips_cached_completion_tokens_details_that_do_not_validate(replay, details):
    cached = _async_payload()
    cached["usage"]["completion_tokens_details"] = details

    chunks = await _collect_async(cached) if replay == "async" else list(convert_to_streaming_response(cached))
    rebuilt = litellm.stream_chunk_builder(chunks=chunks)

    assert rebuilt.choices[0].message.content == "Hello world! How are you?"
    assert rebuilt.usage.completion_tokens == 7


def _prompt_cached_payload(content: str, cache_read: int, cache_write: int) -> dict:
    cached = _async_payload(content=content)
    cached["usage"] = litellm.Usage(
        prompt_tokens=42145,
        completion_tokens=6,
        total_tokens=42151,
        cache_read_input_tokens=cache_read,
        cache_creation_input_tokens=cache_write,
    ).model_dump()
    return cached


@pytest.mark.parametrize("content", ["", "Hello world! How are you?"], ids=["single_frame", "sliced"])
@pytest.mark.parametrize("cache_read, cache_write", [(42139, 0), (0, 42139)], ids=["read", "write"])
@pytest.mark.parametrize("replay", ["async", "sync"])
@pytest.mark.asyncio
async def test_a_replayed_stream_keeps_the_cached_prompt_cache_tokens(replay, cache_read, cache_write, content):
    cached = _prompt_cached_payload(content, cache_read, cache_write)

    chunks = await _collect_async(cached) if replay == "async" else list(convert_to_streaming_response(cached))
    usage: Final = next(c.usage for c in chunks if getattr(c, "usage", None) is not None)

    assert usage.prompt_tokens == 42145
    assert usage.prompt_tokens_details.cached_tokens == cache_read
    assert usage.prompt_tokens_details.cache_write_tokens == cache_write
    assert usage.cache_read_input_tokens == cache_read
    assert usage.cache_creation_input_tokens == cache_write


@pytest.mark.parametrize("replay", ["async", "sync"])
@pytest.mark.asyncio
async def test_a_replayed_stream_keeps_cached_tokens_reported_only_in_the_prompt_tokens_details(replay):
    cached = _async_payload()
    cached["usage"]["prompt_tokens_details"] = {"cached_tokens": 4, "audio_tokens": 1}

    chunks = await _collect_async(cached) if replay == "async" else list(convert_to_streaming_response(cached))
    rebuilt = litellm.stream_chunk_builder(chunks=chunks)

    assert rebuilt.usage.prompt_tokens_details.cached_tokens == 4
    assert rebuilt.usage.prompt_tokens_details.audio_tokens == 1


@pytest.mark.parametrize("replay", ["async", "sync"])
@pytest.mark.asyncio
async def test_a_replayed_stream_keeps_integral_counts_cached_as_floats_or_numeric_strings(replay):
    cached = _async_payload()
    cached["usage"] = {
        "prompt_tokens": 100.0,
        "completion_tokens": "6",
        "total_tokens": 106.0,
        "cache_read_input_tokens": 90.0,
    }

    chunks = await _collect_async(cached) if replay == "async" else list(convert_to_streaming_response(cached))
    usage: Final = next(c.usage for c in chunks if getattr(c, "usage", None) is not None)

    assert (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens) == (100, 6, 106)
    assert usage.cache_read_input_tokens == 90


@pytest.mark.parametrize(
    "field, value",
    [
        ("prompt_tokens_details", {"cached_tokens": "abc"}),
        ("prompt_tokens_details", [1, 2]),
        ("cache_read_input_tokens", "abc"),
        ("cache_creation_input_tokens", 1.5),
    ],
    ids=["details_text", "details_list", "read_text", "write_fraction"],
)
@pytest.mark.parametrize("replay", ["async", "sync"])
@pytest.mark.asyncio
async def test_a_replayed_stream_skips_cached_prompt_cache_usage_that_does_not_validate(replay, field, value):
    cached = _async_payload()
    cached["usage"][field] = value

    chunks = await _collect_async(cached) if replay == "async" else list(convert_to_streaming_response(cached))
    rebuilt = litellm.stream_chunk_builder(chunks=chunks)

    assert rebuilt.choices[0].message.content == "Hello world! How are you?"
    assert rebuilt.usage.prompt_tokens == 5


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
async def test_async_tool_calls_and_function_call_only_on_first_chunk():
    # A cached response combining multi-word content with tool_calls must not
    # repeat the tool_calls on every slice — downstream handlers accumulate
    # tool-call deltas and would collect them N times.
    payload = _async_payload(content="I'll look that up for you")
    payload["choices"][0]["message"]["tool_calls"] = [
        {
            "id": "call_1",
            "type": "function",
            "function": {"name": "lookup", "arguments": "{}"},
            "index": 0,
        }
    ]
    chunks = await _collect_async(payload)
    assert len(chunks) > 1
    first_tool_calls = chunks[0].choices[0].delta.tool_calls
    assert first_tool_calls is not None and len(first_tool_calls) == 1
    assert first_tool_calls[0].function.name == "lookup"
    for c in chunks[1:]:
        assert c.choices[0].delta.tool_calls is None
        assert c.choices[0].delta.function_call is None


@pytest.mark.asyncio
async def test_async_reasoning_items_and_provider_specific_fields_only_on_first_chunk():
    reasoning_items: Final = [{"id": "rs_1", "type": "reasoning", "summary": [], "encrypted_content": "enc-1"}]
    provider_specific_fields: Final = {
        "tool_search_items": [{"id": "tsc_1", "type": "tool_search_call", "arguments": {"query": "weather"}}]
    }
    payload: Final = {
        **_async_payload(),
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "cached content slices here",
                    "reasoning_items": reasoning_items,
                    "provider_specific_fields": provider_specific_fields,
                },
                "finish_reason": "stop",
            }
        ],
    }
    chunks: Final = await _collect_async(payload)
    assert len(chunks) > 1
    first_delta: Final = chunks[0].choices[0].delta
    assert getattr(first_delta, "reasoning_items", None) == reasoning_items
    assert getattr(first_delta, "provider_specific_fields", None) == provider_specific_fields
    for c in chunks[1:]:
        assert getattr(c.choices[0].delta, "reasoning_items", None) is None
        assert getattr(c.choices[0].delta, "provider_specific_fields", None) is None


@pytest.mark.asyncio
async def test_async_logprobs_only_on_first_chunk():
    payload = _async_payload(content="cached token logprobs")
    payload["choices"][0]["logprobs"] = {"content": []}
    chunks = await _collect_async(payload)
    assert len(chunks) > 1
    assert chunks[0].choices[0].logprobs is not None
    for c in chunks[1:]:
        assert c.choices[0].logprobs is None


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


def test_sync_delta_and_choice_metadata_only_on_first_chunk():
    thinking_blocks = [{"type": "thinking", "thinking": "cached thinking", "signature": "sig"}]
    payload = {
        "id": "chatcmpl-test-sync",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "cached content slices",
                    "reasoning_content": "cached reasoning",
                    "thinking_blocks": thinking_blocks,
                },
                "finish_reason": "stop",
                "logprobs": {"content": []},
                "enhancements": {"source": "cache"},
            }
        ],
    }
    chunks = list(convert_to_streaming_response(payload))
    assert len(chunks) > 1
    first_choice = chunks[0].choices[0]
    assert getattr(first_choice.delta, "reasoning_content", None) == "cached reasoning"
    assert getattr(first_choice.delta, "thinking_blocks", None) == thinking_blocks
    assert first_choice.logprobs is not None
    assert first_choice.enhancements == {"source": "cache"}
    for c in chunks[1:]:
        choice = c.choices[0]
        assert getattr(choice.delta, "reasoning_content", None) is None
        assert getattr(choice.delta, "thinking_blocks", None) is None
        assert choice.logprobs is None
        assert getattr(choice, "enhancements", None) is None


def test_sync_non_enumerated_delta_fields_only_on_first_chunk():
    # annotations (and any other Delta field beyond the role/tool_call/reasoning
    # set) must also stay on the first slice. Rebuilding later slices as a bare
    # content delta drops the whole class, so this holds without enumerating
    # every field by hand.
    annotations = [
        {
            "type": "url_citation",
            "url_citation": {
                "url": "https://example.com",
                "title": "Example",
                "start_index": 0,
                "end_index": 1,
            },
        }
    ]
    payload = {
        "id": "chatcmpl-test-sync",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o-mini",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "cached content slices here",
                    "annotations": annotations,
                },
                "finish_reason": "stop",
            }
        ],
    }
    chunks = list(convert_to_streaming_response(payload))
    assert len(chunks) > 1
    assert getattr(chunks[0].choices[0].delta, "annotations", None) == annotations
    for c in chunks[1:]:
        assert getattr(c.choices[0].delta, "annotations", None) is None
