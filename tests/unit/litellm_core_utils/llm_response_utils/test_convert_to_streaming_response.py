"""
Tests for the cache-hit replay generators in
``litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response``.

These generators are used by ``LLMCachingHandler._convert_cached_stream_response``
to replay a cached non-streaming ``ModelResponse`` as a stream when the
incoming request has ``stream=True``. The fix in this test file ensures the
replay yields multiple word-shaped chunks instead of a single one-shot
content frame, restoring per-token cadence on cache hits.
"""

from datetime import datetime
from typing import Final, Literal

import pytest
from pydantic import JsonValue

import litellm
from litellm.litellm_core_utils.llm_response_utils.convert_dict_to_response import (
    _split_assembled_content_for_replay,
    convert_to_streaming_response,
    convert_to_streaming_response_async,
)
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
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
async def test_async_empty_content_yields_the_role_then_a_separate_finish_frame():
    payload = _async_payload(content=None)
    payload["usage"] = None
    chunks = await _collect_async(payload)
    assert [(c.choices[0].delta.role, c.choices[0].finish_reason) for c in chunks] == [
        ("assistant", None),
        (None, "stop"),
    ]
    assert all(c.choices[0].delta.content is None for c in chunks)
    assert all(getattr(c, "usage", None) is None for c in chunks)


@pytest.mark.asyncio
async def test_async_tool_calls_arrive_once_after_the_last_content_slice():
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
    called: Final = [i for i, c in enumerate(chunks) if c.choices[0].delta.tool_calls]
    last_content: Final = max(i for i, c in enumerate(chunks) if c.choices[0].delta.content)
    assert called == [last_content + 1] and last_content > 0
    assert chunks[called[0]].choices[0].delta.tool_calls[0].function.name == "lookup"
    assert chunks[called[0]].choices[0].delta.content is None
    assert all(c.choices[0].delta.function_call is None for c in chunks)
    assert chunks[-1].choices[0].finish_reason == "stop" and len(chunks) == called[0] + 2


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


def test_sync_reasoning_leads_and_choice_metadata_rides_the_first_content_slice():
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
    lead: Final = chunks[0].choices[0]
    assert getattr(lead.delta, "reasoning_content", None) == "cached reasoning"
    assert getattr(lead.delta, "thinking_blocks", None) == thinking_blocks
    assert lead.delta.content is None
    first_text: Final = chunks[1].choices[0]
    assert first_text.delta.content and first_text.logprobs is not None
    assert first_text.enhancements == {"source": "cache"}
    for index, c in enumerate(chunks):
        choice = c.choices[0]
        assert index == 0 or getattr(choice.delta, "reasoning_content", None) is None
        assert index == 0 or getattr(choice.delta, "thinking_blocks", None) is None
        assert index == 1 or choice.logprobs is None
        assert index == 1 or getattr(choice, "enhancements", None) is None


def test_sync_annotations_arrive_once_with_the_last_content_slice():
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
    annotated: Final = [i for i, c in enumerate(chunks) if getattr(c.choices[0].delta, "annotations", None)]
    last_content: Final = max(i for i, c in enumerate(chunks) if c.choices[0].delta.content)
    assert annotated == [last_content] and last_content > 0
    assert chunks[last_content].choices[0].delta.annotations == annotations


async def _client_stream(
    message: dict[str, JsonValue], replay: Literal["async", "sync"]
) -> list[dict[str, JsonValue]]:
    payload: Final = {
        **_async_payload(),
        "choices": [{"index": 0, "message": {"role": "assistant", **message}, "finish_reason": "stop"}],
    }
    logging_obj: Final = litellm.Logging(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": "hi"}],
        stream=True,
        call_type="acompletion",
        start_time=datetime.now(),
        litellm_call_id="cached-replay",
        function_id="cached-replay",
    )
    source: Final = (
        convert_to_streaming_response_async(response_object=payload)
        if replay == "async"
        else convert_to_streaming_response(response_object=payload)
    )
    wrapper: Final = CustomStreamWrapper(
        completion_stream=source, model="gpt-4o-mini", custom_llm_provider="cached_response", logging_obj=logging_obj
    )
    chunks: Final = [chunk async for chunk in wrapper] if replay == "async" else list(wrapper)
    return [
        {**chunk.choices[0].delta.model_dump(exclude_none=True), "finish": chunk.choices[0].finish_reason}
        for chunk in chunks
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", ["async", "sync"])
@pytest.mark.parametrize(
    "field, value",
    [
        ("audio", {"id": "audio_1", "data": "UklGRg==", "expires_at": 1700000600, "transcript": "Hello"}),
        ("images", [{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}, "index": 0}]),
    ],
)
async def test_a_cached_answer_without_text_streams_its_role_and_payload_to_the_client(
    replay: Literal["async", "sync"], field: str, value: JsonValue
) -> None:
    deltas: Final = await _client_stream({"content": None, field: value}, replay)
    assert deltas[0]["role"] == "assistant"
    assert [delta[field] for delta in deltas if field in delta] == [value]
    assert deltas[-1]["finish"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", ["async", "sync"])
async def test_a_cached_answer_streams_reasoning_before_the_text_and_citations_after_it(
    replay: Literal["async", "sync"],
) -> None:
    annotations: Final[JsonValue] = [
        {
            "type": "url_citation",
            "url_citation": {"url": "https://example.com", "title": "Example", "start_index": 0, "end_index": 23},
        }
    ]
    deltas: Final = await _client_stream(
        {
            "content": "The answer is 14 apples",
            "reasoning_content": "Count the apples in each basket",
            "annotations": annotations,
        },
        replay,
    )
    reasoning: Final = [i for i, delta in enumerate(deltas) if "reasoning_content" in delta]
    text: Final = [i for i, delta in enumerate(deltas) if delta.get("content")]
    cited: Final = [i for i, delta in enumerate(deltas) if "annotations" in delta]
    assert "".join(str(deltas[i]["reasoning_content"]) for i in reasoning) == "Count the apples in each basket"
    assert "".join(str(deltas[i]["content"]) for i in text) == "The answer is 14 apples"
    assert max(reasoning) < min(text)
    assert cited == [max(text)] and deltas[max(text)]["annotations"] == annotations
    assert deltas[-1]["finish"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", ["async", "sync"])
async def test_a_cached_answer_streams_its_tool_call_after_the_reasoning_and_the_text(
    replay: Literal["async", "sync"],
) -> None:
    tool_call: Final[JsonValue] = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "count_apples", "arguments": '{"basket": "all"}'},
        "index": 0,
    }
    deltas: Final = await _client_stream(
        {
            "content": "Let me count the apples",
            "reasoning_content": "I need the basket totals",
            "tool_calls": [tool_call],
        },
        replay,
    )
    reasoning: Final = [i for i, delta in enumerate(deltas) if "reasoning_content" in delta]
    text: Final = [i for i, delta in enumerate(deltas) if delta.get("content")]
    called: Final = [i for i, delta in enumerate(deltas) if delta.get("tool_calls")]
    assert "".join(str(deltas[i]["content"]) for i in text) == "Let me count the apples"
    assert reasoning and max(reasoning) < min(text)
    assert called == [max(text) + 1]
    assert deltas[called[0]]["tool_calls"] == [tool_call]
    assert deltas[-1]["finish"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", ["async", "sync"])
async def test_a_cached_tool_turn_without_text_streams_the_reasoning_before_the_tool_call(
    replay: Literal["async", "sync"],
) -> None:
    tool_call: Final[JsonValue] = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "count_apples", "arguments": '{"basket": "all"}'},
        "index": 0,
    }
    deltas: Final = await _client_stream(
        {"content": None, "reasoning_content": "I need the basket totals", "tool_calls": [tool_call]}, replay
    )
    reasoning: Final = [i for i, delta in enumerate(deltas) if "reasoning_content" in delta]
    called: Final = [i for i, delta in enumerate(deltas) if delta.get("tool_calls")]
    assert deltas[0]["role"] == "assistant"
    assert "".join(str(deltas[i]["reasoning_content"]) for i in reasoning) == "I need the basket totals"
    assert reasoning and called and max(reasoning) < min(called)
    assert [deltas[i]["tool_calls"] for i in called] == [[tool_call]]
    assert deltas[-1]["finish"] == "stop"


@pytest.mark.asyncio
@pytest.mark.parametrize("replay", ["async", "sync"])
@pytest.mark.parametrize("whitespace", ["\n\n", " "], ids=["newlines", "space"])
async def test_a_cached_tool_turn_keeps_its_whitespace_only_text_between_the_reasoning_and_the_tool_call(
    replay: Literal["async", "sync"], whitespace: str
) -> None:
    tool_call: Final[JsonValue] = {
        "id": "call_1",
        "type": "function",
        "function": {"name": "count_apples", "arguments": '{"basket": "all"}'},
        "index": 0,
    }
    deltas: Final = await _client_stream(
        {"content": whitespace, "reasoning_content": "I need the basket totals", "tool_calls": [tool_call]}, replay
    )
    reasoning: Final = [i for i, delta in enumerate(deltas) if "reasoning_content" in delta]
    text: Final = [i for i, delta in enumerate(deltas) if delta.get("content")]
    called: Final = [i for i, delta in enumerate(deltas) if delta.get("tool_calls")]
    assert "".join(str(deltas[i]["content"]) for i in text) == whitespace
    assert reasoning and text and called and max(reasoning) < min(text) and max(text) < min(called)
    assert [deltas[i]["tool_calls"] for i in called] == [[tool_call]]
    assert deltas[-1]["finish"] == "stop"
