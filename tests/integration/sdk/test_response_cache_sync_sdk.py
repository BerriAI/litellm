"""The sync SDK paths of the response cache, which the proxy never reaches.

``litellm.completion`` and ``litellm.text_completion`` read and write the Redis cache in process. An upstream
answer with no choices is never stored, a stored entry whose choices were emptied is a miss, and the real answer
the refill brings back is the one served afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from itertools import chain
from typing import Final

import pytest
from pydantic import JsonValue, TypeAdapter

import litellm
from litellm.caching.caching import Cache
from litellm.litellm_core_utils.streaming_handler import CustomStreamWrapper
from litellm.types.caching import LiteLLMCacheType
from litellm.types.utils import ModelResponse, TextCompletionResponse
from tests.integration._support.client import JSON_OBJECT
from tests.integration.caching.response_cache_case import (
    CHAT,
    CHAT_CHUNK,
    CHAT_MODEL,
    CHAT_STREAM,
    PROVIDER_KEY,
    TEXT,
    TEXT_MODEL,
    Upstream,
    await_entry,
    emptied,
    flip_entry_to_empty,
    json_response,
    prompt,
    redis_store,
    rescript,
    rewrite_usage_details,
    scripted_in_process,
)
from tests.integration.cost_calculation.cost_tracking_case import SseResponse


@pytest.fixture
def sdk_cache(monkeypatch: pytest.MonkeyPatch) -> Iterator[Cache]:
    cache: Final = Cache(type=LiteLLMCacheType.REDIS, host=os.environ["REDIS_HOST"], port=os.environ["REDIS_PORT"])
    monkeypatch.setattr(litellm, "cache", cache)
    yield cache
    monkeypatch.setattr(litellm, "cache", None)


def _completion(api_base: str, text: str) -> ModelResponse:
    response: Final = litellm.completion(
        model=CHAT_MODEL, messages=[{"role": "user", "content": text}], api_base=api_base, api_key=PROVIDER_KEY
    )
    assert isinstance(response, ModelResponse), type(response)
    return response


def _text_completion(api_base: str, text: str) -> TextCompletionResponse:
    response: Final = litellm.text_completion(model=TEXT_MODEL, prompt=text, api_base=api_base, api_key=PROVIDER_KEY)
    assert isinstance(response, TextCompletionResponse), type(response)
    return response


def _content(response: ModelResponse) -> object:
    assert len(response.choices) == 1, response.model_dump_json()
    return response.choices[0].model_dump()["message"]["content"]


def _text(response: TextCompletionResponse) -> object:
    assert len(response.choices) == 1, response.model_dump_json()
    return response.choices[0].model_dump()["text"]


@pytest.mark.usefixtures("sdk_cache")
def test_sync_completion_with_empty_choices_reaches_upstream_again_and_the_real_answer_is_cached() -> None:
    upstream: Final = Upstream()
    text: Final = prompt()
    with scripted_in_process(json_response(emptied(CHAT, "choices"))) as handle:
        empty: Final = _completion(handle.api_base(), text)
        assert empty.choices == [], empty.model_dump_json()
        assert upstream.calls(handle.scenario_id) == 1
        rescript(handle, json_response(CHAT))
        refilled: Final = _completion(handle.api_base(), text)
        assert _content(refilled) == "scripted", f"the empty answer was served: {refilled.model_dump_json()}"
        assert refilled.id != empty.id
        assert upstream.calls(handle.scenario_id) == 2
        with redis_store() as store:
            await_entry(store, refilled.id)
        hit: Final = _completion(handle.api_base(), text)
        assert hit.id == refilled.id, hit.model_dump_json()
        assert _content(hit) == "scripted"
        assert upstream.calls(handle.scenario_id) == 2


@pytest.mark.usefixtures("sdk_cache")
def test_sync_completion_treats_a_stale_entry_with_empty_choices_as_a_miss() -> None:
    upstream: Final = Upstream()
    text: Final = prompt()
    with scripted_in_process(json_response(CHAT)) as handle, redis_store() as store:
        first: Final = _completion(handle.api_base(), text)
        assert _content(first) == "scripted", first.model_dump_json()
        stale_id: Final = flip_entry_to_empty(store, await_entry(store, first.id), "choices")
        refilled: Final = _completion(handle.api_base(), text)
        assert refilled.id != stale_id, f"the stale entry was served: {refilled.model_dump_json()}"
        assert _content(refilled) == "scripted" and refilled.id != first.id, refilled.model_dump_json()
        assert upstream.calls(handle.scenario_id) == 2
        await_entry(store, refilled.id)
        hit: Final = _completion(handle.api_base(), text)
        assert hit.id == refilled.id, hit.model_dump_json()
        assert upstream.calls(handle.scenario_id) == 2


@pytest.mark.usefixtures("sdk_cache")
def test_sync_text_completion_with_empty_choices_reaches_upstream_again() -> None:
    upstream: Final = Upstream()
    text: Final = prompt()
    with scripted_in_process(json_response(emptied(TEXT, "choices"))) as handle:
        empty: Final = _text_completion(handle.api_base(), text)
        assert empty.choices == [], empty.model_dump_json()
        assert upstream.calls(handle.scenario_id) == 1
        rescript(handle, json_response(TEXT))
        refilled: Final = _text_completion(handle.api_base(), text)
        assert _text(refilled) == "scripted", f"the empty answer was served: {refilled.model_dump_json()}"
        assert upstream.calls(handle.scenario_id) == 2
        with redis_store() as store:
            await_entry(store, refilled.id)
        hit: Final = _text_completion(handle.api_base(), text)
        assert hit.id == refilled.id, hit.model_dump_json()
        assert upstream.calls(handle.scenario_id) == 2


_REASONING_USAGE_FRAME: Final = (
    CHAT_CHUNK + '[],"usage":{"prompt_tokens":5,"completion_tokens":7,"total_tokens":12,'
    '"completion_tokens_details":{"reasoning_tokens":4}}}'
)
_REASONING_STREAM: Final = SseResponse(
    content_type="text/event-stream",
    frames=(*CHAT_STREAM.frames[:-1], _REASONING_USAGE_FRAME, CHAT_STREAM.frames[-1]),
)
_JSON_LIST: Final = TypeAdapter(list[JsonValue])
_UNVALIDATED_USAGE_DETAILS: Final[dict[str, JsonValue]] = {
    "text": {"reasoning_tokens": "abc"},
    "fraction": {"reasoning_tokens": 1.5},
    "object": {"reasoning_tokens": {"a": 1}},
}


def _stream(api_base: str, text: str) -> tuple[dict[str, JsonValue], ...]:
    stream: Final = litellm.completion(
        model=CHAT_MODEL,
        messages=[{"role": "user", "content": text}],
        api_base=api_base,
        api_key=PROVIDER_KEY,
        stream=True,
        stream_options={"include_usage": True},
    )
    assert isinstance(stream, CustomStreamWrapper), type(stream)
    return tuple(JSON_OBJECT.validate_python(chunk.model_dump(mode="json")) for chunk in stream)


def _replayed(chunks: tuple[dict[str, JsonValue], ...]) -> tuple[frozenset[JsonValue], str, tuple[JsonValue, ...]]:
    choices: Final = tuple(
        JSON_OBJECT.validate_python(choice)
        for choice in chain.from_iterable(_JSON_LIST.validate_python(chunk.get("choices") or []) for chunk in chunks)
    )
    text: Final = "".join(str(JSON_OBJECT.validate_python(choice["delta"]).get("content") or "") for choice in choices)
    usages: Final = tuple(JSON_OBJECT.validate_python(chunk["usage"]) for chunk in chunks if chunk.get("usage"))
    reasoning: Final = tuple(
        JSON_OBJECT.validate_python(usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
        for usage in usages
    )
    return frozenset(chunk["id"] for chunk in chunks), text, reasoning


@pytest.mark.usefixtures("sdk_cache")
def test_sync_stream_cache_hit_replays_the_usage_details_of_the_miss() -> None:
    upstream: Final = Upstream()
    text: Final = prompt()
    with scripted_in_process(_REASONING_STREAM) as handle, redis_store() as store:
        first: Final = _replayed(_stream(handle.api_base(), text))
        (first_id,) = first[0]
        assert first[1:] == ("streamed response", (4,)), first
        await_entry(store, str(first_id))
        hit: Final = _replayed(_stream(handle.api_base(), text))
        assert hit == first, hit
        assert upstream.calls(handle.scenario_id) == 1


@pytest.mark.usefixtures("sdk_cache")
@pytest.mark.parametrize("variant", tuple(_UNVALIDATED_USAGE_DETAILS))
def test_sync_stream_cache_hit_skips_cached_usage_details_that_fail_validation(variant: str) -> None:
    upstream: Final = Upstream()
    text: Final = prompt()
    with scripted_in_process(_REASONING_STREAM) as handle, redis_store() as store:
        first: Final = _replayed(_stream(handle.api_base(), text))
        (first_id,) = first[0]
        assert first[1:] == ("streamed response", (4,)), first
        stale_id: Final = rewrite_usage_details(
            store, await_entry(store, str(first_id)), _UNVALIDATED_USAGE_DETAILS[variant]
        )
        hit: Final = _replayed(_stream(handle.api_base(), text))
        assert hit == (frozenset({stale_id}), "streamed response", (0,)), hit
        assert upstream.calls(handle.scenario_id) == 1
