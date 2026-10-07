"""The sync SDK paths of the response cache, which the proxy never reaches.

``litellm.completion`` and ``litellm.text_completion`` read and write the Redis cache in process. An upstream
answer with no choices is never stored, a stored entry whose choices were emptied is a miss, and the real answer
the refill brings back is the one served afterwards.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from typing import Final

import pytest

import litellm
from litellm.caching.caching import Cache
from litellm.types.caching import LiteLLMCacheType
from litellm.types.utils import ModelResponse, TextCompletionResponse
from tests.integration.caching.response_cache_case import (
    CHAT,
    CHAT_MODEL,
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
    scripted_in_process,
)


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
