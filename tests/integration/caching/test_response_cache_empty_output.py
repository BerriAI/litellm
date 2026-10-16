"""The response cache never stores an answer with no output and never serves a stored one.

Every cell scripts the owned upstream for one deployment, drives the rig proxy through the client a
user would hold (OpenAI SDK, Anthropic SDK, raw httpx) and reads three things: what the caller got,
how often the upstream was called for that deployment (an answer served from the cache never reaches
it), and what landed in Redis or the spend log. An upstream answer carrying no choices, no output items
or no content blocks reaches the upstream again on the next identical request, and the real answer
that follows is the one the cache keeps. Answers with output, errors, ``no-cache`` requests, the
``/cache/delete`` workaround and the other cached call types are unchanged.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import partial
from hashlib import sha256
from typing import Final, TypeAlias, TypeVar

import httpx
import pytest
from anthropic import Anthropic, AsyncAnthropic
from openai import AsyncOpenAI, OpenAI
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value, string_value
from tests.integration._support.database import read_rows
from tests.integration.caching.response_cache_case import (
    CHAT,
    CHAT_MODEL,
    CHAT_STREAM,
    CHAT_STREAM_WITHOUT_CHOICES,
    EMBEDDING,
    MESSAGE,
    MESSAGE_STREAM,
    MESSAGE_STREAM_WITHOUT_CONTENT_BLOCKS,
    MESSAGES_MODEL,
    RERANK,
    RESPONSE,
    RESPONSE_STREAM,
    RESPONSE_STREAM_WITHOUT_OUTPUT,
    SHORT_TTL_CACHE_CONTROL,
    TEXT,
    TEXT_MODEL,
    TEXT_STREAM_WITHOUT_CHOICES,
    TRANSCRIPT,
    Upstream,
    await_entry,
    chat_body,
    choices,
    content_of_first_choice,
    emptied,
    flip_entry_to_empty,
    json_response,
    message_body,
    prompt,
    redis_store,
    rescript,
    response_id,
    scripted,
    text_of_first_choice,
)

MISSING: Final = object()
EMPTY_STREAMS: Final = 6
REPLAY_ATTEMPTS: Final = 8
SETTLE_SECONDS: Final = 15
T = TypeVar("T")
_Answer: TypeAlias = tuple[frozenset[str], str, tuple[str, ...]]
SPEND_SQL: Final = (
    'SELECT request_id, cache_hit, spend FROM "LiteLLM_SpendLogs" WHERE api_key = %s ORDER BY "startTime", request_id'
)


def _proxy_root(gateway: Gateway) -> str:
    return str(gateway.client.base_url).rstrip("/")


def _openai(gateway: Gateway, key: str | None = None) -> OpenAI:
    return OpenAI(base_url=f"{_proxy_root(gateway)}/v1", api_key=key or gateway.key, max_retries=0)


def _async_openai(gateway: Gateway, key: str | None = None) -> AsyncOpenAI:
    return AsyncOpenAI(base_url=f"{_proxy_root(gateway)}/v1", api_key=key or gateway.key, max_retries=0)


def _anthropic(gateway: Gateway) -> Anthropic:
    return Anthropic(base_url=_proxy_root(gateway), api_key=gateway.key, max_retries=0)


def _async_anthropic(gateway: Gateway) -> AsyncAnthropic:
    return AsyncAnthropic(base_url=_proxy_root(gateway), api_key=gateway.key, max_retries=0)


def _chat(gateway: Gateway, body: Mapping[str, JsonValue], *, key: str | None = None) -> httpx.Response:
    return gateway.request("POST", "/v1/chat/completions", body, key=key)


def _spend_rows(key: str) -> list[dict[str, JsonValue]]:
    return read_rows(SPEND_SQL, (sha256(key.encode()).hexdigest(),))


def _assert_prompt_reached_upstream(upstream: Upstream, identity: str, text: str, times: int) -> None:
    received: Final = upstream.received(identity)
    assert len(received) == times, f"{identity}: {len(received)} upstream calls, expected {times}"
    assert all(text in json.dumps(item["body"]) for item in received), received


def _served_without_upstream(upstream: Upstream, identity: str, call: Callable[[], T]) -> T | None:
    before: Final = upstream.calls(identity)
    answer: Final = call()
    return None if upstream.calls(identity) > before else answer


@dataclass(frozen=True, slots=True)
class Streamed:
    ids: frozenset[str]
    text: str
    kinds: tuple[str, ...]
    reached_upstream: bool


def _streams(stream: Callable[[], _Answer], upstream: Upstream, identity: str, attempts: int) -> Iterator[Streamed]:
    """Stream up to ``attempts`` times, stopping after the first answer served without an upstream call."""
    for _ in range(attempts):
        before: Final = upstream.calls(identity)
        ids, text, kinds = stream()
        answer: Final = Streamed(ids, text, kinds, upstream.calls(identity) > before)
        yield answer
        if not answer.reached_upstream:
            return


def _never_replayed(stream: Callable[[], _Answer], upstream: Upstream, identity: str) -> tuple[Streamed, ...]:
    """``EMPTY_STREAMS`` identical streams, every one reaching the upstream and carrying ids of its own."""
    answers: Final = tuple(_streams(stream, upstream, identity, EMPTY_STREAMS))
    assert len(answers) == EMPTY_STREAMS and all(answer.reached_upstream for answer in answers), (
        f"a stream was replayed from the cache: {answers}"
    )
    distinct: Final = frozenset[str]().union(*(answer.ids for answer in answers))
    assert sum(len(answer.ids) for answer in answers) == len(distinct), answers
    return answers


def _replayed(stream: Callable[[], _Answer], upstream: Upstream, identity: str, produced: frozenset[str]) -> Streamed:
    """The first of up to ``REPLAY_ATTEMPTS`` identical streams served without an upstream call.

    A streamed request can reach the upstream once per proxy worker before a replay is served: the proxy adds
    ``stream_options`` to a stream before keying it only when its own router already resolves the model, and the
    worker that did not take the ``/model/new`` call resolves it once its router syncs. The replay carries ids an
    upstream answer produced.
    """
    answers: Final = tuple(_streams(stream, upstream, identity, REPLAY_ATTEMPTS))
    replay: Final = answers[-1]
    assert not replay.reached_upstream, (
        f"no stream was replayed from the cache in {REPLAY_ATTEMPTS} attempts: {answers}"
    )
    seen: Final = produced.union(*(answer.ids for answer in answers[:-1]))
    assert replay.ids <= seen, (replay.ids, seen)
    return replay


def test_chat_empty_choices_reach_upstream_again_and_the_real_answer_is_cached(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario, _openai(gateway) as client:
        handle: Final = scripted(scenario, json_response(emptied(CHAT, "choices")))
        model: Final = scenario.model(api_base=handle.api_base())
        empty: Final = client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}])
        assert empty.choices == [], empty.model_dump_json()
        _assert_prompt_reached_upstream(upstream, handle.scenario_id, text, 1)
        rescript(handle, json_response(CHAT))
        real: Final = client.chat.completions.create(model=model, messages=[{"role": "user", "content": text}])
        assert real.id != empty.id and real.choices[0].message.content == "scripted", real.model_dump_json()
        _assert_prompt_reached_upstream(upstream, handle.scenario_id, text, 2)
        await_entry(store, real.id)
        hit: Final = client.chat.completions.with_raw_response.create(
            model=model, messages=[{"role": "user", "content": text}]
        )
        served: Final = hit.parse()
        assert served.id == real.id and served.choices[0].message.content == "scripted", hit.text
        assert hit.headers.get("x-litellm-cache-key"), dict(hit.headers)
        assert upstream.calls(handle.scenario_id) == 2


def test_async_chat_spend_rows_record_the_empty_answer_and_the_refill_as_misses(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(emptied(CHAT, "choices")))
        model: Final = scenario.model(api_base=handle.api_base())
        key: Final = scenario.key(models=[model])

        async def drive() -> tuple[str, str, str]:
            async with _async_openai(gateway, key) as client:
                messages: Final = [{"role": "user", "content": text}]
                empty = await client.chat.completions.create(model=model, messages=messages)
                assert empty.choices == [], empty.model_dump_json()
                rescript(handle, json_response(CHAT))
                real = await client.chat.completions.create(model=model, messages=messages)
                assert real.choices[0].message.content == "scripted", real.model_dump_json()
                await_entry(store, real.id)
                served = await client.chat.completions.create(model=model, messages=messages)
                return empty.id, real.id, served.id

        empty_id, real_id, served_id = asyncio.run(drive())
        assert served_id == real_id != empty_id
        assert upstream.calls(handle.scenario_id) == 2
        rows: Final = eventually(lambda: _spend_rows(key), lambda found: len(found) == 3, seconds=70)
        by_id: Final = {string_value(row["request_id"]): row for row in rows}
        assert set(by_id) >= {empty_id, real_id}, sorted(by_id)
        assert by_id[empty_id]["cache_hit"] != "True" and by_id[real_id]["cache_hit"] != "True", rows
        hit_row: Final = next(row for request_id, row in by_id.items() if request_id.startswith(f"{real_id}_cache_hit"))
        assert hit_row["cache_hit"] == "True" and float(str(hit_row["spend"])) == 0, rows


def test_text_completion_empty_choices_reach_upstream_again(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(emptied(TEXT, "choices")))
        model: Final = scenario.model(model=TEXT_MODEL, api_base=handle.api_base())
        body: Final[dict[str, JsonValue]] = {"model": model, "prompt": prompt()}
        empty: Final = gateway.request("POST", "/v1/completions", body)
        assert choices(empty) == [], empty.text
        assert upstream.calls(handle.scenario_id) == 1
        rescript(handle, json_response(TEXT))
        real: Final = gateway.request("POST", "/v1/completions", body)
        assert text_of_first_choice(real) == "scripted" and response_id(real) != response_id(empty), real.text
        assert upstream.calls(handle.scenario_id) == 2
        await_entry(store, response_id(real))
        hit: Final = gateway.request("POST", "/v1/completions", body)
        assert response_id(hit) == response_id(real) and text_of_first_choice(hit) == "scripted", hit.text
        assert hit.headers.get("x-litellm-cache-key"), dict(hit.headers)
        assert upstream.calls(handle.scenario_id) == 2


def test_responses_empty_output_reaches_upstream_again(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(emptied(RESPONSE, "output")))
        model: Final = scenario.model(api_base=handle.api_base())

        async def drive() -> None:
            async with _async_openai(gateway) as client:
                empty = await client.responses.create(model=model, input=text)
                assert empty.output == [], empty.model_dump_json()
                assert upstream.calls(handle.scenario_id) == 1
                rescript(handle, json_response(RESPONSE))
                real = await client.responses.create(model=model, input=text)
                assert real.output_text == "scripted", real.model_dump_json()
                assert upstream.calls(handle.scenario_id) == 2
                await_entry(store, real.output[0].id)
                hit = await client.responses.with_raw_response.create(model=model, input=text)
                served = hit.parse()
                assert served.output[0].id == real.output[0].id and served.output_text == "scripted", hit.text
                assert hit.headers.get("x-litellm-cache-key"), dict(hit.headers)
                assert upstream.calls(handle.scenario_id) == 2

        asyncio.run(drive())


def test_messages_empty_content_reaches_upstream_again(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario, _anthropic(gateway) as client:
        handle: Final = scripted(scenario, json_response(emptied(MESSAGE, "content")))
        model: Final = scenario.model(model=MESSAGES_MODEL, api_base=handle.api_base())
        messages: Final = [{"role": "user", "content": text}]
        empty: Final = client.messages.create(model=model, max_tokens=16, messages=messages)
        assert empty.content == [], empty.model_dump_json()
        _assert_prompt_reached_upstream(upstream, handle.scenario_id, text, 1)
        rescript(handle, json_response(MESSAGE))
        real: Final = client.messages.create(model=model, max_tokens=16, messages=messages)
        assert real.id != empty.id and real.content[0].type == "text", real.model_dump_json()
        assert real.content[0].text == "scripted"
        _assert_prompt_reached_upstream(upstream, handle.scenario_id, text, 2)
        await_entry(store, real.id)
        served: Final = client.messages.create(model=model, max_tokens=16, messages=messages)
        assert served.id == real.id and served.content[0].type == "text", served.model_dump_json()
        assert served.content[0].text == "scripted"
        assert upstream.calls(handle.scenario_id) == 2


def test_stale_chat_entry_with_empty_choices_is_a_miss(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(CHAT))
        model: Final = scenario.model(api_base=handle.api_base())
        filled: Final = _chat(gateway, chat_body(model, text, cache=SHORT_TTL_CACHE_CONTROL))
        assert content_of_first_choice(filled) == "scripted", filled.text
        stale_id: Final = flip_entry_to_empty(store, await_entry(store, response_id(filled)), "choices")
        request: Final = chat_body(model, text)
        refilled: Final = eventually(
            lambda: _chat(gateway, request), lambda answer: response_id(answer) != response_id(filled), SETTLE_SECONDS
        )
        assert response_id(refilled) != stale_id, f"the stale entry was served: {refilled.text}"
        assert content_of_first_choice(refilled) == "scripted", refilled.text
        assert upstream.calls(handle.scenario_id) == 2
        await_entry(store, response_id(refilled))
        hit: Final = eventually(
            lambda: _chat(gateway, request), lambda answer: response_id(answer) == response_id(refilled), SETTLE_SECONDS
        )
        assert content_of_first_choice(hit) == "scripted", hit.text
        assert upstream.calls(handle.scenario_id) == 2


def test_stale_text_completion_entry_with_empty_choices_is_a_miss(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(TEXT))
        model: Final = scenario.model(model=TEXT_MODEL, api_base=handle.api_base())
        short_lived: Final[dict[str, JsonValue]] = {"model": model, "prompt": text, "cache": SHORT_TTL_CACHE_CONTROL}
        filled: Final = gateway.request("POST", "/v1/completions", short_lived)
        assert text_of_first_choice(filled) == "scripted", filled.text
        stale_id: Final = flip_entry_to_empty(store, await_entry(store, response_id(filled)), "choices")
        request: Final[dict[str, JsonValue]] = {"model": model, "prompt": text}
        refilled: Final = eventually(
            lambda: gateway.request("POST", "/v1/completions", request),
            lambda answer: response_id(answer) != response_id(filled),
            SETTLE_SECONDS,
        )
        assert response_id(refilled) != stale_id, f"the stale entry was served: {refilled.text}"
        assert text_of_first_choice(refilled) == "scripted", refilled.text
        assert upstream.calls(handle.scenario_id) == 2
        await_entry(store, response_id(refilled))
        hit: Final = eventually(
            lambda: gateway.request("POST", "/v1/completions", request),
            lambda answer: response_id(answer) == response_id(refilled),
            SETTLE_SECONDS,
        )
        assert text_of_first_choice(hit) == "scripted", hit.text
        assert upstream.calls(handle.scenario_id) == 2


def test_stale_responses_entry_with_empty_output_is_a_miss(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario, _openai(gateway) as client:
        handle: Final = scripted(scenario, json_response(RESPONSE))
        model: Final = scenario.model(api_base=handle.api_base())
        filled: Final = client.responses.create(model=model, input=text, extra_body={"cache": SHORT_TTL_CACHE_CONTROL})
        assert filled.output_text == "scripted", filled.model_dump_json()
        flip_entry_to_empty(store, await_entry(store, filled.output[0].id), "output")
        refilled: Final = eventually(
            lambda: client.responses.create(model=model, input=text),
            lambda answer: not answer.output or answer.output[0].id != filled.output[0].id,
            SETTLE_SECONDS,
        )
        assert refilled.output, f"the stale entry was served: {refilled.model_dump_json()}"
        assert refilled.output_text == "scripted", refilled.model_dump_json()
        assert upstream.calls(handle.scenario_id) == 2
        await_entry(store, refilled.output[0].id)
        hit: Final = eventually(
            lambda: client.responses.create(model=model, input=text),
            lambda answer: bool(answer.output) and answer.output[0].id == refilled.output[0].id,
            SETTLE_SECONDS,
        )
        assert hit.output_text == "scripted", hit.model_dump_json()
        assert upstream.calls(handle.scenario_id) == 2


def test_stale_messages_entry_with_empty_content_is_a_miss(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario, _anthropic(gateway) as client:
        handle: Final = scripted(scenario, json_response(MESSAGE))
        model: Final = scenario.model(model=MESSAGES_MODEL, api_base=handle.api_base())
        messages: Final = [{"role": "user", "content": text}]
        filled: Final = client.messages.create(
            model=model, max_tokens=16, messages=messages, extra_body={"cache": SHORT_TTL_CACHE_CONTROL}
        )
        assert filled.content[0].type == "text" and filled.content[0].text == "scripted", filled.model_dump_json()
        stale_id: Final = flip_entry_to_empty(store, await_entry(store, filled.id), "content")
        refilled: Final = eventually(
            lambda: client.messages.create(model=model, max_tokens=16, messages=messages),
            lambda answer: answer.id != filled.id,
            SETTLE_SECONDS,
        )
        assert refilled.id != stale_id, f"the stale entry was served: {refilled.model_dump_json()}"
        assert refilled.content[0].type == "text" and refilled.content[0].text == "scripted", refilled.model_dump_json()
        assert upstream.calls(handle.scenario_id) == 2
        await_entry(store, refilled.id)
        hit: Final = eventually(
            lambda: client.messages.create(model=model, max_tokens=16, messages=messages),
            lambda answer: answer.id == refilled.id,
            SETTLE_SECONDS,
        )
        assert hit.content[0].type == "text" and hit.content[0].text == "scripted", hit.model_dump_json()
        assert upstream.calls(handle.scenario_id) == 2


def _streamed_chat(gateway: Gateway, model: str, text: str) -> _Answer:
    """The chunk ids, the streamed text and the chunk kinds, after the whole stream was read."""
    with (
        _openai(gateway) as client,
        client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": text}], stream=True
        ) as stream,
    ):
        chunks: Final = tuple(stream)
    ids: Final = frozenset(chunk.id for chunk in chunks)
    streamed: Final = "".join(
        choice.delta.content or "" for chunk in chunks for choice in chunk.choices
    )  # comprehension-ok: flattens chunk choices
    return ids, streamed, tuple(chunk.object for chunk in chunks)


def test_streamed_chat_without_choices_is_never_assembled_or_cached(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, CHAT_STREAM_WITHOUT_CHOICES)
        model: Final = scenario.model(api_base=handle.api_base())
        empty: Final = _never_replayed(partial(_streamed_chat, gateway, model, text), upstream, handle.scenario_id)
        assert all(answer.text == "" for answer in empty), empty


def _streamed_text(gateway: Gateway, model: str, text: str) -> _Answer:
    """The chunk ids, the streamed text and the chunk kinds of a text completion, after the whole stream was read."""
    with _openai(gateway) as client, client.completions.create(model=model, prompt=text, stream=True) as stream:
        chunks: Final = tuple(stream)
    ids: Final = frozenset(chunk.id for chunk in chunks)
    streamed: Final = "".join(
        choice.text or "" for chunk in chunks for choice in chunk.choices
    )  # comprehension-ok: flattens chunk choices
    return ids, streamed, tuple(chunk.object for chunk in chunks)


def test_streamed_text_completion_without_choices_is_never_assembled_or_cached(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, TEXT_STREAM_WITHOUT_CHOICES)
        model: Final = scenario.model(model=TEXT_MODEL, api_base=handle.api_base())
        empty: Final = _never_replayed(partial(_streamed_text, gateway, model, text), upstream, handle.scenario_id)
        assert all(answer.text == "" for answer in empty), empty


def test_streamed_chat_with_content_is_replayed_from_the_cache(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, CHAT_STREAM)
        model: Final = scenario.model(api_base=handle.api_base())
        stream: Final = partial(_streamed_chat, gateway, model, text)
        first_ids, first_text, _ = stream()
        assert first_text == "streamed response" and len(first_ids) == 1, first_ids
        assert upstream.calls(handle.scenario_id) == 1
        await_entry(store, next(iter(first_ids)))
        replay: Final = _replayed(stream, upstream, handle.scenario_id, first_ids)
        assert replay.text == "streamed response", replay


def _streamed_response(gateway: Gateway, model: str, text: str) -> _Answer:
    """The completed response's output item ids, its text and the event kinds, after the whole stream was read."""

    async def read() -> _Answer:
        async with (
            _async_openai(gateway) as client,
            await client.responses.create(model=model, input=text, stream=True) as stream,
        ):
            events: Final = [event async for event in stream]
        completed: Final = tuple(event for event in events if event.type == "response.completed")
        assert len(completed) == 1, [event.type for event in events]
        ids: Final = frozenset(item.id for item in completed[0].response.output)
        return ids, completed[0].response.output_text, tuple(event.type for event in events)

    return asyncio.run(read())


def test_streamed_responses_completing_without_output_reach_upstream_again(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, RESPONSE_STREAM_WITHOUT_OUTPUT)
        model: Final = scenario.model(api_base=handle.api_base())
        stream: Final = partial(_streamed_response, gateway, model, text)
        empty: Final = _never_replayed(stream, upstream, handle.scenario_id)
        assert all(answer.ids == frozenset() and answer.text == "" for answer in empty), empty
        rescript(handle, RESPONSE_STREAM)
        real_ids, real_text, _ = stream()
        assert real_text == "streamed response", real_ids
        assert upstream.calls(handle.scenario_id) == EMPTY_STREAMS + 1, "the empty stream was replayed"
        await_entry(store, next(iter(real_ids)))
        replay: Final = _replayed(stream, upstream, handle.scenario_id, real_ids)
        assert replay.text == "streamed response", replay


def test_streamed_responses_with_output_are_replayed_from_the_cache(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, RESPONSE_STREAM)
        model: Final = scenario.model(api_base=handle.api_base())
        stream: Final = partial(_streamed_response, gateway, model, text)
        first_ids, first_text, _ = stream()
        assert first_text == "streamed response" and len(first_ids) == 1, first_ids
        assert all(item.startswith(f"msg_{handle.scenario_id}-") for item in first_ids), first_ids
        assert upstream.calls(handle.scenario_id) == 1
        await_entry(store, next(iter(first_ids)))
        replay: Final = _replayed(stream, upstream, handle.scenario_id, first_ids)
        assert replay.text == "streamed response", replay


def _streamed_message(gateway: Gateway, model: str, text: str) -> _Answer:
    """The message id, the streamed text and the event kinds, after the whole stream was read."""

    async def read() -> _Answer:
        async with _async_anthropic(gateway) as client:
            stream: Final = await client.messages.create(
                model=model, max_tokens=16, messages=[{"role": "user", "content": text}], stream=True
            )
            events: Final = [event async for event in stream]
        starts: Final = tuple(event for event in events if event.type == "message_start")
        assert len(starts) == 1, [event.type for event in events]
        deltas: Final = tuple(event for event in events if event.type == "content_block_delta")
        streamed: Final = "".join(delta.delta.text for delta in deltas if delta.delta.type == "text_delta")
        return frozenset({starts[0].message.id}), streamed, tuple(event.type for event in events)

    return asyncio.run(read())


def test_streamed_messages_without_content_blocks_reach_upstream_again(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, MESSAGE_STREAM_WITHOUT_CONTENT_BLOCKS)
        model: Final = scenario.model(model=MESSAGES_MODEL, api_base=handle.api_base())
        stream: Final = partial(_streamed_message, gateway, model, text)
        empty: Final = _never_replayed(stream, upstream, handle.scenario_id)
        assert all("content_block_start" not in answer.kinds and answer.text == "" for answer in empty), empty
        rescript(handle, MESSAGE_STREAM)
        real_ids, real_text, real_kinds = stream()
        assert real_text == "streamed response" and "content_block_start" in real_kinds, real_kinds
        assert upstream.calls(handle.scenario_id) == EMPTY_STREAMS + 1, "the stream without content blocks was replayed"
        await_entry(store, next(iter(real_ids)))
        replay: Final = _replayed(stream, upstream, handle.scenario_id, real_ids)
        assert replay.text == "streamed response" and replay.kinds == real_kinds, replay


def test_streamed_messages_with_content_are_replayed_from_the_cache(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, MESSAGE_STREAM)
        model: Final = scenario.model(model=MESSAGES_MODEL, api_base=handle.api_base())
        stream: Final = partial(_streamed_message, gateway, model, text)
        first_ids, first_text, first_kinds = stream()
        assert first_text == "streamed response" and "content_block_start" in first_kinds, first_kinds
        assert upstream.calls(handle.scenario_id) == 1
        await_entry(store, next(iter(first_ids)))
        replay: Final = _replayed(stream, upstream, handle.scenario_id, first_ids)
        assert replay.text == "streamed response" and replay.kinds == first_kinds, replay


def test_embeddings_are_still_served_from_the_cache(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(EMBEDDING))
        model: Final = scenario.model(model="openai/text-embedding-3-small", api_base=handle.api_base())
        body: Final[dict[str, JsonValue]] = {"model": model, "input": [prompt()]}
        first: Final = gateway.post("/v1/embeddings", body)
        assert first["object"] == "list" and first["data"], first
        assert upstream.calls(handle.scenario_id) == 1
        second: Final = eventually(
            lambda: _served_without_upstream(
                upstream, handle.scenario_id, lambda: gateway.post("/v1/embeddings", body)
            ),
            lambda answer: answer is not None,
            SETTLE_SECONDS,
        )
        assert second is not None and second["data"] == first["data"], second


@pytest.mark.parametrize(
    "choice",
    (
        pytest.param(
            {"index": 0, "message": {"role": "assistant", "content": ""}, "finish_reason": "content_filter"},
            id="blocked-empty-content",
        ),
        pytest.param(
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
                    ],
                },
                "finish_reason": "tool_calls",
            },
            id="tool-calls-only",
        ),
    ),
)
def test_chat_answers_with_a_choice_but_no_text_are_still_cached(
    gateway: Gateway, choice: dict[str, JsonValue]
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response({**CHAT, "choices": [choice]}))
        model: Final = scenario.model(api_base=handle.api_base())
        request: Final = chat_body(model, prompt())
        first: Final = _chat(gateway, request)
        assert object_value(choices(first)[0])["finish_reason"] == choice["finish_reason"], first.text
        assert upstream.calls(handle.scenario_id) == 1
        await_entry(store, response_id(first))
        second: Final = _chat(gateway, request)
        assert response_id(second) == response_id(first) and choices(second) == choices(first), second.text
        assert upstream.calls(handle.scenario_id) == 1


def test_no_cache_requests_reach_upstream_every_time(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(CHAT))
        model: Final = scenario.model(api_base=handle.api_base())
        request: Final = chat_body(model, prompt(), cache={"no-cache": True})
        ids: Final = tuple(response_id(_chat(gateway, request)) for _ in range(3))
        assert len(set(ids)) == 3, ids
        assert upstream.calls(handle.scenario_id) == 3


@pytest.mark.parametrize("status", (500, 429))
def test_upstream_errors_reach_the_caller_and_are_never_cached(gateway: Gateway, status: int) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    with gateway.scenario() as scenario:
        handle: Final = scripted(
            scenario, json_response({"error": {"message": "Controlled provider failure", "type": "api_error"}}, status)
        )
        model: Final = scenario.model(api_base=handle.api_base())
        request: Final = chat_body(model, prompt())
        first: Final = _chat(gateway, request)
        assert first.status_code == status and "Controlled provider failure" in first.text, first.text
        after_first: Final = upstream.calls(handle.scenario_id)
        assert after_first >= 1
        second: Final = _chat(gateway, request)
        assert second.status_code == status and "Controlled provider failure" in second.text, second.text
        assert upstream.calls(handle.scenario_id) > after_first, "the error was served from the cache"


def test_cache_delete_on_the_entry_key_makes_the_next_request_reach_upstream(gateway: Gateway) -> None:
    """``/cache/delete`` clears the Redis entry; a worker's in-memory copy of it serves until the entry's ttl lapses."""
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    text: Final = prompt()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(CHAT))
        model: Final = scenario.model(api_base=handle.api_base())
        filled: Final = _chat(gateway, chat_body(model, text, cache=SHORT_TTL_CACHE_CONTROL))
        key: Final = await_entry(store, response_id(filled))
        assert gateway.post("/cache/delete", {"keys": [key]}) == {"status": "success"}
        assert store.get(key) is None
        request: Final = chat_body(model, text)
        refilled: Final = eventually(
            lambda: _chat(gateway, request), lambda answer: response_id(answer) != response_id(filled), SETTLE_SECONDS
        )
        assert content_of_first_choice(refilled) == "scripted", refilled.text
        assert upstream.calls(handle.scenario_id) == 2
        await_entry(store, response_id(refilled))
        hit: Final = eventually(
            lambda: _chat(gateway, request), lambda answer: response_id(answer) == response_id(refilled), SETTLE_SECONDS
        )
        assert hit.headers.get("x-litellm-cache-key") == key, dict(hit.headers)
        assert upstream.calls(handle.scenario_id) == 2


def _shape(body: Mapping[str, JsonValue], field: str, value: object) -> dict[str, JsonValue]:
    if value is MISSING:
        return {name: item for name, item in body.items() if name != field}
    assert isinstance(value, (str, int)) or value is None
    return {**body, field: value}


def _request_for(path: str, model: str, text: str) -> dict[str, JsonValue]:
    if path == "/v1/responses":
        return {"model": model, "input": text}
    if path == "/v1/messages":
        return message_body(model, text)
    return chat_body(model, text)


@pytest.mark.parametrize(
    ("path", "body", "litellm_model", "field", "value"),
    (
        pytest.param("/v1/chat/completions", CHAT, CHAT_MODEL, "choices", None, id="chat-choices-null"),
        pytest.param("/v1/chat/completions", CHAT, CHAT_MODEL, "choices", MISSING, id="chat-choices-missing"),
        pytest.param("/v1/chat/completions", CHAT, CHAT_MODEL, "choices", "x", id="chat-choices-string"),
        pytest.param("/v1/chat/completions", CHAT, CHAT_MODEL, "choices", 5, id="chat-choices-int"),
    ),
)
def test_malformed_upstream_answers_are_an_error_and_never_cached(
    gateway: Gateway, path: str, body: Mapping[str, JsonValue], litellm_model: str, field: str, value: object
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(_shape(body, field, value)))
        control: Final = scripted(scenario, json_response(CHAT))
        model: Final = scenario.model(model=litellm_model, api_base=handle.api_base())
        control_model: Final = scenario.model(api_base=control.api_base())
        key: Final = scenario.key(models=[model, control_model])
        request: Final = _request_for(path, model, prompt())
        first: Final = gateway.request("POST", path, request, key=key)
        assert first.status_code == 500, first.text
        assert "error" in first.text.lower(), first.text
        assert upstream.calls(handle.scenario_id) == 1
        second: Final = gateway.request("POST", path, request, key=key)
        assert second.status_code == 500, second.text
        assert upstream.calls(handle.scenario_id) == 2, "the malformed answer was served from the cache"
        unrelated: Final = _chat(gateway, chat_body(control_model, prompt()), key=key)
        assert content_of_first_choice(unrelated) == "scripted", unrelated.text


@pytest.mark.parametrize(
    ("path", "body", "litellm_model", "field", "value"),
    (
        pytest.param("/v1/responses", RESPONSE, CHAT_MODEL, "output", None, id="responses-output-null"),
        pytest.param("/v1/responses", RESPONSE, CHAT_MODEL, "output", MISSING, id="responses-output-missing"),
        pytest.param("/v1/messages", MESSAGE, MESSAGES_MODEL, "content", None, id="messages-content-null"),
        pytest.param("/v1/messages", MESSAGE, MESSAGES_MODEL, "content", MISSING, id="messages-content-missing"),
    ),
)
def test_answers_without_the_output_field_reach_upstream_again(
    gateway: Gateway, path: str, body: Mapping[str, JsonValue], litellm_model: str, field: str, value: object
) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(_shape(body, field, value)))
        model: Final = scenario.model(model=litellm_model, api_base=handle.api_base())
        request: Final = _request_for(path, model, prompt())
        first: Final = gateway.request("POST", path, request)
        assert first.status_code == 200, first.text
        assert upstream.calls(handle.scenario_id) == 1
        second: Final = gateway.request("POST", path, request)
        assert second.status_code == 200, second.text
        assert upstream.calls(handle.scenario_id) == 2, "the answer without output was served from the cache"


def test_per_request_ttl_on_an_empty_answer_does_not_pin_it(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(emptied(CHAT, "choices")))
        model: Final = scenario.model(api_base=handle.api_base())
        request: Final = chat_body(model, prompt(), cache={"ttl": 30})
        empty: Final = _chat(gateway, request)
        assert choices(empty) == [], empty.text
        assert upstream.calls(handle.scenario_id) == 1
        rescript(handle, json_response(CHAT))
        real: Final = _chat(gateway, request)
        assert content_of_first_choice(real) == "scripted", real.text
        assert upstream.calls(handle.scenario_id) == 2, "the empty answer was pinned for the request's ttl"
        key: Final = await_entry(store, response_id(real))
        assert 0 < store.ttl(key) <= 30, store.ttl(key)


def test_rerank_results_are_still_served_from_the_cache(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(RERANK))
        model: Final = scenario.model(model="cohere/rerank-v4.0", api_base=handle.api_base())
        body: Final[dict[str, JsonValue]] = {"model": model, "query": prompt(), "documents": ["first", "second"]}
        first: Final = gateway.post("/v1/rerank", body)
        assert string_value(first["id"]).startswith(f"rerank-{handle.scenario_id}"), first
        assert upstream.calls(handle.scenario_id) == 1
        await_entry(store, string_value(first["id"]))
        second: Final = gateway.post("/v1/rerank", body)
        assert second["id"] == first["id"] and second["results"] == first["results"], second
        assert upstream.calls(handle.scenario_id) == 1


def test_transcriptions_are_still_served_from_the_cache(gateway: Gateway) -> None:
    upstream: Final = Upstream(gateway.upstream_url)
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        handle: Final = scripted(scenario, json_response(TRANSCRIPT))
        model: Final = scenario.model(model="openai/gpt-4o-mini-transcribe", api_base=handle.api_base())
        audio: Final = (f"{uuid.uuid4().hex}.wav", b"RIFF" + uuid.uuid4().bytes, "audio/wav")
        first: Final = gateway.request_multipart("/v1/audio/transcriptions", {"model": model}, {"file": audio})
        assert first.status_code == 200, first.text
        transcript: Final = string_value(JSON_OBJECT.validate_json(first.content)["text"])
        assert transcript.startswith(f"scripted {handle.scenario_id}-"), first.text
        assert upstream.calls(handle.scenario_id) == 1
        await_entry(store, transcript)
        second: Final = gateway.request_multipart("/v1/audio/transcriptions", {"model": model}, {"file": audio})
        assert second.status_code == 200 and JSON_OBJECT.validate_json(second.content)["text"] == transcript, (
            second.text
        )
        assert upstream.calls(handle.scenario_id) == 1
