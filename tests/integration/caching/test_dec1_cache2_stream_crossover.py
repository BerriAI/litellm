"""Streamed and non-streamed requests never share a response cache entry, on every route.

``stream`` is an API parameter, so ``stream: true`` is hashed into the key (``Cache.get_cache_key``) and a
stream and a non-stream request for the same prompt get different entries. A stored stream is replayed only
to a stream request and a stored JSON answer only to a JSON request. The chat cells are in
``test_dec1_cache_key_fields.py`` and ``test_dec1_cache_streams_and_embedding_lists.py``; this file covers
``/v1/messages`` (Rust: ``AnthropicMessagesStreamCacheWriter`` and ``CachedAnthropicMessagesStreamIterator`` on the
Python side, ``capture_stream`` in ``litellm/rust/crates/inference/src/caching.rs`` on Rust), ``/v1/responses``
and ``/v1/completions``, and the accounting of a replayed Messages stream.

A stream can reach the upstream once per worker before it is replayed, so stream cells repeat the request
until it is served without an upstream call.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually, list_value, object_value, string_value
from tests.integration.caching.dec1_cache2_support import PRICED
from tests.integration.caching.dec1_cache_support import (
    CACHE_KEY_HEADER,
    Exchange,
    await_named_entry,
    await_spend_rows,
    filled,
    probe,
    sse_event_names,
)
from tests.integration.caching.response_cache_case import (
    CHAT_STREAM,
    MESSAGE,
    MESSAGE_STREAM,
    MESSAGES_MODEL,
    RESPONSE,
    RESPONSE_STREAM,
    TEXT,
    TEXT_MODEL,
    TEXT_STREAM,
    json_response,
    prompt,
    redis_store,
    rescript,
)

REPLAY_SECONDS: Final = 30


def _replayed(ask: Callable[[], Exchange]) -> Exchange:
    """Repeat ``ask`` until an answer is served without an upstream call."""
    replay: Final = eventually(ask, lambda exchange: not exchange.reached_upstream, REPLAY_SECONDS)
    assert not replay.reached_upstream, replay.response.text
    return replay


def test_a_non_stream_messages_entry_is_not_replayed_to_a_messages_stream(gateway: Gateway) -> None:
    """A JSON ``/v1/messages`` answer is cached. The upstream is re-scripted to stream, and the same request with
    ``stream: true`` is sent. It must reach the upstream and answer with SSE events ending in ``message_stop``,
    never a stream synthesised from the stored JSON message.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(MESSAGE), model=MESSAGES_MODEL)
        text: Final = prompt()
        filled(redis_store(), target.messages(text))
        rescript(target.handle, MESSAGE_STREAM)
        streamed: Final = target.messages(text, stream=True)
        assert streamed.reached_upstream, streamed.response.text
        assert sse_event_names(streamed.response)[-1] == "message_stop", streamed.response.text


def test_a_streamed_messages_entry_is_not_served_to_a_non_stream_request(gateway: Gateway) -> None:
    """A streamed ``/v1/messages`` request is repeated until it is replayed from the cache. The upstream is then
    re-scripted to answer JSON, and the same request without ``stream`` must reach it and return a JSON message.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, MESSAGE_STREAM, model=MESSAGES_MODEL)
        text: Final = prompt()
        _replayed(lambda: target.messages(text, stream=True))
        rescript(target.handle, json_response(MESSAGE))
        plain: Final = target.messages(text)
        assert plain.reached_upstream, plain.response.text
        assert plain.json()["type"] == "message"


def test_a_replayed_messages_stream_carries_the_cache_key_header(gateway: Gateway) -> None:
    """A streamed ``/v1/messages`` request is repeated until replayed. The replay's response headers must carry
    ``x-litellm-cache-key``, as a non-stream hit does, so a caller can tell a replay apart on the Rust path.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, MESSAGE_STREAM, model=MESSAGES_MODEL)
        text: Final = prompt()
        replay: Final = _replayed(lambda: target.messages(text, stream=True))
        assert replay.response.headers.get(CACHE_KEY_HEADER), dict(replay.response.headers)


def test_a_replayed_messages_stream_writes_one_zero_spend_hit_row(gateway: Gateway) -> None:
    """A key on a priced Anthropic model streams the same ``/v1/messages`` request until one stream is replayed.
    There must be exactly one spend row per request sent, exactly one with ``cache_hit = 'True'``, and that row's
    ``spend`` must be 0 and its ``call_type`` ``anthropic_messages`` (the row keeps the sync name, pinned
    current): the replay logs once, when it ends.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, MESSAGE_STREAM, model=MESSAGES_MODEL, **PRICED)
        key: Final = scenario.key(models=[target.model])
        text: Final = prompt()
        sent: list[Exchange] = []  # mutable-ok: every request until the first replay
        for _ in range(8):
            exchange = target.messages(text, key=key, stream=True)
            sent.append(exchange)
            if not exchange.reached_upstream:
                break
        assert not sent[-1].reached_upstream, "no stream was replayed in 8 attempts"
        rows: Final = await_spend_rows(gateway, key, len(sent))
        hits: Final = [row for row in rows if row["cache_hit"] == "True"]
        assert len(hits) == 1, rows
        assert float(str(hits[0]["spend"])) == 0 and hits[0]["call_type"] == "anthropic_messages", hits


def test_a_non_stream_responses_entry_is_not_replayed_to_a_responses_stream(gateway: Gateway) -> None:
    """A JSON ``/v1/responses`` answer is cached. With the upstream re-scripted to stream, the same request with
    ``stream: true`` must reach the upstream and answer SSE ending in ``response.completed``.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(RESPONSE))
        text: Final = prompt()
        first: Final = target.responses(text)
        assert first.reached_upstream, first.response.text
        output_id: Final = string_value(object_value(list_value(first.json()["output"])[0])["id"])
        await_named_entry(redis_store(), output_id)
        rescript(target.handle, RESPONSE_STREAM)
        streamed: Final = target.responses(text, stream=True)
        assert streamed.reached_upstream, streamed.response.text
        assert sse_event_names(streamed.response)[-1] == "response.completed", streamed.response.text


def test_a_non_stream_text_completion_entry_is_not_replayed_to_a_stream(gateway: Gateway) -> None:
    """A JSON ``/v1/completions`` answer is cached. With the upstream re-scripted to stream, the same request with
    ``stream: true`` must reach the upstream.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, json_response(TEXT), model=TEXT_MODEL)
        text: Final = prompt()
        filled(redis_store(), target.completion(text))
        rescript(target.handle, TEXT_STREAM)
        streamed: Final = target.completion(text, stream=True)
        assert streamed.reached_upstream, streamed.response.text


def test_stream_false_keys_like_an_omitted_stream(gateway: Gateway) -> None:
    """A chat entry is filled by a request without ``stream``. The identical request with ``"stream": false`` must
    be a hit: both ask for one JSON answer. Today ``stream: False`` is a non-``None`` value and is hashed as
    ``stream: False`` (``caching.py:438``), unlike an explicit ``null``, so the two miss each other.
    """
    pytest.skip("BUG: an explicit stream false changes the cache key of an otherwise identical request")
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(redis_store(), first)
        second: Final = target.chat(text, stream=False)
        assert not second.reached_upstream and second.id() == first.id(), second.response.text


def test_streams_with_and_without_include_usage_are_separate_entries(gateway: Gateway) -> None:
    """A chat stream with ``stream_options: {"include_usage": true}`` is repeated until replayed. The same stream
    without ``stream_options`` must then reach the upstream: ``stream_options`` is an OpenAI chat parameter and is
    hashed. UNVERIFIED: if the proxy injects ``include_usage`` into every stream itself the two keys coincide and
    this cell shows it.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, CHAT_STREAM)
        text: Final = prompt()
        _replayed(lambda: target.chat(text, stream=True, stream_options={"include_usage": True}))
        assert target.chat(text, stream=True).reached_upstream
