"""Streaming requests filling and reading the cache, and per-input caching of embedding lists.

Streams: a streamed answer is assembled into a complete response when the stream ends and stored under
the streamed request's own key (``stream`` is part of the key). An identical streamed request is then
replayed from the entry as a stream, without an upstream call. A stream can reach the upstream once per
proxy worker before a replay is seen, so the stream cells repeat the request until one is served
without an upstream call and assert on that replay.

Embedding lists: ``/v1/embeddings`` caches every input of a list under its own key
(``LLMCachingHandler._retrieve_from_cache`` builds one key per element). A later list is answered from
the cached elements, only the missing elements are sent to the provider, and the answer is stitched
back in request order with summed usage. The provider here is a local OpenAI-compatible peer that
returns a deterministic vector per input and reports 3 prompt tokens per input.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable, Mapping
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually, object_value, string_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.caching.dec1_cache_support import (
    CACHE_KEY_HEADER,
    EMBEDDING_TOKENS_PER_INPUT,
    SETTLE_SECONDS,
    Exchange,
    Probe,
    chat_stream_text,
    embedding_inputs,
    embedding_model,
    embedding_vector,
    openai_embedding_peer,
    probe,
    sse_event_names,
    sse_payloads,
    vectors,
)
from tests.integration.caching.response_cache_case import (
    CHAT,
    CHAT_STREAM,
    MESSAGE_STREAM,
    MESSAGES_MODEL,
    RESPONSE_STREAM,
    TEXT_MODEL,
    TEXT_STREAM,
    json_response,
    message_body,
    prompt,
    rescript,
)

STREAM_ATTEMPTS: Final = 8


def _replay(send: Callable[[], Exchange]) -> tuple[Exchange, Exchange]:
    """Repeat a streamed request until one answer is served without an upstream call; returns the first
    upstream-served stream and the replay."""
    first: Final = send()
    assert first.reached_upstream and first.response.status_code == 200, first.response.text
    for _ in range(STREAM_ATTEMPTS):
        answer = send()
        assert answer.response.status_code == 200, answer.response.text
        if not answer.reached_upstream:
            return first, answer
    raise AssertionError(f"no stream was replayed from the cache in {STREAM_ATTEMPTS} attempts")


def _chat_replay(target: Probe, text: str, **extra: JsonValue) -> tuple[Exchange, Exchange]:
    return _replay(lambda: target.chat(text, stream=True, **extra))


def test_a_replayed_chat_stream_is_server_sent_events_ending_with_done(gateway: Gateway) -> None:
    """A streamed chat request fills the cache and is repeated until replayed. The replay must be answered
    200 with a ``text/event-stream`` content type, at least one ``chat.completion.chunk`` and a final
    ``data: [DONE]`` line, exactly like a live stream.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, CHAT_STREAM)
        _, replay = _chat_replay(target, prompt())
        assert replay.response.headers["content-type"].startswith("text/event-stream"), dict(replay.response.headers)
        payloads: Final = sse_payloads(replay.response)
        assert payloads and all(payload["object"] == "chat.completion.chunk" for payload in payloads), payloads
        assert replay.response.text.rstrip().endswith("data: [DONE]"), replay.response.text[-200:]


def test_a_replayed_chat_stream_carries_the_original_text_and_a_stop_finish_reason(gateway: Gateway) -> None:
    """The upstream streams "streamed " and "response" and finishes with ``stop``. The replay's content deltas
    must join to exactly ``streamed response`` and one chunk must carry ``finish_reason: "stop"``.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, CHAT_STREAM)
        first, replay = _chat_replay(target, prompt())
        assert chat_stream_text(sse_payloads(first.response)) == "streamed response"
        payloads: Final = sse_payloads(replay.response)
        assert chat_stream_text(payloads) == "streamed response", replay.response.text
        reasons: Final = [
            object_value(choice).get("finish_reason")
            for payload in payloads
            for choice in payload.get("choices") or []  # type: ignore[union-attr]
        ]
        assert "stop" in reasons, reasons


def test_a_replayed_chat_stream_repeats_the_id_of_a_stream_that_reached_the_upstream(gateway: Gateway) -> None:
    """Every chunk of the replay must carry one id, and that id must be the id of one of the earlier streams
    that reached the upstream: the replay is the stored answer, not a new completion.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, CHAT_STREAM)
        text: Final = prompt()
        seen: set[str] = set()  # mutable-ok: ids of upstream-served streams, gathered until the first replay
        replay_ids: frozenset[str] = frozenset()
        for _ in range(STREAM_ATTEMPTS):
            exchange = target.chat(text, stream=True)
            ids = frozenset(string_value(payload["id"]) for payload in sse_payloads(exchange.response))
            if not exchange.reached_upstream:
                replay_ids = ids
                break
            seen.update(ids)
        assert len(replay_ids) == 1 and replay_ids <= seen, (replay_ids, seen)


def test_a_replayed_chat_stream_carries_the_cache_key_header(gateway: Gateway) -> None:
    """The replayed stream's response headers must include ``x-litellm-cache-key`` while the stream that
    reached the upstream must not, the same hit signal as for non-streamed answers.
    UNVERIFIED: the header is built from the stream wrapper's hidden params, set by
    ``_set_cached_hidden_param``; confirm it survives on the streaming response path in
    ``common_request_processing.py``.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, CHAT_STREAM)
        first, replay = _chat_replay(target, prompt())
        assert CACHE_KEY_HEADER not in first.response.headers, dict(first.response.headers)
        assert replay.response.headers.get(CACHE_KEY_HEADER), dict(replay.response.headers)


def test_no_cache_on_a_stream_reaches_the_upstream_although_a_replay_exists(gateway: Gateway) -> None:
    """A streamed request has been replayed from the cache once. The same streamed request with
    ``cache: {"no-cache": true}`` must reach the upstream again.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, CHAT_STREAM)
        text: Final = prompt()
        _chat_replay(target, text)
        forced: Final = target.chat(text, stream=True, cache={"no-cache": True})
        assert forced.reached_upstream, forced.response.text
        assert chat_stream_text(sse_payloads(forced.response)) == "streamed response"


def test_a_stream_entry_is_not_served_to_a_non_stream_request(gateway: Gateway) -> None:
    """A streamed chat request has filled the cache and been replayed. The same request without ``stream``
    must reach the upstream (re-scripted to answer JSON) and return a ``chat.completion`` object with a new
    id, because the streamed and non-streamed forms have different keys.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, CHAT_STREAM)
        text: Final = prompt()
        _, replay = _chat_replay(target, text)
        rescript(target.handle, json_response(CHAT))
        plain: Final = target.chat(text)
        assert plain.reached_upstream, plain.response.text
        assert plain.json()["object"] == "chat.completion", plain.response.text
        assert plain.id() not in {string_value(payload["id"]) for payload in sse_payloads(replay.response)}


def test_a_text_completion_stream_is_replayed_with_the_original_text(gateway: Gateway) -> None:
    """A streamed ``/v1/completions`` request fills the cache and is repeated until replayed. The replay must
    be SSE whose ``choices[0].text`` pieces join to ``streamed response``.
    BUG: a replayed ``/v1/completions`` stream comes back as ``chat.completion.chunk`` frames with the text in
    ``choices[].delta.content`` and no ``choices[].text``: the cached text completion is rebuilt with the chat
    streaming converter (``litellm/caching/caching_handler.py:1192``), so a text completion client reads an
    empty answer.
    """
    pytest.skip("BUG: a cached /v1/completions stream is replayed as chat.completion.chunk frames without text")
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, TEXT_STREAM, model=TEXT_MODEL)
        text: Final = prompt()
        _, replay = _replay(lambda: target.completion(text, stream=True))
        pieces: Final = [
            str(object_value(choice).get("text") or "")
            for payload in sse_payloads(replay.response)
            for choice in payload.get("choices") or []  # type: ignore[union-attr]
        ]
        assert "".join(pieces) == "streamed response", replay.response.text


def test_a_replayed_responses_stream_runs_from_created_to_completed_with_the_original_text(gateway: Gateway) -> None:
    """A streamed ``/v1/responses`` request fills the cache and is repeated until replayed. The replay's first
    event must be ``response.created``, its last ``response.completed``, and the completed response's output
    text must be ``streamed response``.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, RESPONSE_STREAM)
        text: Final = prompt()
        _, replay = _replay(lambda: target.responses(text, stream=True))
        types: Final = [string_value(payload["type"]) for payload in sse_payloads(replay.response)]
        assert types[0] == "response.created" and types[-1] == "response.completed", types
        completed: Final = object_value(sse_payloads(replay.response)[-1]["response"])
        assert "streamed response" in json.dumps(completed["output"]), completed


def test_a_replayed_messages_stream_repeats_the_event_sequence_of_the_original(gateway: Gateway) -> None:
    """A streamed Anthropic ``/v1/messages`` request fills the cache and is repeated until replayed. The
    replay's sequence of SSE event names must equal the sequence the first, upstream-served stream had
    (``message_start`` ... ``message_stop``), and its text delta must be ``streamed response``.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, MESSAGE_STREAM, model=MESSAGES_MODEL)
        body: Final = message_body(target.model, prompt(), stream=True)
        first, replay = _replay(lambda: target.send("/v1/messages", body))
        names: Final = sse_event_names(replay.response)
        assert names == sse_event_names(first.response), (names, sse_event_names(first.response))
        assert names[0] == "message_start" and names[-1] == "message_stop", names
        assert '"text":"streamed response"' in replay.response.text.replace('": "', '":"'), replay.response.text


def _served_without_upstream(
    gateway: Gateway, wire: Wire, body: Mapping[str, JsonValue]
) -> dict[str, JsonValue] | None:
    answer: Final = gateway.post("/v1/embeddings", body)
    return None if wire.drain() else answer


def _await_full_hit(gateway: Gateway, wire: Wire, body: Mapping[str, JsonValue]) -> dict[str, JsonValue]:
    """Repeat ``body`` until it is answered without any request reaching the peer (the writes are async)."""
    served: Final = eventually(
        lambda: _served_without_upstream(gateway, wire, body), lambda answer: answer is not None, SETTLE_SECONDS
    )
    assert served is not None
    return served


def _sent_inputs(wire: Wire) -> list[list[str]]:
    return [embedding_inputs(request) for request in wire.drain()]


def _texts(count: int) -> list[str]:
    return [f"dec1 embedding {index} {uuid.uuid4().hex}" for index in range(count)]


def test_an_embedding_list_is_cached_and_served_whole(gateway: Gateway) -> None:
    """A list of two inputs is embedded. The identical list must then be answered without any request
    reaching the provider, with the same two vectors at indexes 0 and 1.
    """
    with wire_server(openai_embedding_peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire.url)
        first, second = _texts(2)
        body: Final[dict[str, JsonValue]] = {"model": model, "input": [first, second]}
        filling: Final = gateway.post("/v1/embeddings", body)
        assert _sent_inputs(wire) == [[first, second]]
        served: Final = _await_full_hit(gateway, wire, body)
        assert vectors(served) == vectors(filling) == [embedding_vector(first), embedding_vector(second)]


def test_only_the_uncached_inputs_of_a_list_are_sent_to_the_provider(gateway: Gateway) -> None:
    """Inputs A and B are cached. A request for the list [A, C] must send only C to the provider (one upstream
    request whose ``input`` is ``[C]``) and return A's stored vector at index 0 and C's fresh vector at index 1.
    """
    with wire_server(openai_embedding_peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire.url)
        a, b, c = _texts(3)
        gateway.post("/v1/embeddings", {"model": model, "input": [a, b]})
        _await_full_hit(gateway, wire, {"model": model, "input": [a, b]})
        mixed: Final = gateway.post("/v1/embeddings", {"model": model, "input": [a, c]})
        assert _sent_inputs(wire) == [[c]]
        assert vectors(mixed) == [embedding_vector(a), embedding_vector(c)], mixed


def test_a_partial_hit_keeps_the_request_order_when_the_cached_input_comes_last(gateway: Gateway) -> None:
    """Input A is cached. A request for [C, D, A] must send [C, D] upstream and return vectors in the order
    C, D, A with indexes 0, 1, 2: fresh vectors fill the gaps in request order.
    """
    with wire_server(openai_embedding_peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire.url)
        a, c, d = _texts(3)
        gateway.post("/v1/embeddings", {"model": model, "input": a})
        _await_full_hit(gateway, wire, {"model": model, "input": a})
        mixed: Final = gateway.post("/v1/embeddings", {"model": model, "input": [c, d, a]})
        assert _sent_inputs(wire) == [[c, d]]
        assert vectors(mixed) == [embedding_vector(c), embedding_vector(d), embedding_vector(a)], mixed


def test_each_input_of_a_list_is_cached_on_its_own(gateway: Gateway) -> None:
    """The list [A, B] is embedded and cached. A later request for the single string B, and one for the single
    element list [A], must both be answered without reaching the provider.
    """
    with wire_server(openai_embedding_peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire.url)
        a, b = _texts(2)
        gateway.post("/v1/embeddings", {"model": model, "input": [a, b]})
        _await_full_hit(gateway, wire, {"model": model, "input": [a, b]})
        single: Final = gateway.post("/v1/embeddings", {"model": model, "input": b})
        listed: Final = gateway.post("/v1/embeddings", {"model": model, "input": [a]})
        assert _sent_inputs(wire) == []
        assert vectors(single) == [embedding_vector(b)] and vectors(listed) == [embedding_vector(a)]


def test_a_fully_cached_list_reports_the_summed_prompt_tokens_of_its_inputs(gateway: Gateway) -> None:
    """The provider reports 6 prompt tokens for a two-input list, stored as 3 per input. The cached answer for
    that list must report ``usage.prompt_tokens == 6`` and ``total_tokens == 6``.
    """
    with wire_server(openai_embedding_peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire.url)
        body: Final[dict[str, JsonValue]] = {"model": model, "input": _texts(2)}
        gateway.post("/v1/embeddings", body)
        usage: Final = object_value(_await_full_hit(gateway, wire, body)["usage"])
        expected: Final = 2 * EMBEDDING_TOKENS_PER_INPUT
        assert (usage["prompt_tokens"], usage["total_tokens"]) == (expected, expected), usage


def test_a_partial_hit_reports_cached_plus_fresh_prompt_tokens(gateway: Gateway) -> None:
    """Input A is cached with 3 prompt tokens. A request for [A, C] gets C from the provider (3 tokens).
    The combined answer must report ``usage.prompt_tokens == 6``: cached and fresh usage are added.
    """
    with wire_server(openai_embedding_peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire.url)
        a, c = _texts(2)
        gateway.post("/v1/embeddings", {"model": model, "input": a})
        _await_full_hit(gateway, wire, {"model": model, "input": a})
        mixed: Final = gateway.post("/v1/embeddings", {"model": model, "input": [a, c]})
        assert _sent_inputs(wire) == [[c]]
        assert object_value(mixed["usage"])["prompt_tokens"] == 2 * EMBEDDING_TOKENS_PER_INPUT, mixed


def test_the_fresh_inputs_of_a_partial_hit_are_cached_for_next_time(gateway: Gateway) -> None:
    """Input A is cached, then [A, C] is requested, which fetches C. After that the list [A, C] and the single
    input C must be answered without reaching the provider.
    """
    with wire_server(openai_embedding_peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire.url)
        a, c = _texts(2)
        gateway.post("/v1/embeddings", {"model": model, "input": a})
        _await_full_hit(gateway, wire, {"model": model, "input": a})
        gateway.post("/v1/embeddings", {"model": model, "input": [a, c]})
        wire.drain()
        _await_full_hit(gateway, wire, {"model": model, "input": [a, c]})
        gateway.post("/v1/embeddings", {"model": model, "input": c})
        assert _sent_inputs(wire) == []


def test_no_store_on_an_embedding_list_caches_none_of_its_inputs(gateway: Gateway) -> None:
    """A two-input list is embedded with ``cache: {"no-store": true}``. The identical plain request must send
    both inputs to the provider again.
    """
    with wire_server(openai_embedding_peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire.url)
        inputs: Final[list[JsonValue]] = list(_texts(2))
        gateway.post("/v1/embeddings", {"model": model, "input": inputs, "cache": {"no-store": True}})
        wire.drain()
        gateway.post("/v1/embeddings", {"model": model, "input": inputs})
        assert _sent_inputs(wire) == [inputs]


def test_a_provider_error_on_the_uncached_part_fails_the_request_and_keeps_the_cached_part(gateway: Gateway) -> None:
    """Input A is cached. The provider is then made to fail with 500 for any request. A request for [A, C]
    must fail with an error status instead of returning a partial answer, and once the provider recovers a
    request for A alone must still be answered from the cache.
    """
    failing: list[bool] = [False]  # mutable-ok: flipped by the test to break and heal the peer

    def peer(request: Request) -> Reply:
        if failing[0]:
            return Reply(status=500, body=b'{"error":{"message":"scripted embedding outage","type":"server_error"}}')
        return openai_embedding_peer(request)

    with wire_server(peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(
            scenario,
            wire.url,
        )
        a, c = _texts(2)
        gateway.post("/v1/embeddings", {"model": model, "input": a})
        _await_full_hit(gateway, wire, {"model": model, "input": a})
        failing[0] = True
        broken: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": [a, c]})
        assert broken.status_code >= 400, broken.text
        assert "embedding" not in broken.text or "scripted embedding outage" in broken.text, broken.text
        failing[0] = False
        wire.drain()
        healed: Final = gateway.post("/v1/embeddings", {"model": model, "input": a})
        assert _sent_inputs(wire) == [] and vectors(healed) == [embedding_vector(a)]


def test_an_answer_with_fewer_vectors_than_inputs_is_not_cached(gateway: Gateway) -> None:
    """The provider answers a two-input list with a single vector. Whatever the caller receives, nothing may be
    cached for either input (``async_add_cache_pipeline`` skips the write when the counts differ), so the
    identical request must send both inputs to the provider again.
    """

    def short_peer(request: Request) -> Reply:
        body: Final = {
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.25, 0.75]}],
            "usage": {"prompt_tokens": 3, "total_tokens": 3},
        }
        return Reply(body=json.dumps(body).encode())

    with wire_server(short_peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire.url)
        inputs: Final[list[JsonValue]] = list(_texts(2))
        first: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": inputs})
        second: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": inputs})
        assert first.status_code == second.status_code, (first.text, second.text)
        assert _sent_inputs(wire) == [inputs, inputs]
