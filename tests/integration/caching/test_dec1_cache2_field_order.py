"""JSON member order, nesting and whitespace must not change the response cache key.

JSON object members are unordered, so two bodies that differ only in member order or whitespace are the same
request and must share one entry. Today ``Cache.get_cache_key`` (``litellm/caching/caching.py:431``)
concatenates the kwargs in the order the body listed them and formats nested values with ``str()``, which
keeps the member order of every nested dict as well. The top-level chat case is pinned (skipped as a bug) in
``test_dec1_cache_key_fields.py``; ``product-bugs.md`` item 15 of run 42966523075 records it. This file pins
the same contract on nested objects and on the other routes, plus the parts that already hold: excluded
fields may move freely, list order keeps its meaning, and whitespace never matters because the body is
parsed before hashing.

Every body is sent verbatim through ``send_raw`` so the member order on the wire is exactly the one written.
A Rust request path that hashes a canonical form (sorted keys) fixes the skipped cells; one that hashes the
raw bytes would break the whitespace cells.
"""

from __future__ import annotations

import json
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.caching.dec1_cache2_support import (
    CHAT_ROUTE,
    EMBEDDINGS_ROUTE,
    MESSAGES_ROUTE,
    RERANK_ROUTE,
    RESPONSES_ROUTE,
    TEXT_ROUTE,
    ordered_json,
    raw_exchange,
    route_body,
    route_probe,
)
from tests.integration.caching.dec1_cache_support import (
    await_named_entry,
    embedding_model,
    openai_embedding_peer,
    vectors,
)
from tests.integration.caching.response_cache_case import prompt, redis_store

WEATHER_PARAMETERS_AB: Final[dict[str, JsonValue]] = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "unit": {"type": "string"}},
}
WEATHER_PARAMETERS_BA: Final[dict[str, JsonValue]] = {
    "properties": {"unit": {"type": "string"}, "city": {"type": "string"}},
    "type": "object",
}


def _tool(parameters: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return {"type": "function", "function": {"name": "get_weather", "parameters": parameters}}


def test_reordered_members_of_a_message_object_hit_the_same_entry(gateway: Gateway) -> None:
    """A chat entry is filled with the message ``{"role": "user", "content": T}``. The same request is sent
    with the message written as ``{"content": T, "role": "user"}``. The second must be a hit.
    """
    pytest.skip("BUG: the cache key formats nested messages with str(), so message member order changes the key")
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        first: Final = raw_exchange(
            target,
            ordered_json([("model", target.model), ("messages", [{"role": "user", "content": text}])]),
        )
        await_named_entry(store, target.needle(first))
        second: Final = raw_exchange(
            target,
            '{"model":'
            + json.dumps(target.model)
            + ',"messages":[{"content":'
            + json.dumps(text)
            + ',"role":"user"}]}',
        )
        target.assert_hit(first, second)


def test_reordered_members_of_a_tool_schema_hit_the_same_entry(gateway: Gateway) -> None:
    """A chat entry is filled with one tool whose JSON schema lists ``type`` before ``properties`` and ``city``
    before ``unit``. The same request with both orders reversed must be a hit: the schemas are equal.
    """
    pytest.skip("BUG: nested tool schemas are hashed with their member order, so an equal schema misses")
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        first, _ = target.fill(store, text, tools=[_tool(WEATHER_PARAMETERS_AB)])
        target.assert_hit(first, target.ask(text, tools=[_tool(WEATHER_PARAMETERS_BA)]))


@pytest.mark.parametrize(
    "route",
    (
        pytest.param(MESSAGES_ROUTE, id="messages"),
        pytest.param(RESPONSES_ROUTE, id="responses"),
        pytest.param(TEXT_ROUTE, id="completions"),
        pytest.param(EMBEDDINGS_ROUTE, id="embeddings"),
        pytest.param(RERANK_ROUTE, id="rerank"),
    ),
)
def test_reordered_top_level_fields_hit_the_same_entry_on_every_route(gateway: Gateway, route: str) -> None:
    """On each non-chat route the same body is sent twice, the second time with its top-level members in reverse
    order (for example ``messages, max_tokens, model`` instead of ``model, max_tokens, messages``). The second
    must be a hit. The Rust messages path and the Python handler for the other routes must agree on this.
    """
    pytest.skip("BUG: the response cache key depends on the order of the top-level JSON members")
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        extra: Final[dict[str, JsonValue]] = {"temperature": 0} if route != RERANK_ROUTE else {"top_n": 1}
        first, _ = target.fill(store, text, **extra)
        body: Final = route_body(route, target.model, text, **extra)
        target.assert_hit(first, raw_exchange(target, ordered_json(list(reversed(list(body.items()))))))


def test_moving_metadata_in_the_body_does_not_change_the_key(gateway: Gateway) -> None:
    """The fill lists ``metadata`` first, the second request lists it last; both carry the same metadata. The
    second must be a hit: ``metadata`` is excluded from the key, so where it sits cannot matter. This part of
    the order contract holds today and must keep holding.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        messages: Final[list[JsonValue]] = [{"role": "user", "content": text}]
        first: Final = raw_exchange(
            target,
            ordered_json([("metadata", {"trace": "x"}), ("model", target.model), ("messages", messages)]),
        )
        await_named_entry(store, target.needle(first))
        second: Final = raw_exchange(
            target,
            ordered_json([("model", target.model), ("messages", messages), ("metadata", {"trace": "x"})]),
        )
        target.assert_hit(first, second)


def test_moving_the_cache_object_in_the_body_does_not_change_the_key(gateway: Gateway) -> None:
    """The fill puts ``cache: {"ttl": 120}`` before ``model``, the second request puts the same object after
    ``messages``. The second must be a hit: the ``cache`` control object is not a key field.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        messages: Final[list[JsonValue]] = [{"role": "user", "content": text}]
        first: Final = raw_exchange(
            target, ordered_json([("cache", {"ttl": 120}), ("model", target.model), ("messages", messages)])
        )
        await_named_entry(store, target.needle(first))
        second: Final = raw_exchange(
            target, ordered_json([("model", target.model), ("messages", messages), ("cache", {"ttl": 120})])
        )
        target.assert_hit(first, second)


@pytest.mark.parametrize(
    "route",
    (
        pytest.param(CHAT_ROUTE, id="chat"),
        pytest.param(MESSAGES_ROUTE, id="messages"),
        pytest.param(EMBEDDINGS_ROUTE, id="embeddings"),
    ),
)
def test_whitespace_in_the_json_body_does_not_change_the_key(gateway: Gateway, route: str) -> None:
    """The fill is sent as compact JSON; the second request is the same body pretty-printed with two space
    indentation and a trailing newline. The second must be a hit: the key is built from the parsed request,
    never from the raw bytes, and a Rust front door must keep that.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        body: Final = route_body(route, target.model, text)
        first: Final = raw_exchange(target, json.dumps(body, separators=(",", ":")))
        await_named_entry(store, target.needle(first))
        target.assert_hit(first, raw_exchange(target, json.dumps(body, indent=2) + "\n"))


def test_the_order_of_messages_in_the_list_is_part_of_the_key(gateway: Gateway) -> None:
    """The fill sends a system message then a user message; the second sends the same two messages in the
    other order. The second must reach the upstream: list order carries meaning and must stay in the key even
    once object member order stops mattering.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        system: Final[dict[str, JsonValue]] = {"role": "system", "content": "Be brief"}
        user: Final[dict[str, JsonValue]] = {"role": "user", "content": prompt()}
        first: Final = target.send({"model": target.model, "messages": [system, user]})
        await_named_entry(store, target.needle(first))
        target.assert_miss(target.send({"model": target.model, "messages": [user, system]}))


def test_the_order_of_tools_in_the_list_is_part_of_the_key(gateway: Gateway) -> None:
    """The fill sends tools ``[get_weather, get_time]``; the second sends ``[get_time, get_weather]``. The second
    must reach the upstream. Pinned current behaviour: the tool list is hashed in order. A rewrite that sorts
    tools would turn this into a hit, which needs a deliberate decision because tool order can change model
    behaviour.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        weather: Final = _tool(WEATHER_PARAMETERS_AB)
        time_tool: Final[dict[str, JsonValue]] = {"type": "function", "function": {"name": "get_time"}}
        text: Final = prompt()
        target.fill(store, text, tools=[weather, time_tool])
        target.assert_miss(target.ask(text, tools=[time_tool, weather]))


def test_reordered_embedding_inputs_are_each_served_from_the_cache(gateway: Gateway) -> None:
    """The list ``[A, B]`` is embedded once through an OpenAI embeddings peer that answers one vector per input.
    The list ``[B, A]`` must then be answered without any provider call, with the stored vectors in the new
    order: embeddings are cached per input (``caching_handler.py:1029``), so the list order never enters a key
    and each index maps back to its own input.
    """
    with wire_server(openai_embedding_peer) as wire, gateway.scenario() as scenario:
        model: Final = embedding_model(scenario, wire.url)
        first_text, second_text = prompt(), prompt()
        filling: Final = gateway.request("POST", "/v1/embeddings", {"model": model, "input": [first_text, second_text]})
        assert filling.status_code == 200, filling.text
        stored: Final = vectors(filling.json())
        assert len(stored) == 2 and len(wire.drain()) == 1, filling.text

        def reordered() -> tuple[int, list[JsonValue]]:
            response = gateway.request("POST", "/v1/embeddings", {"model": model, "input": [second_text, first_text]})
            assert response.status_code == 200, response.text
            return len(wire.drain()), vectors(response.json())

        calls, answered = eventually(reordered, lambda found: found[0] == 0, 15)
        assert calls == 0, "the reordered list reached the provider again"
        assert answered == [stored[1], stored[0]], answered
