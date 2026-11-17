"""Shared helpers for the ``test_dec1_cache2_*`` response cache contracts (second wave).

It builds on ``dec1_cache_support`` (``Probe``, ``Exchange``, owned cache rigs) and adds one thing those
helpers lack: a ``RouteProbe`` that asks any cached route (chat, text completions, Responses, Anthropic
Messages, embeddings, rerank) the same way, knows how to recognise the stored answer of that route, and
finds the Redis entry the answer was written under, namespaced or not. A hit is still proven the black-box
way: the request did not reach the scripted upstream and returned the stored answer.

It also has a counting provider peer (``provider_peer``) for the big provider cells, which answer in the
provider's own wire shape (Anthropic, Gemini, Bedrock Converse, Azure) instead of the OpenAI shape the
scripted upstream speaks, and a raw body sender for cells about JSON member order and whitespace.
"""

from __future__ import annotations

import json
import threading
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Final

import httpx
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import JSON_OBJECT, Gateway, Scenario, list_value, object_value, string_value
from tests.integration._support.wire import Reply, Request, Wire, wire_server
from tests.integration.caching.dec1_cache_support import (
    CACHE_KEY_HEADER,
    Exchange,
    Probe,
    await_named_entry,
    probe,
    unique_embedding,
)
from tests.integration.caching.response_cache_case import (
    CHAT,
    MESSAGE,
    MESSAGES_MODEL,
    RERANK,
    RESPONSE,
    TEXT,
    TEXT_MODEL,
    chat_body,
    json_response,
    message_body,
)

PRICED: Final[dict[str, JsonValue]] = {"input_cost_per_token": 0.001, "output_cost_per_token": 0.002}
ONE_CALL_COST: Final = 1 * 0.001 + 1 * 0.002
EMBEDDING_MODEL: Final = "openai/text-embedding-3-small"
RERANK_MODEL: Final = "cohere/rerank-v4.0"
RERANK_DOCUMENTS: Final[list[JsonValue]] = ["alpha", "beta"]
DEFAULT_ENTRY_TTL_SECONDS: Final = 60

CHAT_ROUTE: Final = "chat"
TEXT_ROUTE: Final = "completions"
RESPONSES_ROUTE: Final = "responses"
MESSAGES_ROUTE: Final = "messages"
EMBEDDINGS_ROUTE: Final = "embeddings"
RERANK_ROUTE: Final = "rerank"
ALL_ROUTES: Final = (CHAT_ROUTE, TEXT_ROUTE, RESPONSES_ROUTE, MESSAGES_ROUTE, EMBEDDINGS_ROUTE, RERANK_ROUTE)
NON_CHAT_ROUTES: Final = ALL_ROUTES[1:]

PATHS: Final[Mapping[str, str]] = {
    CHAT_ROUTE: "/v1/chat/completions",
    TEXT_ROUTE: "/v1/completions",
    RESPONSES_ROUTE: "/v1/responses",
    MESSAGES_ROUTE: "/v1/messages",
    EMBEDDINGS_ROUTE: "/v1/embeddings",
    RERANK_ROUTE: "/v1/rerank",
}
CALL_TYPES: Final[Mapping[str, str]] = {
    CHAT_ROUTE: "acompletion",
    TEXT_ROUTE: "atext_completion",
    RESPONSES_ROUTE: "aresponses",
    MESSAGES_ROUTE: "aanthropic_messages",
    EMBEDDINGS_ROUTE: "aembedding",
    RERANK_ROUTE: "arerank",
}


def route_body(route: str, model: str, text: str, **extra: JsonValue) -> dict[str, JsonValue]:
    """The smallest valid body of ``route`` whose prompt (or query, or input) is ``text``."""
    if route == CHAT_ROUTE:
        return chat_body(model, text, **extra)
    if route == MESSAGES_ROUTE:
        return message_body(model, text, **extra)
    if route == TEXT_ROUTE:
        return {"model": model, "prompt": text, **extra}
    if route == RERANK_ROUTE:
        return {"model": model, "query": text, "documents": list(RERANK_DOCUMENTS), **extra}
    return {"model": model, "input": text, **extra}


@dataclass(frozen=True, slots=True)
class RouteProbe:
    """One proxy model that serves ``route`` on one scripted upstream scenario.

    ``marker`` is set for embeddings only: an embedding answer has no id, so the unique float in its vector
    identifies the stored entry and the replayed answer.
    """

    route: str
    probe: Probe
    marker: str | None = None

    @property
    def model(self) -> str:
        return self.probe.model

    def calls(self) -> int:
        return self.probe.calls()

    def ask(
        self,
        text: str,
        *,
        key: str | None = None,
        headers: Mapping[str, str] | None = None,
        **extra: JsonValue,
    ) -> Exchange:
        return self.probe.send(
            PATHS[self.route], route_body(self.route, self.model, text, **extra), key=key, headers=headers
        )

    def send(self, body: Mapping[str, JsonValue], *, key: str | None = None) -> Exchange:
        return self.probe.send(PATHS[self.route], body, key=key)

    def answer_id(self, exchange: Exchange) -> str:
        """What identifies the answer: the body id, the first Responses output item id (the top level Responses
        id may be re-encoded per caller), or the embedding marker when the vector carries it."""
        body: Final = exchange.json()
        if self.route == EMBEDDINGS_ROUTE:
            assert self.marker is not None
            return self.marker if self.marker in exchange.response.text else exchange.response.text
        if self.route == RESPONSES_ROUTE:
            return string_value(object_value(list_value(body["output"])[0])["id"])
        return string_value(body["id"])

    def needle(self, exchange: Exchange) -> str:
        """The text the stored Redis entry of ``exchange`` must contain."""
        return self.marker if self.marker is not None else self.answer_id(exchange)

    def fill(self, store: Redis[bytes], text: str, **extra: JsonValue) -> tuple[Exchange, str]:
        """Ask once, require that it reached the upstream, and return it with the Redis key it was written under."""
        first: Final = self.ask(text, **extra)
        assert first.reached_upstream, f"the filling request was served from the cache: {first.response.text}"
        return first, await_named_entry(store, self.needle(first))

    def assert_hit(self, first: Exchange, second: Exchange) -> None:
        assert second.response.status_code == 200, second.response.text
        assert not second.reached_upstream, (
            f"an identical {self.route} request reached the upstream: {second.response.text}"
        )
        assert self.answer_id(second) == self.answer_id(first), (first.response.text, second.response.text)
        if self.route != EMBEDDINGS_ROUTE:
            assert second.response.headers.get(CACHE_KEY_HEADER), dict(second.response.headers)

    def assert_miss(self, second: Exchange) -> None:
        assert second.response.status_code == 200, second.response.text
        assert second.reached_upstream, f"the {self.route} request was served a stored answer: {second.response.text}"


def route_probe(scenario: Scenario, route: str, *, name: str | None = None, **parameters: JsonValue) -> RouteProbe:
    """Register a scripted scenario answering ``route`` in its own shape and a proxy model on it."""
    if route == EMBEDDINGS_ROUTE:
        answer, marker = unique_embedding()
        return RouteProbe(route, probe(scenario, answer, name=name, model=EMBEDDING_MODEL, **parameters), marker)
    if route == TEXT_ROUTE:
        return RouteProbe(route, probe(scenario, json_response(TEXT), name=name, model=TEXT_MODEL, **parameters))
    if route == MESSAGES_ROUTE:
        return RouteProbe(route, probe(scenario, json_response(MESSAGE), name=name, model=MESSAGES_MODEL, **parameters))
    if route == RERANK_ROUTE:
        return RouteProbe(route, probe(scenario, json_response(RERANK), name=name, model=RERANK_MODEL, **parameters))
    if route == RESPONSES_ROUTE:
        return RouteProbe(route, probe(scenario, json_response(RESPONSE), name=name, **parameters))
    return RouteProbe(route, probe(scenario, json_response(CHAT), name=name, **parameters))


def entry_ttl(store: Redis[bytes], key: str) -> int:
    """The remaining Redis TTL of ``key`` in seconds (-1 without expiry, -2 when missing)."""
    return int(store.ttl(key))


def send_raw(
    gateway: Gateway, path: str, text: str, *, key: str | None = None, headers: Mapping[str, str] | None = None
) -> httpx.Response:
    """POST ``text`` verbatim as the JSON body, so member order and whitespace reach the proxy exactly as written."""
    return gateway.client.post(
        path,
        content=text.encode(),
        headers={
            "Authorization": f"Bearer {gateway.key if key is None else key}",
            "Content-Type": "application/json",
            **(headers or {}),
        },
    )


def raw_exchange(target: RouteProbe, text: str, *, key: str | None = None) -> Exchange:
    before: Final = target.calls()
    response: Final = send_raw(target.probe.gateway, PATHS[target.route], text, key=key)
    return Exchange(response, target.calls() > before)


def ordered_json(pairs: list[tuple[str, JsonValue]]) -> str:
    """A JSON object whose members appear in exactly the order of ``pairs``."""
    return "{" + ",".join(f"{json.dumps(name)}:{json.dumps(value)}" for name, value in pairs) + "}"


@dataclass(slots=True)
class ProviderPeer:
    """A provider that answers every POST with ``answer(request, n)`` in its own wire shape and counts the POSTs."""

    wire: Wire
    seen: list[Request] = field(default_factory=list)
    lock: threading.Lock = field(default_factory=threading.Lock)

    @property
    def url(self) -> str:
        return self.wire.url

    def calls(self) -> int:
        with self.lock:
            return sum(1 for request in self.seen if request.method == "POST")


@contextmanager
def provider_peer(answer: Callable[[Request, str], Mapping[str, JsonValue]]) -> Iterator[ProviderPeer]:
    """A counting provider peer. ``answer`` gets the request and a fresh unique id for this reply."""
    holder: Final[list[ProviderPeer]] = []

    def respond(request: Request) -> Reply:
        peer: Final = holder[0]
        with peer.lock:
            peer.seen.append(request)
        return Reply(body=json.dumps(answer(request, uuid.uuid4().hex[:16])).encode())

    with wire_server(respond) as wire:
        holder.append(ProviderPeer(wire))
        yield holder[0]


@dataclass(frozen=True, slots=True)
class PeerExchange:
    response: httpx.Response
    reached_provider: bool

    def json(self) -> dict[str, JsonValue]:
        assert self.response.status_code == 200, self.response.text
        return JSON_OBJECT.validate_json(self.response.content)


def ask_peer(gateway: Gateway, peer: ProviderPeer, path: str, body: Mapping[str, JsonValue]) -> PeerExchange:
    before: Final = peer.calls()
    response: Final = gateway.request("POST", path, body)
    return PeerExchange(response, peer.calls() > before)
