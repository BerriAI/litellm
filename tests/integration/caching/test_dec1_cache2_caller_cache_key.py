"""A caller-supplied ``cache_key`` in the request body.

A top-level ``cache_key`` replaces the computed key for both the lookup (``_request_cache_key``,
``litellm/caching/caching_handler.py:325`` and ``:1048``) and the write (``Cache._add_cache_logic``,
``litellm/caching/caching.py:770``). It is used verbatim: no sha256, and no request namespace, because the
namespace is only added inside ``get_cache_key``. ``RedisCache.check_and_fix_namespace`` still adds a
configured ``cache_params.namespace`` in front of it. Embeddings ignore it, since every input is keyed on its
own (``add_embedding_response_to_cache`` overwrites ``cache_key``, ``caching.py:946``).

``product-bugs.md`` of run 42966523075 lists this under "Needs a decision": any caller can read or overwrite
any entry whose key it knows, across models. The cells below pin what happens today so the Rust path either
keeps it or changes it on purpose; the cross-model cell is the one a security fix would flip.
"""

from __future__ import annotations

import uuid
from typing import Final

from tests.integration._support.client import Gateway, eventually
from tests.integration.caching.dec1_cache2_support import (
    CHAT_ROUTE,
    EMBEDDINGS_ROUTE,
    MESSAGES_ROUTE,
    RESPONSES_ROUTE,
    TEXT_ROUTE,
    route_probe,
)
from tests.integration.caching.dec1_cache_support import (
    CACHE_KEY_HEADER,
    SETTLE_SECONDS,
    assert_hit,
    entry_keys_naming,
    probe,
    sse_payloads,
    stored_entry,
)
from tests.integration.caching.response_cache_case import CHAT_STREAM, prompt, redis_store


def _caller_key() -> str:
    return f"dec1-cache2-caller-{uuid.uuid4().hex}"


def test_a_caller_cache_key_names_the_redis_entry_the_answer_is_written_under(gateway: Gateway) -> None:
    """A chat request carries ``cache_key: K`` where K is not a sha256 digest. After it reaches the upstream,
    Redis must hold a string key exactly equal to K whose stored ``response`` names the answer id: the write
    uses the caller's key verbatim.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        caller_key: Final = _caller_key()
        first: Final = target.chat(prompt(), cache_key=caller_key)
        assert first.reached_upstream, first.response.text
        keys: Final = eventually(
            lambda: entry_keys_naming(store, first.id()), lambda found: len(found) > 0, SETTLE_SECONDS
        )
        assert keys == frozenset({caller_key}), keys
        assert first.id() in str(stored_entry(store, caller_key)["response"])


def test_a_different_prompt_with_the_same_caller_cache_key_is_served_the_stored_answer(gateway: Gateway) -> None:
    """Prompt A is sent with ``cache_key: K``, then prompt B with the same K. The second must not reach the
    upstream and must return A's answer: with a caller key the prompt plays no part in the lookup.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        caller_key: Final = _caller_key()
        first: Final = target.chat(prompt(), cache_key=caller_key)
        eventually(lambda: redis_store().exists(caller_key), lambda found: found == 1, SETTLE_SECONDS)
        assert_hit(first, target.chat(prompt(), cache_key=caller_key))


def test_the_hit_header_carries_the_caller_cache_key(gateway: Gateway) -> None:
    """After a fill with ``cache_key: K``, the identical request is a hit and its ``x-litellm-cache-key``
    header must be exactly K, not a sha256 of the request.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        caller_key: Final = _caller_key()
        text: Final = prompt()
        target.chat(text, cache_key=caller_key)
        eventually(lambda: redis_store().exists(caller_key), lambda found: found == 1, SETTLE_SECONDS)
        second: Final = target.chat(text, cache_key=caller_key)
        assert second.response.headers.get(CACHE_KEY_HEADER) == caller_key, dict(second.response.headers)


def test_a_caller_cache_key_is_never_forwarded_to_the_provider(gateway: Gateway) -> None:
    """A chat request with ``cache_key: K`` reaches the upstream. The body the upstream received must not
    contain a ``cache_key`` member: it is a litellm control, not a provider parameter.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        first: Final = target.chat(prompt(), cache_key=_caller_key())
        assert first.reached_upstream, first.response.text
        assert all("cache_key" not in body for body in target.upstream_bodies()), target.upstream_bodies()


def test_a_caller_cache_key_ignores_the_request_namespace(gateway: Gateway) -> None:
    """A request carries ``cache_key: K`` and ``cache: {"namespace": "dec1-ns"}``. The entry must be written
    under exactly K, not ``dec1-ns:K``: the request namespace is only added to a computed key
    (``Cache._add_namespace_to_cache_key`` runs inside ``get_cache_key``).
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        caller_key: Final = _caller_key()
        first: Final = target.chat(prompt(), cache_key=caller_key, cache={"namespace": "dec1-ns"})
        keys: Final = eventually(
            lambda: entry_keys_naming(store, first.id()), lambda found: len(found) > 0, SETTLE_SECONDS
        )
        assert keys == frozenset({caller_key}), keys


def test_a_caller_cache_key_serves_its_entry_to_a_request_for_another_model(gateway: Gateway) -> None:
    """Model A answers a request carrying ``cache_key: K``. A request to model B (another deployment and
    upstream) with the same K must be served A's answer and B's upstream must not be called. Pinned current
    behaviour, flagged "needs a decision" in product-bugs.md: the caller key is not scoped by model, key or
    team. A rewrite that scopes it must flip this cell deliberately.
    """
    with gateway.scenario() as scenario:
        first_model: Final = probe(scenario)
        second_model: Final = probe(scenario)
        caller_key: Final = _caller_key()
        first: Final = first_model.chat(prompt(), cache_key=caller_key)
        eventually(lambda: redis_store().exists(caller_key), lambda found: found == 1, SETTLE_SECONDS)
        second: Final = second_model.chat(prompt(), cache_key=caller_key)
        assert second.response.status_code == 200, second.response.text
        assert second_model.calls() == 0
        assert second.id() == first.id()


def test_a_caller_cache_key_is_served_across_virtual_keys(gateway: Gateway) -> None:
    """Virtual key one fills an entry with ``cache_key: K``. Virtual key two, allowed the same model, sends
    another prompt with K and must be served key one's answer. Pinned current behaviour: entries are shared
    across callers, and a guessable caller key makes that explicit.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        first_key: Final = scenario.key(models=[target.model])
        second_key: Final = scenario.key(models=[target.model])
        caller_key: Final = _caller_key()
        first: Final = target.chat(prompt(), key=first_key, cache_key=caller_key)
        eventually(lambda: redis_store().exists(caller_key), lambda found: found == 1, SETTLE_SECONDS)
        assert_hit(first, target.chat(prompt(), key=second_key, cache_key=caller_key))


def test_no_cache_with_a_caller_cache_key_reaches_upstream_and_overwrites_that_entry(gateway: Gateway) -> None:
    """An entry exists under ``cache_key: K`` holding answer A. A request with K and ``cache: {"no-cache": true}``
    must reach the upstream and get answer B, and the value under K must then name B: ``no-cache`` skips the
    read but the write still goes to the caller's key.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        caller_key: Final = _caller_key()
        text: Final = prompt()
        first: Final = target.chat(text, cache_key=caller_key)
        eventually(lambda: store.exists(caller_key), lambda found: found == 1, SETTLE_SECONDS)
        fresh: Final = target.chat(text, cache_key=caller_key, cache={"no-cache": True})
        assert fresh.reached_upstream and fresh.id() != first.id(), fresh.response.text
        eventually(
            lambda: str(stored_entry(store, caller_key)["response"]),
            lambda value: fresh.id() in value,
            SETTLE_SECONDS,
        )


def test_a_caller_cache_key_is_not_restricted_by_allowed_cache_controls(gateway: Gateway) -> None:
    """A virtual key created with ``allowed_cache_controls: ["no-cache"]`` sends a request with a top-level
    ``cache_key``. It must answer 200: the allow-list (``_PROXY_CacheControlCheck``,
    ``litellm/proxy/hooks/cache_control_check.py``) only inspects the members of the ``cache`` object.
    """
    with gateway.scenario() as scenario:
        target: Final = probe(scenario)
        restricted: Final = scenario.key(models=[target.model], allowed_cache_controls=["no-cache"])
        answered: Final = target.chat(prompt(), key=restricted, cache_key=_caller_key())
        assert answered.response.status_code == 200, answered.response.text


def test_a_caller_cache_key_on_embeddings_is_ignored(gateway: Gateway) -> None:
    """An embedding request carries ``cache_key: K``. No Redis key equal to K may appear, and the stored vector
    must be found under a bare sha256 per-input key instead: embeddings key every input on its own and
    overwrite the caller's key (``caching.py:946``).
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, EMBEDDINGS_ROUTE)
        caller_key: Final = _caller_key()
        first: Final = target.ask(prompt(), cache_key=caller_key)
        assert first.reached_upstream, first.response.text
        filled_key: Final = eventually(
            lambda: entry_keys_naming(store, target.needle(first)), lambda found: len(found) > 0, SETTLE_SECONDS
        )
        assert caller_key not in filled_key, filled_key
        assert store.exists(caller_key) == 0


def test_a_caller_cache_key_on_a_stream_writes_and_replays_under_that_key(gateway: Gateway) -> None:
    """A streamed chat request with ``cache_key: K`` is repeated until it is replayed from the cache. Redis must
    hold K, and the replayed stream's ``x-litellm-cache-key`` header must be K: the stream lookup keeps a
    caller key (``caching_handler.py:1025`` only computes one when ``cache_key`` is absent).
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = probe(scenario, CHAT_STREAM)
        caller_key: Final = _caller_key()
        text: Final = prompt()
        replay = eventually(
            lambda: target.chat(text, cache_key=caller_key, stream=True),
            lambda exchange: not exchange.reached_upstream,
            30,
        )
        assert sse_payloads(replay.response), replay.response.text
        assert store.exists(caller_key) == 1
        assert replay.response.headers.get(CACHE_KEY_HEADER) == caller_key, dict(replay.response.headers)


def test_a_caller_cache_key_names_the_entry_on_every_other_route(gateway: Gateway) -> None:
    """For ``/v1/completions``, ``/v1/responses`` and the Rust-served ``/v1/messages``, a request with
    ``cache_key: K`` is followed by a different prompt with the same K. Each second request must be served the
    first answer without an upstream call, and Redis must hold K: every non-embedding route honours the
    caller key the same way chat does. UNVERIFIED for ``/v1/messages``: the Rust messages path receives the
    key from the Python host (``inference-messages/src/handler.rs`` ``caller_keyed``).
    """
    store: Final = redis_store()
    for route in (TEXT_ROUTE, RESPONSES_ROUTE, MESSAGES_ROUTE):
        with gateway.scenario() as scenario:
            target = route_probe(scenario, route)
            caller_key = _caller_key()
            first = target.ask(prompt(), cache_key=caller_key)
            assert first.reached_upstream, (route, first.response.text)
            eventually(lambda key=caller_key: store.exists(key), lambda found: found == 1, SETTLE_SECONDS)
            target.assert_hit(first, target.ask(prompt(), cache_key=caller_key))


def test_a_computed_key_and_a_caller_key_for_the_same_request_are_separate_entries(gateway: Gateway) -> None:
    """A plain chat request fills an entry under its computed sha256 key. The identical request with
    ``cache_key: K`` (a fresh K) must reach the upstream, because K names an empty slot, and its answer must be
    stored under K while the computed entry keeps the first answer.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        first, computed = target.fill(store, text)
        caller_key: Final = _caller_key()
        second: Final = target.ask(text, cache_key=caller_key)
        target.assert_miss(second)
        eventually(lambda: store.exists(caller_key), lambda found: found == 1, SETTLE_SECONDS)
        assert first.id() in str(stored_entry(store, computed)["response"])
