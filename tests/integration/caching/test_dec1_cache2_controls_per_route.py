"""Per-request cache controls on every cached route, the ``Cache-Control`` header, and configured defaults.

The ``cache`` body object carries ``ttl``, ``s-maxage`` (or ``s-max-age``), ``no-cache``, ``no-store`` and
``namespace`` (``DynamicCacheControl``). The chat cells, and ``no-cache`` on most routes, are in
``test_dec1_cache_request_controls.py``. This file checks that each remaining control behaves the same on
``/v1/completions``, ``/v1/responses``, ``/v1/messages`` (Rust), ``/v1/embeddings`` and ``/v1/rerank``, where
the Rust request path will parse the controls itself.

The HTTP header ``Cache-Control`` is read in ``add_litellm_data_to_request``
(``litellm/proxy/litellm_pre_call_utils.py:3078``): its ``s-maxage`` becomes the top-level ``ttl`` of the
write, and any header without ``s-maxage`` sets ``ttl`` to ``None``. It is never a read freshness limit.

A second owned proxy runs with ``cache_params.ttl: 120`` and ``namespace: dec1c2cfg`` to pin how configured
defaults combine with request controls: ``Cache._add_cache_logic`` (``caching.py:765``) sets the configured
ttl first and then lets ``cache.ttl`` override it, so a top-level ``ttl`` loses to the configured one;
``RedisCache.check_and_fix_namespace`` (``redis_cache.py:756``) prefixes every key with the configured
namespace, also in front of a request namespace.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from typing import Final

import pytest
from redis import Redis

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.redis_process import owned_redis
from tests.integration.caching.dec1_cache2_support import (
    CHAT_ROUTE,
    DEFAULT_ENTRY_TTL_SECONDS,
    EMBEDDINGS_ROUTE,
    MESSAGES_ROUTE,
    NON_CHAT_ROUTES,
    RERANK_ROUTE,
    RESPONSES_ROUTE,
    TEXT_ROUTE,
    entry_ttl,
    route_probe,
)
from tests.integration.caching.dec1_cache_support import (
    CACHE_KEY_HEADER,
    CacheRig,
    await_named_entry,
    cache_config,
    cache_rig,
    entry_keys_naming,
    redis_params,
)
from tests.integration.caching.response_cache_case import CACHE_KEY, prompt, redis_store

pytestmark: Final = [pytest.mark.shares_worker_state, pytest.mark.timeout(300)]

ROUTE_PARAMS: Final = tuple(pytest.param(route, id=route) for route in NON_CHAT_ROUTES)
CONFIGURED_TTL: Final = 120
CONFIGURED_NAMESPACE: Final = "dec1c2cfg"


@pytest.fixture(scope="module")
def configured_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CacheRig]:
    directory: Final = tmp_path_factory.mktemp("dec1-cache2-configured")
    config: Final = cache_config(
        directory, cache_params=redis_params(ttl=CONFIGURED_TTL, namespace=CONFIGURED_NAMESPACE)
    )
    with owned_redis(directory) as redis, cache_rig(session_gateway, directory, config, redis=redis) as rig:
        yield rig


@pytest.mark.parametrize("route", ROUTE_PARAMS)
def test_request_ttl_sets_the_entry_expiry_on_every_route(gateway: Gateway, route: str) -> None:
    """On each route a request with ``cache: {"ttl": 300}`` fills an entry. Its Redis TTL must be above the 60
    second default and at most 300. For embeddings every per-input entry gets the request ttl.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        _, key = target.fill(store, prompt(), cache={"ttl": 300})
        assert DEFAULT_ENTRY_TTL_SECONDS < entry_ttl(store, key) <= 300, (key, entry_ttl(store, key))


@pytest.mark.parametrize("route", ROUTE_PARAMS)
def test_no_store_leaves_no_entry_on_every_route(gateway: Gateway, route: str) -> None:
    """On each route a request with ``cache: {"no-store": true}`` is answered 200 from the upstream. A plain
    identical request afterwards must reach the upstream as well, because nothing was written.
    """
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        first: Final = target.ask(text, cache={"no-store": True})
        target.assert_miss(first)
        target.assert_miss(target.ask(text))


@pytest.mark.parametrize("route", ROUTE_PARAMS)
def test_s_maxage_refuses_an_old_entry_on_every_route(gateway: Gateway, route: str) -> None:
    """On each route an entry is filled, then the identical request with ``cache: {"s-maxage": 1}`` is repeated.
    Once the entry is older than one second that request must reach the upstream, while a plain repeat right
    before was still a hit: ``s-maxage`` is checked against the stored ``timestamp`` (``Cache.get_cache_logic``).
    UNVERIFIED for embeddings: the per-input lookup passes no ``cache`` object to ``async_get_cache``
    (``caching_handler.py:1033``, the same gap as product-bugs item 26), so ``s-maxage`` may be ignored there;
    if so this cell is a bug for embeddings only.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        first, _ = target.fill(store, text)
        target.assert_hit(first, target.ask(text))
        aged: Final = eventually(
            lambda: target.ask(text, cache={"s-maxage": 1}), lambda exchange: exchange.reached_upstream, 15
        )
        target.assert_miss(aged)


@pytest.mark.parametrize("route", ROUTE_PARAMS)
def test_request_namespace_prefixes_the_entry_key_on_every_route(gateway: Gateway, route: str) -> None:
    """On each route a request with ``cache: {"namespace": "dec1c2ns"}`` fills an entry. Its Redis key must be
    ``dec1c2ns:`` followed by a sha256 hex digest, and a plain request (no namespace) for the same prompt must
    reach the upstream: namespaces isolate entries on every route.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        _, key = target.fill(store, text, cache={"namespace": "dec1c2ns"})
        namespace, _, digest = key.partition(":")
        assert namespace == "dec1c2ns" and CACHE_KEY.match(digest), key
        target.assert_miss(target.ask(text))


@pytest.mark.parametrize("route", ROUTE_PARAMS)
def test_caching_false_bypasses_the_cache_on_every_route(gateway: Gateway, route: str) -> None:
    """On each route an entry is filled, then the identical request with top-level ``"caching": false`` is sent.
    It must reach the upstream: ``LLMCachingHandler.reads_cache`` only reads when ``caching`` is unset or true.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        target.fill(store, text)
        target.assert_miss(target.ask(text, caching=False))


@pytest.mark.parametrize(
    "route", (pytest.param(MESSAGES_ROUTE, id="messages"), pytest.param(EMBEDDINGS_ROUTE, id="embeddings"))
)
def test_cache_control_header_s_maxage_sets_the_entry_ttl_on_other_routes(gateway: Gateway, route: str) -> None:
    """A request with the HTTP header ``Cache-Control: s-maxage=300`` and no ``cache`` body object fills an entry
    on ``/v1/messages`` or ``/v1/embeddings``. The entry's Redis TTL must be above 60 and at most 300: the header
    is read for every LLM route, not only chat.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        first: Final = target.ask(prompt(), headers={"Cache-Control": "s-maxage=300"})
        assert first.reached_upstream, first.response.text
        key: Final = await_named_entry(store, target.needle(first))
        assert DEFAULT_ENTRY_TTL_SECONDS < entry_ttl(store, key) <= 300, entry_ttl(store, key)


def test_cache_control_header_s_maxage_wins_over_a_top_level_ttl(gateway: Gateway) -> None:
    """A chat request carries top-level ``"ttl": 300`` and the header ``Cache-Control: s-maxage=200``. The entry's
    TTL must be above 60 and at most 200: the header overwrites ``data["ttl"]`` after the body was read.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        first: Final = target.ask(prompt(), headers={"Cache-Control": "s-maxage=200"}, ttl=300)
        assert first.reached_upstream, first.response.text
        key: Final = await_named_entry(store, target.needle(first))
        assert DEFAULT_ENTRY_TTL_SECONDS < entry_ttl(store, key) <= 200, entry_ttl(store, key)


def test_cache_control_header_without_s_maxage_clears_a_top_level_ttl(gateway: Gateway) -> None:
    """A chat request carries top-level ``"ttl": 300`` and the header ``Cache-Control: no-store``. The answer must
    still be written (the header form of ``no-store`` is not a control) and its TTL must be at most the 60 second
    default: any ``Cache-Control`` header sets ``data["ttl"] = cache_dict.get("s-maxage")``, which is ``None``
    here, so the caller's own ``ttl`` is dropped. Pinned current behaviour (``litellm_pre_call_utils.py:3081``).
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        first: Final = target.ask(prompt(), headers={"Cache-Control": "no-store"}, ttl=300)
        assert first.reached_upstream, first.response.text
        key: Final = await_named_entry(store, target.needle(first))
        assert 0 < entry_ttl(store, key) <= DEFAULT_ENTRY_TTL_SECONDS, entry_ttl(store, key)


def test_cache_control_header_s_maxage_is_not_a_read_freshness_limit(gateway: Gateway) -> None:
    """An entry is filled. Three seconds later the identical request carries ``Cache-Control: s-maxage=1``. It must
    still be a hit: the header only sets the ttl of a write, unlike ``cache: {"s-maxage": 1}`` in the body, which
    refuses an entry older than one second.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        first, key = target.fill(store, text)
        eventually(
            lambda: _entry_age(store, key),
            lambda age: age >= 3,
            15,
        )
        target.assert_hit(first, target.ask(text, headers={"Cache-Control": "s-maxage=1"}))


def _entry_age(store: Redis[bytes], key: str) -> float:
    """Seconds since the stored entry's ``timestamp``."""
    raw: Final = store.get(key)
    assert raw is not None, key
    return time.time() - float(json.loads(raw)["timestamp"])


def test_the_hit_header_names_the_namespaced_key_on_messages(gateway: Gateway) -> None:
    """A ``/v1/messages`` request with ``cache: {"namespace": "dec1c2msg"}`` fills an entry and the identical
    request is a hit. The hit's ``x-litellm-cache-key`` must be ``dec1c2msg:<sha256>`` and equal the Redis key,
    so ``/cache/delete`` with the header value evicts it, also on the Rust messages path.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, MESSAGES_ROUTE)
        text: Final = prompt()
        first, key = target.fill(store, text, cache={"namespace": "dec1c2msg"})
        second: Final = target.ask(text, cache={"namespace": "dec1c2msg"})
        target.assert_hit(first, second)
        assert second.response.headers[CACHE_KEY_HEADER] == key


def test_configured_ttl_sets_the_expiry_of_every_entry(configured_rig: CacheRig) -> None:
    """On the proxy with ``cache_params.ttl: 120`` a plain chat request fills an entry. Its Redis TTL must be above
    60 and at most 120: the configured ttl replaces the 60 second default.
    """
    with configured_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        _, key = target.fill(configured_rig.store(), prompt())
        assert DEFAULT_ENTRY_TTL_SECONDS < entry_ttl(configured_rig.store(), key) <= CONFIGURED_TTL


def test_a_request_cache_ttl_overrides_the_configured_ttl(configured_rig: CacheRig) -> None:
    """On the ``ttl: 120`` proxy a request with ``cache: {"ttl": 600}`` fills an entry. Its TTL must be above 120
    and at most 600: ``cache.ttl`` is applied after the configured ttl (``caching.py:782``).
    """
    with configured_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        _, key = target.fill(configured_rig.store(), prompt(), cache={"ttl": 600})
        assert CONFIGURED_TTL < entry_ttl(configured_rig.store(), key) <= 600


def test_a_top_level_ttl_loses_to_the_configured_ttl(configured_rig: CacheRig) -> None:
    """On the ``ttl: 120`` proxy a request with top-level ``"ttl": 600`` fills an entry. Its TTL must be at most
    120: ``_add_cache_logic`` writes the configured ttl over ``kwargs["ttl"]`` and only ``cache.ttl`` overrides
    it back. Pinned current behaviour; on the shared rig without a configured ttl the top-level ttl does apply.
    """
    with configured_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        _, key = target.fill(configured_rig.store(), prompt(), ttl=600)
        assert 0 < entry_ttl(configured_rig.store(), key) <= CONFIGURED_TTL


def test_configured_namespace_prefixes_every_entry_key(configured_rig: CacheRig) -> None:
    """On the proxy with ``cache_params.namespace: dec1c2cfg`` a plain chat request and an embedding request fill
    entries. Both Redis keys must be ``dec1c2cfg:`` followed by a sha256 digest.
    """
    store: Final = configured_rig.store()
    with configured_rig.gateway.scenario() as scenario:
        for route in (CHAT_ROUTE, EMBEDDINGS_ROUTE):
            target = route_probe(scenario, route)
            _, key = target.fill(store, prompt())
            namespace, _, digest = key.partition(":")
            assert (namespace, bool(CACHE_KEY.match(digest))) == (CONFIGURED_NAMESPACE, True), key


def test_a_request_namespace_is_nested_under_the_configured_namespace(configured_rig: CacheRig) -> None:
    """On the ``dec1c2cfg`` proxy a request with ``cache: {"namespace": "dec1c2req"}`` fills an entry. Its Redis
    key must be ``dec1c2cfg:dec1c2req:<sha256>``: the request namespace goes into the computed key and the Redis
    client then prefixes the configured namespace. The hit header carries the key without the configured prefix
    (``dec1c2req:<sha256>``). UNVERIFIED: the header value is taken from the computed key before the Redis client
    prefixes it; check ``hit_cache_key`` in ``caching_handler.py:559``.
    """
    store: Final = configured_rig.store()
    with configured_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        first, key = target.fill(store, text, cache={"namespace": "dec1c2req"})
        parts: Final = key.split(":")
        assert parts[:2] == [CONFIGURED_NAMESPACE, "dec1c2req"] and CACHE_KEY.match(parts[2]), key
        second: Final = target.ask(text, cache={"namespace": "dec1c2req"})
        target.assert_hit(first, second)
        assert second.response.headers[CACHE_KEY_HEADER] == key.removeprefix(f"{CONFIGURED_NAMESPACE}:")


def test_a_request_namespace_wins_over_metadata_redis_namespace(gateway: Gateway) -> None:
    """A chat request carries both ``cache: {"namespace": "dec1c2a"}`` and ``metadata: {"redis_namespace":
    "dec1c2b"}``. The entry key must start with ``dec1c2a:``: ``_add_namespace_to_cache_key`` takes the
    ``cache`` namespace first, then ``metadata.redis_namespace``, then the configured one (``caching.py:615``).
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        _, key = target.fill(store, prompt(), cache={"namespace": "dec1c2a"}, metadata={"redis_namespace": "dec1c2b"})
        assert key.startswith("dec1c2a:"), key


def test_an_empty_request_namespace_falls_back_to_no_namespace(gateway: Gateway) -> None:
    """A chat request with ``cache: {"namespace": ""}`` fills an entry. Its key must be a bare sha256 digest, and
    a plain request for the same prompt must be a hit on it: an empty namespace is falsy and ignored.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        first, key = target.fill(store, text, cache={"namespace": ""})
        assert CACHE_KEY.match(key), key
        target.assert_hit(first, target.ask(text))


@pytest.mark.parametrize(
    "route",
    (
        pytest.param(TEXT_ROUTE, id="completions"),
        pytest.param(RESPONSES_ROUTE, id="responses"),
        pytest.param(RERANK_ROUTE, id="rerank"),
    ),
)
def test_no_store_still_reads_an_existing_entry_on_other_routes(gateway: Gateway, route: str) -> None:
    """On ``/v1/completions``, ``/v1/responses`` and ``/v1/rerank`` an entry exists. The identical request with
    ``cache: {"no-store": true}`` must be served from it without an upstream call: ``no-store`` only disables
    the write.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        first, _ = target.fill(store, text)
        target.assert_hit(first, target.ask(text, cache={"no-store": True}))
        assert entry_keys_naming(store, target.needle(first)), "the stored entry vanished"
