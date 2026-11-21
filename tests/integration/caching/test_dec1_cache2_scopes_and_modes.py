"""Where the response cache applies: ``supported_call_types``, ``mode: default_off`` per route, and key or team
cache settings in metadata.

``cache_params.supported_call_types`` lists the call types that may read or write the cache
(``LLMCachingHandler._is_call_type_supported_by_cache``, ``litellm/caching/caching_handler.py:1409``). The
check uses the cached function's name (``acompletion``, ``aembedding``, ``atext_completion``, ``arerank``, and
``anthropic_messages``, whose async function has the plain name) and ``aresponses`` is also covered by the plain
name ``responses``. One
owned proxy runs with ``["acompletion", "responses", "anthropic_messages"]`` to pin both rules.

``mode: default_off`` makes every route opt-in with ``cache: {"use-cache": true}``; the chat and embedding cells
are in ``test_dec1_cache_mode_default_off.py`` and this file adds the other routes on a second owned proxy.

Key metadata ``cache`` is merged into the request (``add_key_level_controls``,
``litellm/proxy/litellm_pre_call_utils.py:2435``); team metadata ``cache`` is not read at all. The cells pin
both, plus the edge cases of the merge that the existing key-control cells do not reach.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import Gateway
from tests.integration._support.redis_process import owned_redis
from tests.integration.caching.dec1_cache2_support import (
    CHAT_ROUTE,
    EMBEDDINGS_ROUTE,
    MESSAGES_ROUTE,
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
    probe,
    redis_params,
)
from tests.integration.caching.response_cache_case import MESSAGE_STREAM, MESSAGES_MODEL, prompt, redis_store

pytestmark: Final = [pytest.mark.shares_worker_state, pytest.mark.timeout(300)]

USE_CACHE: Final[dict[str, JsonValue]] = {"use-cache": True}


@pytest.fixture(scope="module")
def call_type_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CacheRig]:
    directory: Final = tmp_path_factory.mktemp("dec1-cache2-call-types")
    config: Final = cache_config(
        directory,
        cache_params=redis_params(supported_call_types=["acompletion", "responses", "anthropic_messages"]),
    )
    with owned_redis(directory) as redis, cache_rig(session_gateway, directory, config, redis=redis) as rig:
        yield rig


@pytest.fixture(scope="module")
def default_off_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CacheRig]:
    directory: Final = tmp_path_factory.mktemp("dec1-cache2-default-off")
    config: Final = cache_config(directory, cache_params=redis_params(mode="default_off"))
    with owned_redis(directory) as redis, cache_rig(session_gateway, directory, config, redis=redis) as rig:
        yield rig


def test_a_listed_call_type_is_cached(call_type_rig: CacheRig) -> None:
    """On the proxy whose ``supported_call_types`` is ``["acompletion", "responses", "anthropic_messages"]`` two
    identical chat requests are sent. The second must be a hit: ``acompletion`` is listed.
    """
    with call_type_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        text: Final = prompt()
        first, _ = target.fill(call_type_rig.store(), text)
        target.assert_hit(first, target.ask(text))


def test_the_plain_responses_name_covers_async_responses_calls(call_type_rig: CacheRig) -> None:
    """On the same proxy two identical ``/v1/responses`` requests are sent. The second must be a hit: the proxy
    calls ``aresponses``, and the list's ``responses`` covers it (``caching_handler.py:1426``).
    """
    with call_type_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, RESPONSES_ROUTE)
        text: Final = prompt()
        first, _ = target.fill(call_type_rig.store(), text)
        target.assert_hit(first, target.ask(text))


def test_the_plain_anthropic_messages_name_covers_messages_calls(call_type_rig: CacheRig) -> None:
    """On the same proxy two identical ``/v1/messages`` requests are sent. The second must be a hit. Pinned current
    behaviour: the cached function behind ``/v1/messages`` is the ``@client`` decorated ``async def
    anthropic_messages`` (litellm/llms/anthropic/messages/native/handler.py:552-553), so the call type checked
    against the list is the plain ``anthropic_messages`` (``caching_handler.py:1425``), the name the spend rows
    store too.
    """
    with call_type_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, MESSAGES_ROUTE)
        text: Final = prompt()
        first, _ = target.fill(call_type_rig.store(), text)
        target.assert_hit(first, target.ask(text))


@pytest.mark.parametrize(
    "route",
    (
        pytest.param(EMBEDDINGS_ROUTE, id="embeddings"),
        pytest.param(TEXT_ROUTE, id="completions"),
        pytest.param(RERANK_ROUTE, id="rerank"),
    ),
)
def test_an_unlisted_call_type_is_neither_read_nor_written(call_type_rig: CacheRig, route: str) -> None:
    """On the same proxy two identical requests are sent to ``/v1/embeddings``, ``/v1/completions`` or
    ``/v1/rerank``. Both must reach the upstream, neither response may carry ``x-litellm-cache-key``, and no Redis
    entry may name the first answer.
    """
    with call_type_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        first: Final = target.ask(text)
        second: Final = target.ask(text)
        target.assert_miss(first)
        target.assert_miss(second)
        assert CACHE_KEY_HEADER not in second.response.headers
        assert entry_keys_naming(call_type_rig.store(), target.needle(first)) == frozenset()


@pytest.mark.parametrize(
    "route",
    (
        pytest.param(MESSAGES_ROUTE, id="messages"),
        pytest.param(RESPONSES_ROUTE, id="responses"),
        pytest.param(TEXT_ROUTE, id="completions"),
        pytest.param(RERANK_ROUTE, id="rerank"),
    ),
)
def test_default_off_caches_nothing_without_opt_in_on_other_routes(default_off_rig: CacheRig, route: str) -> None:
    """On the ``default_off`` proxy two identical plain requests are sent to each route. Both must reach the
    upstream and no entry may name the first answer.
    """
    with default_off_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        first: Final = target.ask(text)
        target.assert_miss(first)
        target.assert_miss(target.ask(text))
        assert entry_keys_naming(default_off_rig.store(), target.needle(first)) == frozenset()


@pytest.mark.parametrize(
    "route",
    (
        pytest.param(MESSAGES_ROUTE, id="messages"),
        pytest.param(RESPONSES_ROUTE, id="responses"),
        pytest.param(TEXT_ROUTE, id="completions"),
        pytest.param(RERANK_ROUTE, id="rerank"),
    ),
)
def test_default_off_use_cache_opts_in_on_other_routes(default_off_rig: CacheRig, route: str) -> None:
    """On the ``default_off`` proxy two identical requests with ``cache: {"use-cache": true}`` are sent to each
    route. The first must fill an entry and the second must be a hit on it, including on the Rust messages path.
    """
    with default_off_rig.gateway.scenario() as scenario:
        target: Final = route_probe(scenario, route)
        text: Final = prompt()
        first, _ = target.fill(default_off_rig.store(), text, cache=USE_CACHE)
        target.assert_hit(first, target.ask(text, cache=USE_CACHE))


def test_default_off_streams_are_not_cached_without_opt_in(default_off_rig: CacheRig) -> None:
    """On the ``default_off`` proxy the same streamed ``/v1/messages`` request is sent three times without
    ``use-cache``. Every one must reach the upstream: a stream is not a way around the opt-in.
    """
    with default_off_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario, MESSAGE_STREAM, model=MESSAGES_MODEL)
        text: Final = prompt()
        for _ in range(3):
            assert target.messages(text, stream=True).reached_upstream
        assert target.calls() == 3


def test_team_metadata_cache_settings_are_not_applied(gateway: Gateway) -> None:
    """A team is created with ``metadata: {"cache": {"no-cache": true}}`` and a key in it. An entry exists for a
    prompt; the team key sends the identical request and must be served the stored answer: only the key's own
    metadata ``cache`` is merged into the request (``litellm_pre_call_utils.py:3213``), team metadata is not.
    A rewrite that adds team level cache controls must flip this cell on purpose.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        team: Final = scenario.team(metadata={"cache": {"no-cache": True}})
        key: Final = scenario.key(team_id=team, models=[target.model])
        text: Final = prompt()
        first, _ = target.fill(store, text)
        target.assert_hit(first, target.ask(text, key=key))


def test_a_key_whose_metadata_cache_is_not_an_object_drops_the_callers_controls(gateway: Gateway) -> None:
    """A key has ``metadata: {"cache": "off"}`` (a string). An entry exists; the key sends the identical request
    with ``cache: {"no-cache": true}``. It must be a hit: a non-object key ``cache`` replaces the request's
    ``cache`` with ``{}`` (``litellm_pre_call_utils.py:2443``), so the caller's ``no-cache`` is dropped and the
    key does not disable caching either. Pinned current behaviour.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        key: Final = scenario.key(models=[target.model], metadata={"cache": "off"})
        text: Final = prompt()
        first, _ = target.fill(store, text)
        target.assert_hit(first, target.ask(text, key=key, cache={"no-cache": True}))


def test_a_caller_that_opts_in_keeps_its_own_controls_next_to_the_keys(gateway: Gateway) -> None:
    """A key has ``metadata: {"cache": {"ttl": 300}}``. An entry exists; the key sends the identical request with
    ``cache: {"use-cache": true, "no-cache": true}``. It must reach the upstream: when the request opts in with
    ``use-cache`` and the key does not opt out, the request's controls are kept and merged with the key's
    (``_request_cache_kept_for_key``), and the fresh answer is written with the key's 300 second ttl.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        key: Final = scenario.key(models=[target.model], metadata={"cache": {"ttl": 300}})
        text: Final = prompt()
        target.fill(store, text)
        fresh: Final = target.ask(text, key=key, cache={"use-cache": True, "no-cache": True})
        target.assert_miss(fresh)
        written: Final = await_named_entry(store, target.needle(fresh))
        assert 60 < entry_ttl(store, written) <= 300, entry_ttl(store, written)


def test_a_key_with_use_cache_false_in_metadata_still_hits_on_a_default_on_proxy(gateway: Gateway) -> None:
    """A key has ``metadata: {"cache": {"use-cache": false}}``. On the default-on shared rig an entry exists and
    the key sends the identical plain request. It must be a hit: ``use-cache`` is not one of the four supported
    key controls (``SupportedCacheControls``, ``litellm/types/utils.py:182``), so it is dropped, and on a default-on
    cache it could not opt out anyway. Pinned current behaviour.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, CHAT_ROUTE)
        key: Final = scenario.key(models=[target.model], metadata={"cache": {"use-cache": False}})
        text: Final = prompt()
        first, _ = target.fill(store, text)
        target.assert_hit(first, target.ask(text, key=key))


def test_key_metadata_no_cache_applies_on_the_rust_messages_path(gateway: Gateway) -> None:
    """A key has ``metadata: {"cache": {"no-cache": true}}``. A ``/v1/messages`` entry exists (filled by the master
    key); the restricted key sends the identical request. It must reach the upstream: the merged key controls
    must reach the Rust messages path just as they reach chat.
    """
    store: Final = redis_store()
    with gateway.scenario() as scenario:
        target: Final = route_probe(scenario, MESSAGES_ROUTE)
        key: Final = scenario.key(models=[target.model], metadata={"cache": {"no-cache": True}})
        text: Final = prompt()
        target.fill(store, text)
        target.assert_miss(target.ask(text, key=key))
