"""``cache_params.mode: default_off``: caching is opt-in per request with ``cache: {"use-cache": true}``.

One owned proxy runs the Redis response cache with ``mode: default_off``. ``Cache.should_use_cache``
(``litellm/caching/caching.py``) then returns true only when the request's ``cache`` object carries
``use-cache`` equal to the literal ``true``; it guards both the lookup and the write. Every cell proves
the result with upstream call counts and with what is in the proxy's own Redis.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, Gateway, held
from tests.integration._support.redis_process import owned_redis
from tests.integration.caching.dec1_cache_support import (
    CACHE_KEY_HEADER,
    CacheRig,
    assert_hit,
    assert_miss,
    cache_config,
    cache_rig,
    entry_keys_naming,
    filled,
    probe,
    redis_params,
    unique_embedding,
)
from tests.integration.caching.response_cache_case import CACHE_KEY, await_entry, entry_key, prompt

pytestmark: Final = [pytest.mark.shares_worker_state, pytest.mark.timeout(300)]

USE_CACHE: Final[dict[str, JsonValue]] = {"use-cache": True}


@pytest.fixture(scope="module")
def opt_in_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CacheRig]:
    directory: Final = tmp_path_factory.mktemp("dec1-cache-default-off")
    config: Final = cache_config(directory, cache_params=redis_params(mode="default_off"))
    with owned_redis(directory) as redis, cache_rig(session_gateway, directory, config, redis=redis) as rig:
        yield rig


def test_plain_requests_are_never_cached_under_default_off(opt_in_rig: CacheRig) -> None:
    """Two identical plain chat requests are sent to the ``default_off`` proxy. Both must reach the upstream,
    neither response may carry ``x-litellm-cache-key``, and no Redis entry may name either answer.
    """
    with opt_in_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        second: Final = target.chat(text)
        assert_miss(first, second)
        assert CACHE_KEY_HEADER not in second.response.headers, dict(second.response.headers)
        store: Final = opt_in_rig.store()
        assert entry_keys_naming(store, first.id()) | entry_keys_naming(store, second.id()) == frozenset()


def test_use_cache_true_fills_and_then_hits(opt_in_rig: CacheRig) -> None:
    """Two identical requests carry ``cache: {"use-cache": true}``. The first must reach the upstream and be
    written to Redis under a sha256 key; the second must be served from it with that key in
    ``x-litellm-cache-key``.
    """
    with opt_in_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text, cache=USE_CACHE)
        key: Final = filled(opt_in_rig.store(), first)
        assert CACHE_KEY.match(key), key
        second: Final = target.chat(text, cache=USE_CACHE)
        assert_hit(first, second)
        assert second.cache_key() == key


def test_an_opted_in_entry_is_not_served_to_a_request_that_did_not_opt_in(opt_in_rig: CacheRig) -> None:
    """An entry was written by a ``use-cache`` request. The same request without the ``cache`` object must
    reach the upstream: under ``default_off`` the lookup itself is opt-in, not just the write.
    """
    with opt_in_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text, cache=USE_CACHE)
        filled(opt_in_rig.store(), first)
        assert_miss(first, target.chat(text))


@pytest.mark.parametrize(
    "control",
    ({"use-cache": False}, {"use-cache": "true"}, {"use-cache": 1}, {"ttl": 300}, {"s-maxage": 300}, {}),
    ids=("false", "string-true", "one", "ttl-only", "s-maxage-only", "empty-object"),
)
def test_only_a_literal_true_use_cache_opts_a_request_in(opt_in_rig: CacheRig, control: dict[str, JsonValue]) -> None:
    """For each ``cache`` object that is not exactly ``{"use-cache": true}`` (false, the string "true", the
    number 1, a ``ttl`` or ``s-maxage`` alone, an empty object), two identical requests must both reach the
    upstream. ``should_use_cache`` tests ``use-cache is True``, so truthy look-alikes do not opt in.
    """
    with opt_in_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text, cache=control)
        second: Final = target.chat(text, cache=control)
        assert_miss(first, second)


def test_use_cache_with_a_ttl_applies_the_ttl_to_the_opted_in_entry(opt_in_rig: CacheRig) -> None:
    """A request with ``cache: {"use-cache": true, "ttl": 300}`` fills an entry whose Redis TTL must be above
    60 and at most 300: the other controls keep working once a request has opted in.
    """
    with opt_in_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        store: Final = opt_in_rig.store()
        key: Final = filled(store, target.chat(prompt(), cache={"use-cache": True, "ttl": 300}))
        assert 60 < store.ttl(key) <= 300, store.ttl(key)


def test_use_cache_with_no_store_reads_an_entry_but_writes_none(opt_in_rig: CacheRig) -> None:
    """A request with ``use-cache`` and ``no-store`` reaches the upstream and leaves no entry, so repeating it
    reaches the upstream again. After a plain ``use-cache`` fill, the ``no-store`` form is served from the entry.
    """
    with opt_in_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        read_only: Final[dict[str, JsonValue]] = {"use-cache": True, "no-store": True}
        first: Final = target.chat(text, cache=read_only)
        second: Final = target.chat(text, cache=read_only)
        assert_miss(first, second)
        stored: Final = target.chat(text, cache=USE_CACHE)
        filled(opt_in_rig.store(), stored)
        assert_hit(stored, target.chat(text, cache=read_only))


def test_use_cache_opts_an_embedding_request_in(opt_in_rig: CacheRig) -> None:
    """On ``/v1/embeddings`` a plain request repeated twice reaches the upstream twice, while the same input
    with ``cache: {"use-cache": true}`` is filled once and then served without an upstream call.
    The entry is found by the unique vector the provider returned, since the answer's ``model`` is the
    public model name (``litellm/proxy/common_request_processing.py:3689``) and the entry keeps the
    provider's model (``litellm/caching/caching.py:961``).
    BUG: the opted-in answer is written but never read back. The embedding lookup calls
    ``litellm.cache.async_get_cache`` with only ``cache_key`` and ``dynamic_cache_object``
    (``litellm/caching/caching_handler.py:979``), so ``should_use_cache`` (``litellm/caching/caching.py:1035``)
    never sees the request's ``cache`` object and returns false under ``default_off``, while the write
    (``async_add_cache_pipeline``, ``litellm/caching/caching.py:975``) gets it. In run 43044106414 the entry was
    in Redis and the repeat still reached the upstream.
    """
    pytest.skip("BUG: default_off embeddings with use-cache are written to the cache but never read back")
    with opt_in_rig.gateway.scenario() as scenario:
        answer, marker = unique_embedding()
        target: Final = probe(scenario, answer, model="openai/text-embedding-3-small")
        body: Final[dict[str, JsonValue]] = {"model": target.model, "input": prompt()}
        assert target.send("/v1/embeddings", body).reached_upstream
        assert target.send("/v1/embeddings", body).reached_upstream
        assert entry_key(opt_in_rig.store(), marker) is None
        opted: Final[dict[str, JsonValue]] = {**body, "cache": USE_CACHE}
        first: Final = target.send("/v1/embeddings", opted)
        assert first.reached_upstream, first.response.text
        await_entry(opt_in_rig.store(), marker)
        second: Final = target.send("/v1/embeddings", opted)
        assert not second.reached_upstream and second.json()["data"] == first.json()["data"]


def test_key_metadata_cache_cannot_opt_a_key_in_but_keeps_the_callers_opt_in(opt_in_rig: CacheRig) -> None:
    """A key has ``metadata: {"cache": {"use-cache": true}}``. Its plain request must reach the upstream and
    leave no entry in Redis, since key-level cache metadata only carries ``ttl``, ``s-maxage``, ``no-cache`` and
    ``no-store`` (``SupportedCacheControls``, ``litellm/types/utils.py:181``), so the key cannot opt itself in.
    Its request that sends its own ``cache: {"use-cache": true}`` keeps that opt-in, because the key's cache
    block is merged into the request's instead of replacing it (``_request_cache_kept_for_key``,
    ``litellm/proxy/litellm_pre_call_utils.py:2234``, from #50999): it fills an entry and the repeat is served
    from it.
    The absence of an entry for the plain request is held for a few seconds, since an entry is written by a
    background task after the answer.
    Before #50999 the key's block replaced the caller's, and this cell pinned that the opt-in was dropped;
    run 43314424795 saw the opted-in answer written to Redis after main brought the merge in.
    """
    with opt_in_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        key: Final = scenario.key(models=[target.model], metadata={"cache": {"use-cache": True}})
        text: Final = prompt()
        plain: Final = target.chat(text, key=key)
        assert plain.reached_upstream, plain.response.text
        store: Final = opt_in_rig.store()
        held(lambda: entry_keys_naming(store, plain.id()), lambda found: found == frozenset(), holding=5)
        opted: Final = target.chat(text, key=key, cache=USE_CACHE)
        assert_miss(plain, opted)
        filled(store, opted)
        again: Final = target.chat(text, key=key, cache=USE_CACHE)
        assert_hit(opted, again)
        assert target.calls() == 2


def test_cache_ping_is_healthy_under_default_off(opt_in_rig: CacheRig) -> None:
    """``GET /cache/ping`` on the ``default_off`` proxy must answer 200 with ``status`` ``healthy``,
    ``cache_type`` ``redis`` and ``ping_response`` true: the mode changes who is cached, not whether the
    backend is up.
    """
    response: Final = opt_in_rig.get("/cache/ping")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert (body["status"], body["cache_type"], body["ping_response"]) == ("healthy", "redis", True), body
