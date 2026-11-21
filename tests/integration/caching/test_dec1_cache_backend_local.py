"""``cache_params.type: local``: the in-memory response cache of one proxy process, and the cache routes on it.

One owned single-worker proxy runs with ``litellm_settings.cache: true`` and ``cache_params: {type: local}``.
Entries live in that worker's memory only (``InMemoryCache``), nothing is written to Redis, and the
Redis-only admin routes answer with their documented refusals:

``/cache/ping`` 200 ``{"status": "healthy", "cache_type": "local"}``; ``/cache/delete`` 500 "Cache type
local does not support deleting a key. only `redis` is supported"; ``/cache/flushall`` and
``/cache/redis/info`` 503 "Service Unhealthy (...)" (``litellm/proxy/caching_routes.py``).
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Final

import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually
from tests.integration.caching.dec1_cache_support import (
    SETTLE_SECONDS,
    CacheRig,
    assert_hit,
    cache_config,
    cache_rig,
    entry_keys_naming,
    error_message,
    probe,
)
from tests.integration.caching.response_cache_case import CACHE_KEY, prompt

pytestmark: Final = [pytest.mark.shares_worker_state, pytest.mark.timeout(300)]


@pytest.fixture(scope="module")
def local_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CacheRig]:
    directory: Final = tmp_path_factory.mktemp("dec1-cache-local")
    with cache_rig(session_gateway, directory, cache_config(directory, cache_params={"type": "local"})) as rig:
        yield rig


def test_local_cache_serves_the_second_identical_request_from_memory(local_rig: CacheRig) -> None:
    """On the ``type: local`` proxy two identical chat requests are sent. The second must return the first
    answer's id without an upstream call. With one worker the write and the read share one memory store, so
    the hit is polled only for the background write to land.
    """
    with local_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        assert first.reached_upstream, first.response.text
        second: Final = eventually(
            lambda: target.chat(text), lambda exchange: not exchange.reached_upstream, SETTLE_SECONDS
        )
        assert_hit(first, second)


def test_local_cache_hit_carries_a_sha256_cache_key_header(local_rig: CacheRig) -> None:
    """A hit on the local backend must carry ``x-litellm-cache-key`` as a bare sha256 hex digest, the same key
    format as on Redis, since the key is computed before the backend is chosen.
    """
    with local_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        target.chat(text)
        hit: Final = eventually(lambda: target.chat(text), lambda exchange: not exchange.reached_upstream, SETTLE_SECONDS)
        header: Final = hit.cache_key()
        assert header is not None and CACHE_KEY.match(header), dict(hit.response.headers)


def test_local_cache_writes_no_response_entry_to_redis(local_rig: CacheRig) -> None:
    """After a fill and a hit on the local backend, no string key in the worker's Redis may hold a value that
    names the answer id: the response cache is memory only even though Redis is configured for other state.
    """
    with local_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        eventually(lambda: target.chat(text), lambda exchange: not exchange.reached_upstream, SETTLE_SECONDS)
        assert entry_keys_naming(local_rig.store(), first.id()) == frozenset()


def test_local_cache_entry_with_a_request_ttl_expires(local_rig: CacheRig) -> None:
    """An entry is filled with ``cache: {"ttl": 2}`` on the local backend and hit. Repeating the plain request
    must reach the upstream again within a few seconds: the in-memory store honours the per-request TTL.
    """
    with local_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text, cache={"ttl": 2})
        expired: Final = eventually(
            lambda: target.chat(text, cache={"no-store": True}),
            lambda exchange: exchange.reached_upstream,
            SETTLE_SECONDS,
        )
        assert expired.id() != first.id(), expired.response.text


def test_cache_ping_on_the_local_backend_reports_healthy_and_its_type(local_rig: CacheRig) -> None:
    """``GET /cache/ping`` with the master key on the local backend must answer 200 with ``status`` ``healthy``
    and ``cache_type`` ``local``, and must not claim a Redis ping (``ping_response`` is not ``true``).
    """
    response: Final = local_rig.get("/cache/ping")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert (body["status"], body["cache_type"]) == ("healthy", "local"), body
    assert body.get("ping_response") is not True, body


def test_cache_delete_on_the_local_backend_is_refused_with_500(local_rig: CacheRig) -> None:
    """``POST /cache/delete`` with a key list on the local backend must answer 500 and say that cache type
    ``local`` does not support deleting a key and that only ``redis`` is supported.
    """
    response: Final = local_rig.gateway.request("POST", "/cache/delete", {"keys": ["a" * 64]})
    assert response.status_code == 500, response.text
    message: Final = error_message(response)
    assert "Cache Delete Failed" in message, message
    assert "Cache type local does not support deleting a key. only `redis` is supported" in message, message


def test_cache_flushall_on_the_local_backend_is_refused_with_503(local_rig: CacheRig) -> None:
    """``POST /cache/flushall`` on the local backend must answer 503 with "Service Unhealthy" and "Cache type
    local does not support flushing", and a cached entry must still be served afterwards.
    """
    with local_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        eventually(lambda: target.chat(text), lambda exchange: not exchange.reached_upstream, SETTLE_SECONDS)
        response: Final = local_rig.gateway.request("POST", "/cache/flushall")
        assert response.status_code == 503, response.text
        message: Final = error_message(response)
        assert "Service Unhealthy" in message and "Cache type local does not support flushing" in message, message
        assert_hit(first, target.chat(text))


def test_cache_redis_info_on_the_local_backend_is_refused_with_503(local_rig: CacheRig) -> None:
    """``GET /cache/redis/info`` on the local backend must answer 503 with "Service Unhealthy" and "Cache type
    local does not support redis info".
    """
    response: Final = local_rig.get("/cache/redis/info")
    assert response.status_code == 503, response.text
    message: Final = error_message(response)
    assert "Service Unhealthy" in message and "Cache type local does not support redis info" in message, message
