"""The cache admin routes on a Redis backed proxy: ``/cache/ping``, ``/cache/delete``, ``/cache/flushall``
and ``/cache/redis/info``, with their authentication.

One owned proxy runs the Redis response cache against an owned Redis, so ``/cache/flushall`` can really
flush it. Contract read from ``litellm/proxy/caching_routes.py``:

``GET /cache/ping`` 200 ``{"status": "healthy", "cache_type": "redis", "ping_response": true,
"set_cache_response": "success", "litellm_cache_params": "<json string>", "health_check_cache_params":
{host, port, ...}}`` and, as a side effect, writes a probe entry; ``POST /cache/delete`` with
``{"keys": [...]}`` deletes those Redis keys and answers ``{"status": "success"}``, any failure is a 500
"Cache Delete Failed(...)"; ``POST /cache/flushall`` runs ``FLUSHALL`` and answers ``{"status":
"success"}``; ``GET /cache/redis/info`` returns ``num_clients``, ``clients`` and ``info``. All four depend
on ``user_api_key_auth``.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from typing import Final

import httpx
import pytest

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, object_value
from tests.integration._support.redis_process import owned_redis
from tests.integration.caching.dec1_cache_support import (
    SETTLE_SECONDS,
    CacheRig,
    cache_config,
    cache_rig,
    entry_keys_naming,
    error_message,
    filled,
    probe,
    redis_params,
)
from tests.integration.caching.response_cache_case import prompt

pytestmark: Final = [pytest.mark.shares_worker_state, pytest.mark.timeout(300)]

ROUTES: Final = (
    pytest.param("GET", "/cache/ping", id="ping"),
    pytest.param("POST", "/cache/delete", id="delete"),
    pytest.param("POST", "/cache/flushall", id="flushall"),
    pytest.param("GET", "/cache/redis/info", id="redis-info"),
)


@pytest.fixture(scope="module")
def redis_rig(session_gateway: Gateway, tmp_path_factory: pytest.TempPathFactory) -> Iterator[CacheRig]:
    directory: Final = tmp_path_factory.mktemp("dec1-cache-routes")
    config: Final = cache_config(directory, cache_params=redis_params())
    with owned_redis(directory) as redis, cache_rig(session_gateway, directory, config, redis=redis) as rig:
        yield rig


def _unauthenticated(rig: CacheRig, method: str, path: str, authorization: str | None) -> httpx.Response:
    headers: Final = {} if authorization is None else {"Authorization": authorization}
    return rig.gateway.client.request(
        method, path, headers=headers, json={"keys": ["a" * 64]} if method == "POST" else None
    )


def test_cache_ping_reports_a_healthy_redis_backend(redis_rig: CacheRig) -> None:
    """``GET /cache/ping`` with the master key must answer 200 with ``status`` ``healthy``, ``cache_type``
    ``redis``, ``ping_response`` true and ``set_cache_response`` ``success``.
    """
    response: Final = redis_rig.get("/cache/ping")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert body["status"] == "healthy" and body["cache_type"] == "redis", body
    assert body["ping_response"] is True and body["set_cache_response"] == "success", body


def test_cache_ping_reports_the_configured_host_and_port(redis_rig: CacheRig) -> None:
    """The ping body's ``health_check_cache_params`` must name the Redis ``host`` and ``port`` the proxy was
    configured with (the owned Redis), which is what the admin UI shows as the cache connection.
    Pinned current behaviour: ``RedisCache`` keeps its connection settings only in ``redis_kwargs``
    (``litellm/caching/redis_cache.py:595``), so the top level ``host`` and ``port`` of
    ``HealthCheckCacheParams`` stay null and the owned Redis is named under ``redis_kwargs``, with the port as
    the string the config resolved from ``os.environ/REDIS_PORT``.
    """
    assert redis_rig.redis is not None
    body: Final = JSON_OBJECT.validate_json(redis_rig.get("/cache/ping").content)
    params: Final = object_value(body["health_check_cache_params"])
    assert (params["host"], params["port"]) == (None, None), params
    connection: Final = object_value(params["redis_kwargs"])
    assert connection["host"] == redis_rig.redis.host, params
    assert str(connection["port"]) == str(redis_rig.redis.port), params


def test_cache_ping_returns_the_cache_settings_as_a_json_string(redis_rig: CacheRig) -> None:
    """The ping body's ``litellm_cache_params`` must be a string holding a JSON object with the cache ``type``
    ``redis`` and the ``supported_call_types`` list, and must not embed the backend object itself (the
    ``cache`` attribute is removed before dumping).
    """
    body: Final = JSON_OBJECT.validate_json(redis_rig.get("/cache/ping").content)
    assert isinstance(body["litellm_cache_params"], str), body
    params: Final = object_value(json.loads(body["litellm_cache_params"]))
    assert params["type"] == "redis", params
    assert "acompletion" in json.dumps(params["supported_call_types"]), params
    assert "cache" not in params, sorted(params)


def test_cache_ping_writes_a_probe_entry_to_redis(redis_rig: CacheRig) -> None:
    """A ping is more than a ``PING``: it also stores a probe response (``result="test_key"`` for model
    ``test-model``). After a ping, a string key in the proxy's Redis must hold a value containing
    ``test_key``, and it must expire (TTL above 0).
    """
    assert redis_rig.get("/cache/ping").status_code == 200
    store: Final = redis_rig.store()
    keys: Final = eventually(lambda: entry_keys_naming(store, "test_key"), lambda found: len(found) > 0, SETTLE_SECONDS)
    assert all(store.ttl(key) > 0 for key in keys), {key: store.ttl(key) for key in keys}


@pytest.mark.parametrize(("method", "path"), ROUTES)
def test_cache_routes_refuse_a_request_without_credentials(redis_rig: CacheRig, method: str, path: str) -> None:
    """Each cache route is called with no ``Authorization`` header. Every one must answer 401, and
    ``/cache/flushall`` in particular must not have flushed anything.
    """
    response: Final = _unauthenticated(redis_rig, method, path, None)
    assert response.status_code == 401, f"{method} {path}: {response.status_code} {response.text}"


@pytest.mark.parametrize(("method", "path"), ROUTES)
def test_cache_routes_refuse_an_unknown_key(redis_rig: CacheRig, method: str, path: str) -> None:
    """Each cache route is called with a bearer token that is not a key of this proxy. Every one must answer
    401.
    """
    response: Final = _unauthenticated(redis_rig, method, path, "Bearer sk-dec1-not-a-real-key")
    assert response.status_code == 401, f"{method} {path}: {response.status_code} {response.text}"


@pytest.mark.parametrize(("method", "path"), ROUTES)
def test_cache_routes_refuse_an_ordinary_virtual_key(redis_rig: CacheRig, method: str, path: str) -> None:
    """A plain virtual key (no admin role) calls each cache route. Every one must be refused with 401 or 403,
    and the entry cached beforehand must still be in Redis: only a proxy admin may inspect, delete from or
    flush the shared cache.
    UNVERIFIED: no ``/cache/*`` route appears in the non-admin allow-lists of ``litellm/proxy/_types.py``, so
    the generic "Only proxy admin can be used..." 401 is expected; confirm in ``auth/route_checks.py``.
    """
    with redis_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        entry: Final = filled(redis_rig.store(), target.chat(prompt()))
        key: Final = scenario.key(models=[target.model])
        response: Final = _unauthenticated(redis_rig, method, path, f"Bearer {key}")
        assert response.status_code in (401, 403), f"{method} {path}: {response.status_code} {response.text}"
        assert redis_rig.store().exists(entry) == 1


def test_cache_delete_removes_exactly_the_named_entries(redis_rig: CacheRig) -> None:
    """Three entries exist. ``POST /cache/delete`` with two of their keys must answer 200
    ``{"status": "success"}``, those two keys must be gone from Redis and the third must remain.
    """
    store: Final = redis_rig.store()
    with redis_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        keys: Final = [filled(store, target.chat(prompt())) for _ in range(3)]
        assert redis_rig.gateway.post("/cache/delete", {"keys": keys[:2]}) == {"status": "success"}
        assert [store.exists(key) for key in keys] == [0, 0, 1]


def test_cache_delete_of_a_key_that_does_not_exist_succeeds(redis_rig: CacheRig) -> None:
    """``POST /cache/delete`` with a well-formed key that is not in Redis must still answer 200
    ``{"status": "success"}``: deleting is idempotent.
    """
    assert redis_rig.gateway.post("/cache/delete", {"keys": ["f" * 64]}) == {"status": "success"}


@pytest.mark.parametrize(
    "body",
    ({}, {"keys": None}, {"keys": []}),
    ids=("no-keys-field", "null-keys", "empty-list"),
)
def test_cache_delete_without_any_key_fails_with_500_and_deletes_nothing(
    redis_rig: CacheRig, body: dict[str, None | list[str]]
) -> None:
    """``POST /cache/delete`` with no ``keys`` field, ``keys: null`` or an empty list must answer 500 with
    "Cache Delete Failed" and must leave an existing entry in place; it must never be read as "delete all".
    UNVERIFIED for the empty list: Redis ``DEL`` with no arguments is an error, which the route maps to 500.
    """
    store: Final = redis_rig.store()
    with redis_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        entry: Final = filled(store, target.chat(prompt()))
        response: Final = redis_rig.gateway.request("POST", "/cache/delete", dict(body))
        assert response.status_code == 500, response.text
        assert "Cache Delete Failed" in error_message(response), response.text
        assert store.exists(entry) == 1


def test_cache_delete_with_a_body_that_is_not_json_is_refused_with_400(redis_rig: CacheRig) -> None:
    """``POST /cache/delete`` with the raw body ``not json`` must be refused as a bad request, never an
    unhandled error page. Authentication parses the body before the route runs
    (``litellm/proxy/common_utils/http_parsing_utils.py:345``), so the caller gets a 400
    ``invalid_request_error`` "Invalid JSON payload" rather than the route's own 500.
    """
    response: Final = redis_rig.gateway.client.post(
        "/cache/delete",
        content=b"not json",
        headers={"Authorization": f"Bearer {redis_rig.gateway.key}", "Content-Type": "application/json"},
    )
    assert response.status_code == 400, response.text
    assert "Invalid JSON payload" in error_message(response), response.text
    error: Final = object_value(JSON_OBJECT.validate_json(response.content)["error"])
    assert (error["type"], error["code"]) == ("invalid_request_error", "400"), response.text


def test_cache_delete_makes_a_later_request_reach_the_upstream(redis_rig: CacheRig) -> None:
    """An entry is filled with a 3 second TTL and deleted through ``/cache/delete``. The identical request
    must reach the upstream again at the latest when the worker's own in-memory copy lapses (the route
    deletes from Redis only), and must then return a new id.
    """
    store: Final = redis_rig.store()
    with redis_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text, cache={"ttl": 3})
        entry: Final = filled(store, first)
        redis_rig.gateway.post("/cache/delete", {"keys": [entry]})
        assert store.exists(entry) == 0
        again: Final = eventually(lambda: target.chat(text), lambda exchange: exchange.reached_upstream, SETTLE_SECONDS)
        assert again.id() != first.id(), again.response.text


def test_get_on_cache_delete_is_method_not_allowed(redis_rig: CacheRig) -> None:
    """``GET /cache/delete`` must answer 405: deleting is POST only, so a crawler or a browser prefetch with
    admin credentials cannot delete anything.
    """
    response: Final = redis_rig.get("/cache/delete")
    assert response.status_code == 405, response.text


def test_cache_redis_info_returns_clients_and_server_info(redis_rig: CacheRig) -> None:
    """``GET /cache/redis/info`` must answer 200 with ``num_clients`` of at least 1, a ``clients`` list of
    that length and an ``info`` object that has a ``redis_version`` and the owned Redis' ``tcp_port``.
    """
    assert redis_rig.redis is not None
    response: Final = redis_rig.get("/cache/redis/info")
    assert response.status_code == 200, response.text
    body: Final = JSON_OBJECT.validate_json(response.content)
    assert sorted(body) == ["clients", "info", "num_clients"], sorted(body)
    assert isinstance(body["num_clients"], int) and body["num_clients"] >= 1, body["num_clients"]
    assert isinstance(body["clients"], list) and len(body["clients"]) == body["num_clients"]
    info: Final = object_value(body["info"])
    assert info["redis_version"] and int(str(info["tcp_port"])) == redis_rig.redis.port, info


def test_cache_flushall_empties_the_response_cache(redis_rig: CacheRig) -> None:
    """Two entries exist. ``POST /cache/flushall`` must answer 200 ``{"status": "success"}`` and both entry
    keys must be gone from Redis. It runs last in this module because it also clears the proxy's other
    Redis state.
    """
    store: Final = redis_rig.store()
    with redis_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        entries: Final = [filled(store, target.chat(prompt())) for _ in range(2)]
        assert redis_rig.gateway.post("/cache/flushall", {}) == {"status": "success"}
        assert [store.exists(entry) for entry in entries] == [0, 0]


def test_the_proxy_keeps_serving_and_caching_after_a_flushall(redis_rig: CacheRig) -> None:
    """After ``/cache/flushall`` the proxy must still authenticate the master key, answer a chat request,
    write its entry to Redis and serve the repeat from it; flushing removes data, it does not break the
    cache.
    """
    store: Final = redis_rig.store()
    assert redis_rig.gateway.post("/cache/flushall", {}) == {"status": "success"}
    with redis_rig.gateway.scenario() as scenario:
        target: Final = probe(scenario)
        text: Final = prompt()
        first: Final = target.chat(text)
        filled(store, first)
        second: Final = target.chat(text)
        assert not second.reached_upstream and second.id() == first.id(), second.response.text
