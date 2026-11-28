"""Rate limits across proxy instances: shared Redis, in-memory only, and Redis outages.

The contract with Redis: two proxy instances that share a Redis also share every rate-limit window and
concurrency gauge. A request admitted by one instance is counted by the other, the ``x-ratelimit-*`` remaining
values reflect the combined traffic, and a window exhausted on one instance refuses on the other with the same
429.

The contract without Redis: a proxy with no Redis keeps the counters in its own memory. Limits are still
enforced on that instance (rpm, tpm reservation and concurrency), nothing is written to Redis, and the
counters are not visible to any other instance.

The contract when Redis is configured but unreachable: by default the limiter fails open onto its in-memory
counters, so callers keep being served and each instance still enforces the limit locally. With
``general_settings.fail_closed_rate_limit_enforcement: true`` a request whose counters cannot be verified is
refused with HTTP 503 instead.

Read from ``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (``_execute_redis_batch_rate_limiter_script``,
``_reject_if_rate_limit_unverifiable``, ``RATE_LIMIT_UNVERIFIABLE_MESSAGE``, ``atomic_check_and_increment_by_n``).
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.process import owned_proxy
from tests.integration._support.redis_process import OwnedRedis, owned_redis
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    DEFAULT_USAGE,
    assert_limited,
    chat,
    counter_key,
    error_message,
    hashed,
    limits_config,
    limits_proxy,
    posts,
    provider,
    redis_integer,
    scope_headers,
    shared_redis,
    statuses,
)

UNVERIFIABLE: Final = (
    "Rate limit enforcement unavailable: request counters could not be verified against Redis, and "
    "fail_closed_rate_limit_enforcement is enabled, so the request was rejected to avoid exceeding the "
    "configured rate limit. Retry shortly."
)


def _warm(peer: Gateway, model: str) -> None:
    """The peer loads models another instance created from the database; wait until it serves this one."""
    eventually(lambda: chat(peer, model, peer.key), lambda response: response.status_code == 200, seconds=60)


@contextmanager
def _proxy_on_its_own_redis(
    gateway: Gateway, directory: Path, general_settings: dict[str, JsonValue] | None = None
) -> Iterator[tuple[Gateway, OwnedRedis]]:
    """A proxy whose only Redis is one this test owns and can stop. Assumes the suite's config reads the Redis
    address from ``REDIS_HOST`` and ``REDIS_PORT``, which the owned proxy gets overridden."""
    with owned_redis(directory / "redis") as redis:
        config: Final = limits_config(directory, general_settings=general_settings)
        with owned_proxy(
            gateway, directory, {"REDIS_HOST": redis.host, "REDIS_PORT": str(redis.port)}, config=config
        ) as candidate:
            yield candidate, redis


def test_requests_admitted_by_one_instance_count_against_the_key_on_the_other(gateway: Gateway, peer: Gateway) -> None:
    """A key has rpm_limit=2. One request is served by the first instance and one by its peer on the same Redis.
    A third request to either instance must be refused with the 429 for ``api_key`` requests, limit 2."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        _warm(peer, model)
        key: Final = scenario.key(models=[model], rpm_limit=2)
        assert statuses([chat(gateway, model, key), chat(peer, model, key)]) == [200, 200]
        assert_limited(chat(peer, model, key), "api_key", hashed(key), "requests", 2)
        assert_limited(chat(gateway, model, key), "api_key", hashed(key), "requests", 2)


def test_the_remaining_request_header_counts_traffic_from_both_instances(gateway: Gateway, peer: Gateway) -> None:
    """A key has rpm_limit=10 and alternates between two instances. The ``remaining-requests`` header must count
    down 9, 8, 7, 6 across the instances, not 9, 9, 8, 8 as two private counters would."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        _warm(peer, model)
        key: Final = scenario.key(models=[model], rpm_limit=10)
        responses: Final = [chat(instance, model, key) for instance in (gateway, peer, gateway, peer)]
        assert statuses(responses) == [200, 200, 200, 200], [response.text for response in responses]
        assert [scope_headers(response, "api_key")["remaining-requests"] for response in responses] == [
            "9",
            "8",
            "7",
            "6",
        ]


def test_tokens_settled_on_one_instance_are_seen_by_the_other(gateway: Gateway, peer: Gateway) -> None:
    """A key has tpm_limit=100. The first instance serves a request billed 18 tokens. Once Redis shows the window
    settled at 18, a request reserving 70 tokens on the peer must be served with ``remaining-tokens: 12``
    (100 - 18 - 70)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        _warm(peer, model)
        key: Final = scenario.key(models=[model], tpm_limit=100)
        assert chat(gateway, model, key, max_tokens=60).status_code == 200
        tokens: Final = counter_key("api_key", hashed(key), "tokens")
        eventually(lambda: redis_integer(redis, tokens), lambda value: value == DEFAULT_USAGE.total, seconds=20)
        served: Final = chat(peer, model, key, max_tokens=60)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-tokens": "100", "remaining-tokens": "12"}


def test_a_slot_held_on_one_instance_refuses_the_key_on_the_other(gateway: Gateway, peer: Gateway) -> None:
    """A key has max_parallel_requests=1 and one request is held open through the first instance. A request to
    the peer must be refused with the 429 for max_parallel_requests, and served once the first has finished."""
    release: Final = threading.Event()
    release.set()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        _warm(peer, model)
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        posts(wire)
        release.clear()
        first: Final = pool.submit(chat, gateway, model, key)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            assert_limited(chat(peer, model, key), "api_key", hashed(key), "max_parallel_requests", 1)
        finally:
            release.set()
        assert first.result().status_code == 200
        served: Final = eventually(lambda: chat(peer, model, key), lambda response: response.status_code == 200)
        assert scope_headers(served, "api_key") == {
            "limit-max_parallel_requests": "1",
            "remaining-max_parallel_requests": "0",
        }


def test_a_team_window_is_shared_by_keys_calling_different_instances(gateway: Gateway, peer: Gateway) -> None:
    """A team has rpm_limit=1 and two keys. The first key is served by one instance; the second key calling the
    peer must be refused with ``Rate limit exceeded for team: <team_id>``."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        _warm(peer, model)
        team: Final = scenario.team(models=[model], rpm_limit=1)
        first: Final = scenario.key(team_id=team)
        second: Final = scenario.key(team_id=team)
        assert chat(gateway, model, first).status_code == 200
        assert_limited(chat(peer, model, second), "team", team, "requests", 1)


def test_a_proxy_without_redis_enforces_rpm_from_memory_and_writes_nothing_to_redis(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy is started with no Redis configured and a key has rpm_limit=2. Three requests must answer 200,
    200, 429 with the usual ``api_key`` refusal and headers counting 1 then 0, and the shared Redis must hold no
    ``{api_key:<hash>}:requests`` key."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, without_redis=True) as candidate,
        candidate.scenario() as scenario,
        shared_redis() as redis,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=2)
        responses: Final = [chat(candidate, model, key) for _ in range(3)]
        assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
        assert [scope_headers(response, "api_key")["remaining-requests"] for response in responses[:2]] == ["1", "0"]
        assert_limited(responses[2], "api_key", hashed(key), "requests", 2)
        assert redis.get(counter_key("api_key", hashed(key), "requests")) is None
        assert len(posts(wire)) == 2


def test_a_proxy_without_redis_still_reserves_tokens_and_holds_slots(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy with no Redis has a key with tpm_limit=100 and max_parallel_requests=1. While a request reserving
    70 tokens is held at the provider, a second must be refused for max_parallel_requests. After release, a
    request too large for the window (510 tokens) must be refused for tokens, all from in-memory state."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        limits_proxy(gateway, tmp_path, without_redis=True) as candidate,
        candidate.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], tpm_limit=100, max_parallel_requests=1)
        first: Final = pool.submit(chat, candidate, model, key, max_tokens=60)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            assert_limited(chat(candidate, model, key), "api_key", hashed(key), "max_parallel_requests", 1)
        finally:
            release.set()
        assert first.result().status_code == 200
        too_large: Final = eventually(
            lambda: chat(candidate, model, key, max_tokens=500),
            lambda response: (
                response.status_code == 429
                and "Limit type: tokens" in response.text
                and "Remaining: 82" in response.text
            ),
        )
        assert_limited(too_large, "api_key", hashed(key), "tokens", 100, remaining=82)


def test_counters_of_a_proxy_without_redis_are_invisible_to_other_instances(gateway: Gateway, tmp_path: Path) -> None:
    """A key with rpm_limit=1 is exhausted on a proxy that has no Redis. The same key must still get its one
    request served by the Redis-backed shared proxy: in-memory counters stay on the instance that made them. The
    isolated proxy runs on the shared database (``shared=True``) so it serves the model and key the shared
    gateway created; only its Redis is removed."""
    with wire_server(provider()) as wire, gateway.scenario() as shared_scenario:
        model: Final = shared_scenario.model(api_base=f"{wire.url}/v1")
        key: Final = shared_scenario.key(models=[model], rpm_limit=1)
        with limits_proxy(gateway, tmp_path, without_redis=True, shared=True) as isolated:
            _warm(isolated, model)
            assert statuses([chat(isolated, model, key), chat(isolated, model, key)]) == [200, 429]
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-requests": "1", "remaining-requests": "0"}


def test_when_redis_goes_away_the_limiter_fails_open_onto_memory_and_keeps_enforcing(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy uses a Redis that is then shut down. A key with rpm_limit=1 created before the outage must still
    get one request served (not a 5xx) and its next request refused with the ordinary 429 for ``api_key``
    requests. UNVERIFIED end to end: the limiter falls back to in-memory counters on a Redis error, but other
    Redis users on the request path (auth cache, response cache) are assumed to degrade as well."""
    with (
        wire_server(provider()) as wire,
        _proxy_on_its_own_redis(gateway, tmp_path) as (candidate, redis),
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        redis.stop()
        served: Final = eventually(
            lambda: chat(candidate, model, key), lambda response: response.status_code != 500, seconds=30
        )
        assert served.status_code == 200, served.text
        refused: Final = chat(candidate, model, key)
        assert_limited(refused, "api_key", hashed(key), "requests", 1)
        redis.start()


def test_fail_closed_enforcement_answers_503_when_redis_cannot_verify_the_counters(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy runs with ``general_settings.fail_closed_rate_limit_enforcement: true`` and its Redis is shut down.
    A request from a key with rpm_limit=5 must be refused with HTTP 503 whose ``error.message`` is the fixed
    text starting ``Rate limit enforcement unavailable:``, and the provider must not be called."""
    settings: Final[dict[str, JsonValue]] = {"fail_closed_rate_limit_enforcement": True}
    with (
        wire_server(provider()) as wire,
        _proxy_on_its_own_redis(gateway, tmp_path, settings) as (candidate, redis),
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=5)
        redis.stop()
        refused: Final = chat(candidate, model, key)
        assert refused.status_code == 503, refused.text
        assert error_message(refused) == UNVERIFIABLE
        assert posts(wire) == ()
        redis.start()


def test_fail_closed_enforcement_changes_nothing_while_redis_is_healthy(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy runs with ``fail_closed_rate_limit_enforcement: true`` and a healthy Redis. A key with rpm_limit=1
    must behave exactly as without the setting: one 200 with the request headers, then the 429 for ``api_key``."""
    with (
        wire_server(provider()) as wire,
        limits_proxy(gateway, tmp_path, general_settings={"fail_closed_rate_limit_enforcement": True}) as candidate,
        candidate.scenario() as scenario,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1)
        served: Final = chat(candidate, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {"limit-requests": "1", "remaining-requests": "0"}
        assert_limited(chat(candidate, model, key), "api_key", hashed(key), "requests", 1)
        assert len(posts(wire)) == 1
