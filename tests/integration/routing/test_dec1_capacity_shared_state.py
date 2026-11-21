"""Model capacity across proxy instances, without Redis, and while Redis is down.

The model-wide window and every priority pool live in the rate-limit store, so two proxy instances on one Redis
share them: a request admitted by one instance is counted by the other, and saturation is read from Redis on every
check (#51519), so a model saturated through one instance enforces priority shares on the other at once. Without
Redis each instance keeps its own in-memory windows and still enforces capacity and saturation locally.

When Redis is configured but unreachable, the saturation read falls back to the worker's own copy with a warning
and the atomic check-and-increment falls back to in-memory counters, so callers are still served and capacity is
still enforced per instance (fail open). With ``general_settings.fail_closed_rate_limit_enforcement: true`` the
request is refused with HTTP 503 and the fixed ``Rate limit enforcement unavailable: ...`` text instead.

Draft PR #51540 moves all of this onto the Rust ``RateLimits`` store and polls the Rust future inline when there is
no Redis, so these are the behaviors that port has to keep.

Read from ``litellm/proxy/hooks/dynamic_rate_limiter_v3.py`` (``_get_saturation_value_from_cache``,
``_check_model_saturation``) and ``litellm/proxy/hooks/parallel_request_limiter_v3.py``
(``atomic_check_and_increment_by_n``, ``_atomic_lua_per_descriptor``, ``_reject_if_rate_limit_unverifiable``).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Final

from pydantic import JsonValue

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.redis_process import OwnedRedis, owned_redis
from tests.integration._support.wire import wire_server
from tests.integration.routing.dec1_capacity_support import (
    MODEL_SCOPE,
    capacity_message,
    capacity_proxy,
    chat,
    entry,
    error_message,
    group_name,
    model_counter,
    posts,
    priority_message,
    provider,
    scope_headers,
    settled,
    shared_redis,
    statuses,
    wait_for_model,
)

UNVERIFIABLE: Final = (
    "Rate limit enforcement unavailable: request counters could not be verified against Redis, and "
    "fail_closed_rate_limit_enforcement is enabled, so the request was rejected to avoid exceeding the "
    "configured rate limit. Retry shortly."
)


@contextmanager
def _pair(
    gateway: Gateway,
    directory: Path,
    model_list: Sequence[Mapping[str, JsonValue]],
    reservation: Mapping[str, JsonValue] | None = None,
) -> Iterator[tuple[Gateway, Gateway]]:
    """Two instances booted from the same config on the shared database and Redis."""
    for name in ("first", "second"):
        (directory / name).mkdir(parents=True, exist_ok=True)
    with (
        capacity_proxy(
            gateway, directory / "first", model_list=model_list, priority_reservation=reservation, shared=True
        ) as first,
        capacity_proxy(
            gateway, directory / "second", model_list=model_list, priority_reservation=reservation, shared=True
        ) as second,
    ):
        yield first, second


@contextmanager
def _on_owned_redis(
    gateway: Gateway,
    directory: Path,
    model_list: Sequence[Mapping[str, JsonValue]],
    general_settings: Mapping[str, JsonValue] | None = None,
    reservation: Mapping[str, JsonValue] | None = None,
) -> Iterator[tuple[Gateway, OwnedRedis]]:
    """A capacity proxy whose only Redis is one this test owns and can stop. Assumes, as the limits family does, that
    the suite's config reads the Redis address from ``REDIS_HOST`` and ``REDIS_PORT``."""
    with (
        owned_redis(directory / "redis") as redis,
        capacity_proxy(
            gateway,
            directory,
            model_list=model_list,
            priority_reservation=reservation,
            general_settings=general_settings,
            environment={"REDIS_HOST": redis.host, "REDIS_PORT": str(redis.port)},
        ) as candidate,
    ):
        yield candidate, redis


def test_capacity_used_on_one_instance_refuses_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """Two instances share Redis and a model with ``rpm: 2``. One request is served by each instance. A third request,
    to either instance, is refused with ``Model capacity reached for <group>. ... Model RPM: 2``."""
    group: Final = group_name("pair")
    with wire_server(provider()) as wire:
        with _pair(gateway, tmp_path, [entry(group, wire.url, rpm=2)]) as (first, second):
            assert statuses([chat(first, group, first.key), chat(second, group, second.key)]) == [200, 200]
            refused: Final = chat(first, group, first.key)
            assert refused.status_code == 429, refused.text
            assert error_message(refused) == capacity_message(group, None, "requests", rpm=2)
            assert len(posts(wire)) == 2


def test_the_model_remaining_header_counts_requests_from_both_instances(gateway: Gateway, tmp_path: Path) -> None:
    """Two instances share Redis and a model with ``rpm: 10``. Alternating requests report model-wide
    ``remaining-requests`` 9, 8, 7, 6, not 9, 9, 8, 8 as two private windows would."""
    group: Final = group_name("countdown")
    with wire_server(provider()) as wire:
        with _pair(gateway, tmp_path, [entry(group, wire.url, rpm=10)]) as (first, second):
            responses: Final = [chat(instance, group, instance.key) for instance in (first, second, first, second)]
            assert statuses(responses) == [200] * 4
            assert [scope_headers(response, MODEL_SCOPE)["remaining-requests"] for response in responses] == [
                "9",
                "8",
                "7",
                "6",
            ]


def test_saturation_reached_through_one_instance_enforces_shares_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """Two instances share Redis, a model with ``rpm: 10`` and ``{"low": 0.2}``. A ``low`` key is served five times
    by the first instance. Its sixth request, sent to the second instance, is refused with the priority-based text
    and ``Model saturation: 50.0%``: the second instance reads saturation from Redis (#51519) instead of its own
    idle copy."""
    group: Final = group_name("saturate")
    with wire_server(provider()) as wire:
        with _pair(gateway, tmp_path, [entry(group, wire.url, rpm=10)], {"low": 0.2}) as (first, second):
            with first.scenario() as scenario:
                key: Final = scenario.key(metadata={"priority": "low"})
                assert statuses([chat(first, group, key) for _ in range(5)]) == [200] * 5
                refused: Final = chat(second, group, key)
                assert refused.status_code == 429, refused.text
                assert error_message(refused) == priority_message(group, "low", "requests", "50.0%", rpm=10)


def test_tokens_settled_on_one_instance_count_toward_capacity_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """Two instances share Redis and a model with ``tpm: 30``. The first instance serves one answer of 18 tokens.
    Once Redis holds 18 settled tokens, the second instance serves one more (18 < 30) and, once that settled too
    (36), refuses the next with ``Model capacity reached for <group>. ... Rate limit type: tokens, Model TPM: 30``."""
    group: Final = group_name("tokens")
    with wire_server(provider()) as wire, shared_redis() as redis:
        with _pair(gateway, tmp_path, [entry(group, wire.url, tpm=30)]) as (first, second):
            assert chat(first, group, first.key).status_code == 200
            settled(redis, model_counter(group, "tokens"), 18)
            assert chat(second, group, second.key).status_code == 200
            settled(redis, model_counter(group, "tokens"), 36)
            refused: Final = chat(second, group, second.key)
            assert refused.status_code == 429, refused.text
            assert error_message(refused) == capacity_message(group, None, "tokens", tpm=30)


def test_a_model_added_through_one_instance_is_limited_on_the_other(gateway: Gateway, tmp_path: Path) -> None:
    """Two instances share the database and Redis. A model with ``rpm: 1`` is created through the first instance's
    ``POST /model/new``. Once the second instance lists it, one request through the first instance uses the window
    and the same model is refused on the second instance with ``Model RPM: 1``."""
    with wire_server(provider()) as wire:
        with _pair(gateway, tmp_path, []) as (first, second), first.scenario() as scenario:
            group: Final = scenario.model(api_base=f"{wire.url}/v1", rpm=1)
            wait_for_model(second, group)
            assert chat(first, group, first.key).status_code == 200
            refused: Final = chat(second, group, second.key)
            assert refused.status_code == 429, refused.text
            assert error_message(refused) == capacity_message(group, None, "requests", rpm=1)


def test_without_redis_capacity_is_enforced_from_memory(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy with no Redis and a model with ``rpm: 2``: 200, 200, then 429 with the capacity text. The in-memory
    path takes a lock instead of a Lua script, and the Rust port polls inline here instead."""
    group: Final = group_name("memory")
    with wire_server(provider()) as wire:
        with capacity_proxy(
            gateway, tmp_path, model_list=[entry(group, wire.url, rpm=2)], without_redis=True
        ) as candidate:
            responses: Final = [chat(candidate, group, candidate.key) for _ in range(3)]
            assert statuses(responses) == [200, 200, 429], [response.text for response in responses]
            assert error_message(responses[2]) == capacity_message(group, None, "requests", rpm=2)


def test_without_redis_saturation_and_priority_shares_are_enforced_from_memory(
    gateway: Gateway, tmp_path: Path
) -> None:
    """A proxy with no Redis, ``rpm: 10`` and ``{"low": 0.2}``. A ``low`` key is served five times and its sixth
    request is refused with the priority-based text and ``Model saturation: 50.0%``: the saturation read uses the
    worker's own counter when there is no Redis."""
    group: Final = group_name("memory-sat")
    with (
        wire_server(provider()) as wire,
        capacity_proxy(
            gateway,
            tmp_path,
            model_list=[entry(group, wire.url, rpm=10)],
            priority_reservation={"low": 0.2},
            without_redis=True,
        ) as candidate,
        candidate.scenario() as scenario,
    ):
        key: Final = scenario.key(metadata={"priority": "low"})
        responses: Final = [chat(candidate, group, key) for _ in range(6)]
        assert statuses(responses) == [200] * 5 + [429], [response.text for response in responses]
        assert error_message(responses[5]) == priority_message(group, "low", "requests", "50.0%", rpm=10)


def test_when_redis_stops_capacity_fails_open_onto_memory_and_still_refuses(gateway: Gateway, tmp_path: Path) -> None:
    """A proxy whose Redis is then stopped, with a model of ``rpm: 2``. Requests are still served (not a 5xx), and the
    third one after the outage is refused with the capacity text from the in-memory window. UNVERIFIED end to end:
    other Redis users on the request path (auth cache, response cache) are assumed to degrade as well, as the limits
    family assumes."""
    group: Final = group_name("outage")
    with wire_server(provider()) as wire:
        with _on_owned_redis(gateway, tmp_path, [entry(group, wire.url, rpm=2)]) as (candidate, redis):
            redis.stop()
            first: Final = eventually(
                lambda: chat(candidate, group, candidate.key),
                lambda response: response.status_code < 500,
                seconds=30,
            )
            assert first.status_code == 200, first.text
            assert chat(candidate, group, candidate.key).status_code == 200
            refused: Final = chat(candidate, group, candidate.key)
            assert refused.status_code == 429, refused.text
            assert error_message(refused) == capacity_message(group, None, "requests", rpm=2)
            redis.start()


def test_fail_closed_capacity_answers_503_when_redis_cannot_verify_the_window(gateway: Gateway, tmp_path: Path) -> None:
    """``general_settings.fail_closed_rate_limit_enforcement: true``, the dynamic limiter on and Redis stopped. A
    request for a capacity-limited model is refused with HTTP 503 whose ``error.message`` is the fixed
    ``Rate limit enforcement unavailable: ...`` text, and the provider is never called."""
    group: Final = group_name("closed")
    with (
        wire_server(provider()) as wire,
        _on_owned_redis(
            gateway,
            tmp_path,
            [entry(group, wire.url, rpm=100)],
            general_settings={"fail_closed_rate_limit_enforcement": True},
        ) as (candidate, redis),
    ):
        redis.stop()
        refused: Final = chat(candidate, group, candidate.key)
        assert refused.status_code == 503, refused.text
        assert error_message(refused) == UNVERIFIABLE
        assert posts(wire) == ()
        redis.start()


def test_fail_closed_capacity_still_serves_and_refuses_normally_while_redis_is_healthy(
    gateway: Gateway, tmp_path: Path
) -> None:
    """``fail_closed_rate_limit_enforcement: true`` with a healthy Redis changes nothing for model capacity: on
    ``rpm: 1`` the first request is served and the second refused with the ordinary capacity 429."""
    group: Final = group_name("closed-healthy")
    with (
        wire_server(provider()) as wire,
        _on_owned_redis(
            gateway,
            tmp_path,
            [entry(group, wire.url, rpm=1)],
            general_settings={"fail_closed_rate_limit_enforcement": True},
        ) as (candidate, _redis),
    ):
        responses: Final = [chat(candidate, group, candidate.key) for _ in range(2)]
        assert statuses(responses) == [200, 429], [response.text for response in responses]
        assert error_message(responses[1]) == capacity_message(group, None, "requests", rpm=1)
