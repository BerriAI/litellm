"""Concurrency limits on a virtual key (``max_parallel_requests``).

The contract: a key created with ``max_parallel_requests=N`` may have N requests in flight at once. A request
that arrives while N are in flight is refused with 429 before the provider is called, with the same error
shape as the rpm refusal but ``Limit type: max_parallel_requests`` and ``rate_limit_type:
max_parallel_requests``. A slot is taken when a request is admitted and given back when it ends, whether it
succeeded, failed at the provider, was refused later in the same pre-call check (tokens), or finished
streaming. A refused request never holds a slot. Unlike rpm, the gauge has no window: it only counts what is
running now.

Served responses carry ``x-ratelimit-api_key-limit-max_parallel_requests`` and
``x-ratelimit-api_key-remaining-max_parallel_requests`` (limit minus in-flight, this request included).
In Redis the gauge is the sorted set ``{api_key:<hash>}:max_parallel_requests`` with one member per in-flight
request.

Read from ``litellm/proxy/hooks/parallel_request_limiter_v3.py`` (``_check_parallel_request_gauges``,
``_release_stashed_parallel_slot``) and ``litellm-rust/crates/redis-store/src/rate_limits/lua.rs``.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest

from tests.integration._support.client import Gateway, eventually
from tests.integration._support.wire import wire_server
from tests.integration.authorization.dec1_limits_support import (
    WINDOW_SECONDS,
    assert_limited,
    chat,
    counter_key,
    hashed,
    posts,
    provider,
    scope_headers,
    shared_redis,
    statuses,
)


def test_a_second_request_is_refused_while_the_keys_only_slot_is_in_use(gateway: Gateway) -> None:
    """A key has max_parallel_requests=1 and one request is held open at the provider. A second request must be
    refused with the 429 for ``api_key``, ``Limit type: max_parallel_requests. Current limit: 1, Remaining: 0``,
    with headers ``rate_limit_type: max_parallel_requests`` and ``retry-after: 60``, and must not reach the
    provider. The held request then completes with 200."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        first: Final = pool.submit(chat, gateway, model, key)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            refused: Final = chat(gateway, model, key)
            assert_limited(refused, "api_key", hashed(key), "max_parallel_requests", 1)
            assert refused.headers["rate_limit_type"] == "max_parallel_requests"
            assert refused.headers["retry-after"] == str(WINDOW_SECONDS)
        finally:
            release.set()
        assert first.result().status_code == 200
        assert len(posts(wire)) == 1


def test_a_key_with_two_slots_runs_two_requests_at_once_and_refuses_the_third(gateway: Gateway) -> None:
    """A key has max_parallel_requests=2 and two requests are held open at the provider. A third must be refused
    with ``Current limit: 2, Remaining: 0``. After release both held requests answer 200 and the provider saw
    exactly two requests."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(2) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=2)
        held: Final = [pool.submit(chat, gateway, model, key) for _ in range(2)]
        try:
            eventually(wire.received.qsize, lambda count: count == 2)
            assert_limited(chat(gateway, model, key), "api_key", hashed(key), "max_parallel_requests", 2)
        finally:
            release.set()
        assert [call.result().status_code for call in held] == [200, 200]
        assert len(posts(wire)) == 2


def test_a_finished_request_gives_its_slot_back(gateway: Gateway) -> None:
    """A key has max_parallel_requests=1 and sends four requests one after another. All four must answer 200:
    the limit is on concurrency, so requests that do not overlap are never refused."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        responses: Final = [chat(gateway, model, key) for _ in range(4)]
        assert statuses(responses) == [200, 200, 200, 200], [response.text for response in responses]
        assert len(posts(wire)) == 4


def test_a_request_the_provider_fails_gives_its_slot_back(gateway: Gateway) -> None:
    """A key has max_parallel_requests=1 and the provider answers 400 to everything. Three sequential requests
    must each relay the provider's 400 and none may be the proxy's 429, so a failed call cannot leak its slot."""
    with wire_server(provider(status=400)) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        responses: Final = [chat(gateway, model, key) for _ in range(3)]
        assert statuses(responses) == [400, 400, 400], [response.text for response in responses]
        assert len(posts(wire)) == 3


def test_a_served_response_reports_the_parallel_limit_and_the_free_slots(gateway: Gateway) -> None:
    """A key has max_parallel_requests=3 and sends one request with nothing else in flight. The 200 must carry
    ``x-ratelimit-api_key-limit-max_parallel_requests: 3`` and
    ``x-ratelimit-api_key-remaining-max_parallel_requests: 2`` (the request itself holds one slot)."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=3)
        served: Final = chat(gateway, model, key)
        assert served.status_code == 200, served.text
        assert scope_headers(served, "api_key") == {
            "limit-max_parallel_requests": "3",
            "remaining-max_parallel_requests": "2",
        }


def test_an_open_stream_holds_its_slot_until_the_stream_ends(gateway: Gateway) -> None:
    """A key has max_parallel_requests=1 and a streamed chat completion has sent its first chunk but not
    finished. A second request must be refused for max_parallel_requests. Once the stream is allowed to end, a
    new request must be served again."""
    finish: Final = threading.Event()
    with (
        wire_server(provider(gate_stream=finish)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        streaming: Final = pool.submit(chat, gateway, model, key, stream=True)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            assert_limited(chat(gateway, model, key), "api_key", hashed(key), "max_parallel_requests", 1)
        finally:
            finish.set()
        assert streaming.result().status_code == 200
        finish.set()
        after: Final = eventually(
            lambda: chat(gateway, model, key, stream=True), lambda response: response.status_code == 200
        )
        assert "[DONE]" in after.text


def test_refused_requests_never_hold_a_slot(gateway: Gateway) -> None:
    """A key has max_parallel_requests=1. While its slot is in use five more requests are refused. As soon as
    the held request ends, the next request must be served: the five refusals must not have queued up slots."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1)
        first: Final = pool.submit(chat, gateway, model, key)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            assert statuses([chat(gateway, model, key) for _ in range(5)]) == [429] * 5
        finally:
            release.set()
        assert first.result().status_code == 200
        assert chat(gateway, model, key).status_code == 200
        assert len(posts(wire)) == 2


def test_the_in_flight_gauge_is_a_redis_sorted_set_with_one_member_per_request(gateway: Gateway) -> None:
    """A key has max_parallel_requests=5 and two requests are held open. The Redis sorted set
    ``{api_key:<hash>}:max_parallel_requests`` must hold exactly 2 members, and 0 once both have finished. Other
    instances count concurrency from this set."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        shared_redis() as redis,
        ThreadPoolExecutor(2) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=5)
        gauge: Final = counter_key("api_key", hashed(key), "max_parallel_requests")
        held: Final = [pool.submit(chat, gateway, model, key) for _ in range(2)]
        try:
            eventually(wire.received.qsize, lambda count: count == 2)
            assert int(redis.zcard(gauge)) == 2
        finally:
            release.set()
        assert [call.result().status_code for call in held] == [200, 200]
        eventually(lambda: int(redis.zcard(gauge)), lambda members: members == 0)


def test_an_rpm_refusal_does_not_take_a_parallel_slot(gateway: Gateway) -> None:
    """A key has rpm_limit=1 and max_parallel_requests=1. Its second request is refused for requests. The Redis
    gauge for the key must then be empty: the window check runs before the slot is taken, so an rpm refusal can
    never strand a slot."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario, shared_redis() as redis:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], rpm_limit=1, max_parallel_requests=1)
        assert chat(gateway, model, key).status_code == 200
        assert_limited(chat(gateway, model, key), "api_key", hashed(key), "requests", 1)
        gauge: Final = counter_key("api_key", hashed(key), "max_parallel_requests")
        eventually(lambda: int(redis.zcard(gauge)), lambda members: members == 0)
        assert len(posts(wire)) == 1


def test_a_token_refusal_gives_back_the_slot_it_had_taken(gateway: Gateway) -> None:
    """A key has max_parallel_requests=1 and tpm_limit=100. A request estimated at 510 tokens takes the slot and
    is then refused for tokens. The very next request (estimate 30) must be served, so the token refusal
    released the slot it had just acquired."""
    with wire_server(provider()) as wire, gateway.scenario() as scenario:
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = scenario.key(models=[model], max_parallel_requests=1, tpm_limit=100)
        assert_limited(chat(gateway, model, key, max_tokens=500), "api_key", hashed(key), "tokens", 100, 100)
        served: Final = chat(gateway, model, key, max_tokens=20)
        assert served.status_code == 200, served.text
        assert len(posts(wire)) == 1


def test_two_keys_each_have_their_own_slots(gateway: Gateway) -> None:
    """Two keys each have max_parallel_requests=1. While the first key's slot is held, the second key must be
    served, because concurrency is counted per key."""
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(2) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        first_key: Final = scenario.key(models=[model], max_parallel_requests=1)
        second_key: Final = scenario.key(models=[model], max_parallel_requests=1)
        first: Final = pool.submit(chat, gateway, model, first_key)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            second: Final = pool.submit(chat, gateway, model, second_key)
            eventually(wire.received.qsize, lambda count: count == 2)
        finally:
            release.set()
        assert [first.result().status_code, second.result().status_code] == [200, 200]


@pytest.mark.parametrize("owner", ["user", "team"])
def test_max_parallel_requests_set_on_a_user_or_team_limits_their_keys(gateway: Gateway, owner: str) -> None:
    """A user (or team) is created with max_parallel_requests=1 and owns a key with no limit of its own. While
    one of the key's requests is held at the provider, a second must be refused with 429. The ``/user/new`` docs
    promise this ("Raises 429 error, if user's parallel requests > x") but the v3 limiter only builds a
    concurrency gauge for the key's own ``max_parallel_requests``."""
    pytest.skip("BUG: max_parallel_requests on a user or team is stored but never enforced by the v3 limiter")
    release: Final = threading.Event()
    with (
        wire_server(provider(hold=release)) as wire,
        gateway.scenario() as scenario,
        ThreadPoolExecutor(1) as pool,
    ):
        model: Final = scenario.model(api_base=f"{wire.url}/v1")
        key: Final = (
            scenario.key(models=[model], user_id=scenario.user(max_parallel_requests=1))
            if owner == "user"
            else scenario.key(team_id=scenario.team(models=[model], max_parallel_requests=1))
        )
        first: Final = pool.submit(chat, gateway, model, key)
        try:
            eventually(wire.received.qsize, lambda count: count == 1)
            assert chat(gateway, model, key).status_code == 429
        finally:
            release.set()
        assert first.result().status_code == 200
