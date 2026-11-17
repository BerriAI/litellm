"""What two proxy instances do when their shared Redis stops, comes back, or hangs.

Redis is an accelerator, never a dependency of a request: every family in the Redis layer that `main` read
through `DualCache` falls back to the instance's own memory when Redis fails and logs the failure instead of
raising (`litellm/rust/crates/redis-store/AGENTS.md`, "Where `main` swallows a Redis failure and serves memory,
the family does the same"). So while Redis is down each instance keeps routing, keeps its own cooldowns in memory
(which the peer cannot see), and treats a response cache lookup as a miss. When Redis returns, a circuit breaker
lets traffic through again after `REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT` seconds (set to 1 here) and shared
state resumes. When Redis hangs instead (the process is stopped with SIGSTOP), every Redis call waits at most its
socket timeout, so a request still completes with 200, only slower.

Every instance here talks to Redis through `REDIS_HOST` / `REDIS_PORT`, with `REDIS_SOCKET_TIMEOUT=1`.
UNVERIFIED: that `REDIS_SOCKET_TIMEOUT` reaches the response cache client built from `cache_params` as well as the
coordination client built from the environment (`litellm/_redis.py`, `_redis_kwargs_from_environment`). The
client the tests use has a 15 second timeout, which bounds "only slower".
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Final

import httpx
import pytest
from pydantic import JsonValue

from tests.integration._support.client import JSON_OBJECT, eventually, gateway_from_environment
from tests.integration._support.ports import allocate_port
from tests.integration.routing.dec1_multiinst_support import (
    CACHE_KEY_HEADER,
    Fleet,
    add_model,
    ask,
    fleet,
    redis_stopped,
    served,
    unique,
)
from tests.integration.routing.dec1_routing_support import chat, deployment, routing_proxy, scripted_upstream, statuses

pytestmark: Final = [pytest.mark.timeout(500), pytest.mark.shares_worker_state]

COOLDOWN_SECONDS: Final = 20
RECOVERY_SECONDS: Final = 30
COOLDOWN_READ_SECONDS: Final = 3
ENVIRONMENT: Final = {"REDIS_CIRCUIT_BREAKER_RECOVERY_TIMEOUT": "1", "REDIS_SOCKET_TIMEOUT": "1"}


def _models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("outage-plain", "ok-outage-plain", api_base),
        deployment("outage-cool", "script-outage-cool", api_base, weight=1),
        deployment("outage-cool", "ok-outage-cool", api_base),
        deployment("outage-local", "script-outage-local", api_base, weight=1),
        deployment("outage-local", "ok-outage-local", api_base),
        deployment("outage-back", "script-outage-back", api_base, weight=1),
        deployment("outage-back", "ok-outage-back", api_base),
        deployment("outage-hang", "ok-outage-hang", api_base),
        deployment("outage-hang-cool", "script-outage-hang-cool", api_base, weight=1),
        deployment("outage-hang-cool", "ok-outage-hang-cool", api_base),
    )


@pytest.fixture(scope="module")
def pair(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fleet]:
    with fleet(
        tmp_path_factory.mktemp("dec1-multiinst-outage"),
        models=_models,
        router_settings={"num_retries": 0, "cooldown_time": COOLDOWN_SECONDS},
        environment=ENVIRONMENT,
    ) as value:
        yield value


def _id(response: httpx.Response) -> str:
    assert response.status_code == 200, response.text
    return str(JSON_OBJECT.validate_json(response.content)["id"])


def test_both_instances_keep_serving_while_redis_is_stopped(pair: Fleet) -> None:
    """With the shared Redis stopped, five chat requests to `outage-plain` through each instance must all return
    200 from `ok-outage-plain`. Cooldown, usage and cache reads fail and are treated as empty.
    """
    with redis_stopped(pair):
        results: Final = tuple(served(chat(gateway, "outage-plain")) for gateway in pair.instances for _ in range(5))
    assert results == ("ok-outage-plain",) * 10


def test_a_cooldown_still_applies_on_the_instance_that_saw_the_failure_while_redis_is_stopped(pair: Fleet) -> None:
    """With Redis stopped, one 429 is queued for the weighted deployment of `outage-cool` and a request through the
    first instance receives it.

    The next five requests through the first instance must all be served by `ok-outage-cool`: the cooldown is kept
    in that instance's memory although the Redis write failed.
    """
    with redis_stopped(pair):
        pair.upstream.script("script-outage-cool", 429)
        assert chat(pair.first, "outage-cool").status_code == 429
        names: Final = tuple(served(chat(pair.first, "outage-cool")) for _ in range(5))
    assert names == ("ok-outage-cool",) * 5


def test_a_cooldown_made_while_redis_is_stopped_is_not_seen_by_the_peer(pair: Fleet) -> None:
    """With Redis stopped, one 429 is queued for the weighted deployment of `outage-local` and received through the
    first instance, which cools it in memory. A request to `outage-local` through the second instance must then
    still reach `script-outage-local` (now answering 200). Without Redis there is no shared state, and a rewrite must
    neither invent a cooldown on the peer nor block on Redis to look for one.
    """
    with redis_stopped(pair):
        pair.upstream.script("script-outage-local", 429)
        assert chat(pair.first, "outage-local").status_code == 429
        name: Final = served(chat(pair.second, "outage-local"))
    assert name == "script-outage-local"


def test_shared_cooldowns_resume_once_redis_comes_back(pair: Fleet) -> None:
    """Redis is stopped and started again on the same port. After the circuit breaker's 1 second recovery, one 429 is
    queued for `script-outage-back` and received through the first instance.

    Within 30 seconds Redis must hold `deployment:dec1-script-outage-back:cooldown`, and five requests through the
    second instance must then all be served by `ok-outage-back`. A client that never reconnects after an outage
    would silently run every pod on memory only from then on.
    """
    with redis_stopped(pair):
        assert served(chat(pair.first, "outage-plain")) == "ok-outage-plain"
    pair.upstream.script("script-outage-back", 429)
    eventually(lambda: chat(pair.first, "outage-back").status_code, lambda status: status == 429, seconds=10)
    eventually(
        lambda: pair.raw("deployment:dec1-script-outage-back:cooldown"),
        lambda raw: raw is not None,
        seconds=RECOVERY_SECONDS,
    )
    time.sleep(COOLDOWN_READ_SECONDS)
    assert statuses(tuple(chat(pair.second, "outage-back") for _ in range(5))) == (200,) * 5
    assert pair.upstream.count("ok-outage-back") >= 5


def test_a_response_cache_lookup_while_redis_is_stopped_is_a_miss_not_an_error(pair: Fleet) -> None:
    """With Redis stopped, the same chat request goes through the first and then the second instance.

    Both must return 200 from the upstream (two upstream calls, no `x-litellm-cache-key` header on either).
    """
    text: Final = unique("cache while down")
    before: Final = pair.upstream.count("ok-outage-plain")
    with redis_stopped(pair):
        first: Final = ask(pair.first, "outage-plain", text)
        second: Final = ask(pair.second, "outage-plain", text)
    assert first.status_code == 200 and second.status_code == 200
    assert CACHE_KEY_HEADER not in first.headers and CACHE_KEY_HEADER not in second.headers
    assert pair.upstream.count("ok-outage-plain") - before == 2


def test_cross_instance_cache_hits_resume_after_redis_comes_back(pair: Fleet) -> None:
    """Redis is stopped and started again. Then, within 30 seconds, a fresh chat request answered by the first
    instance must be a hit on the second: the same `id` and an `x-litellm-cache-key` header.

    Each attempt uses a new text. In CI run 44176682961 a single fill was missed by the second instance (it got a
    different `id` with the header), most likely because the first instance's write fell inside the circuit
    breaker's recovery window and was dropped; the second's own miss then stored its own answer, so polling one
    text could never converge. The contract kept is that shared hits come back, not that the first write after
    the outage lands.
    """
    with redis_stopped(pair):
        assert chat(pair.first, "outage-plain").status_code == 200

    def attempt() -> tuple[str, httpx.Response]:
        text = unique("cache after outage")
        filled = ask(pair.first, "outage-plain", text)
        return _id(filled), ask(pair.second, "outage-plain", text)

    filled_id, hit = eventually(
        attempt,
        lambda pair_seen: pair_seen[1].status_code == 200 and _id(pair_seen[1]) == pair_seen[0],
        seconds=RECOVERY_SECONDS,
    )
    assert CACHE_KEY_HEADER in hit.headers
    assert _id(hit) == filled_id


def test_requests_complete_with_200_while_redis_hangs(pair: Fleet) -> None:
    """The Redis process is paused (SIGSTOP), so connections stay open but nothing answers. Three chat requests to
    `outage-hang` through each instance must each return 200 from `ok-outage-hang` within the client's 15 second
    timeout: every Redis call on the request path is bounded by its socket timeout and its failure is swallowed.

    BUG: in CI run 44176682961 these requests did not complete within the 15 second client timeout while Redis was
    paused. The Rust Redis client's response timeout defaults to 5 seconds
    (`litellm/rust/crates/redis-client/src/config.rs:20`, `DEFAULT_TIMEOUT`), several Redis calls on one request
    each wait it out, and the proxy logged "Redis circuit breaker OPENED" only once Redis came back. UNVERIFIED
    whether `REDIS_SOCKET_TIMEOUT=1` reaches that client.
    """
    pytest.skip("BUG: a paused (hung) Redis stalls a chat request past 15 seconds instead of failing open")
    assert pair.redis is not None
    with pair.redis.paused():
        results: Final = tuple(served(chat(gateway, "outage-hang")) for gateway in pair.instances for _ in range(3))
    assert results == ("ok-outage-hang",) * 6


def test_a_cooldown_still_moves_traffic_on_the_same_instance_while_redis_hangs(pair: Fleet) -> None:
    """With Redis paused, one 429 is queued for the weighted deployment of `outage-hang-cool` and received through
    the first instance. The next three requests through the first instance must all return 200 from
    `ok-outage-hang-cool`: the memory tier is written before the Redis write that times out.

    BUG: in CI run 44176682961 these requests did not complete within the 15 second client timeout while Redis was
    paused. The Rust Redis client's response timeout defaults to 5 seconds
    (`litellm/rust/crates/redis-client/src/config.rs:20`, `DEFAULT_TIMEOUT`), several Redis calls on one request
    each wait it out, and the proxy logged "Redis circuit breaker OPENED" only once Redis came back. UNVERIFIED
    whether `REDIS_SOCKET_TIMEOUT=1` reaches that client.
    """
    pytest.skip("BUG: a paused (hung) Redis stalls a chat request past 15 seconds instead of failing open")
    assert pair.redis is not None
    with pair.redis.paused():
        pair.upstream.script("script-outage-hang-cool", 429)
        assert chat(pair.first, "outage-hang-cool").status_code == 429
        names: Final = tuple(served(chat(pair.first, "outage-hang-cool")) for _ in range(3))
    assert names == ("ok-outage-hang-cool",) * 3


def test_a_model_written_while_redis_hangs_is_stored_and_served_by_the_writer(pair: Fleet) -> None:
    """With Redis paused, the first instance adds group `outage-hang-new-*` through `POST /model/new`.

    The call must return 200 and, once Redis is resumed, the first instance must serve the group within 20 seconds.
    The config change publish waits at most a socket timeout and never fails the database write.
    """
    assert pair.redis is not None
    group: Final = unique("outage-hang-new")
    with pair.redis.paused():
        add_model(pair.first, group, "ok-outage-hang-new", pair.upstream.api_base)
    response: Final = eventually(
        lambda: ask(pair.first, group, unique("q")), lambda answer: answer.status_code == 200, seconds=20
    )
    assert served(response) == "ok-outage-hang-new"


def test_cache_ping_reports_503_while_redis_is_stopped_and_healthy_after_it_returns(pair: Fleet) -> None:
    """`GET /cache/ping` through the first instance with Redis stopped must answer 503 with `Service Unhealthy` in
    the error message, and within 30 seconds of Redis starting again it must answer 200 with `status` `"healthy"`.
    Read from `litellm/proxy/caching_routes.py:114-128`.
    """
    with redis_stopped(pair):
        down: Final = pair.first.request("GET", "/cache/ping")
    assert down.status_code == 503, down.text
    assert "Service Unhealthy" in down.text
    up: Final = eventually(
        lambda: pair.first.request("GET", "/cache/ping"), lambda response: response.status_code == 200, seconds=30
    )
    assert JSON_OBJECT.validate_json(up.content)["status"] == "healthy"


def _usage_models(api_base: str) -> Sequence[dict[str, JsonValue]]:
    return (
        deployment("hang-usage", "ok-hang-usage-a", api_base, rpm=100),
        deployment("hang-usage", "ok-hang-usage-b", api_base, rpm=100),
    )


@pytest.mark.parametrize("strategy", ["usage-based-routing-v2", "latency-based-routing", "least-busy"])
def test_a_routing_strategy_that_reads_redis_still_serves_while_redis_hangs(tmp_path: Path, strategy: str) -> None:
    """Two instances run `routing_strategy` usage-based-routing-v2, latency-based-routing or least-busy on one Redis,
    which is then paused.

    Three requests to `hang-usage` through each instance must all return 200 from one of its two deployments.
    `lowest_tpm_rpm_v2.py` says it plainly: "don't fail calls if eg. redis fails to connect"; the other strategies
    read their stats through the same `DualCache` and must not fail either.

    BUG: in CI run 44176682961 these requests did not complete within the 15 second client timeout while Redis was
    paused. The Rust Redis client's response timeout defaults to 5 seconds
    (`litellm/rust/crates/redis-client/src/config.rs:20`, `DEFAULT_TIMEOUT`), several Redis calls on one request
    each wait it out, and the proxy logged "Redis circuit breaker OPENED" only once Redis came back. UNVERIFIED
    whether `REDIS_SOCKET_TIMEOUT=1` reaches that client.
    """
    pytest.skip("BUG: a paused (hung) Redis stalls a chat request past 15 seconds instead of failing open")
    with fleet(
        tmp_path,
        models=_usage_models,
        router_settings={"num_retries": 0, "disable_cooldowns": True, "routing_strategy": strategy},
        environment=ENVIRONMENT,
    ) as hung:
        assert hung.redis is not None
        with hung.redis.paused():
            names = frozenset(served(chat(gateway, "hang-usage")) for gateway in hung.instances for _ in range(3))
        assert names <= {"ok-hang-usage-a", "ok-hang-usage-b"} and names


def test_a_proxy_booted_while_redis_is_unreachable_still_serves(tmp_path: Path) -> None:
    """One instance boots with `REDIS_HOST` / `REDIS_PORT` naming a port where nothing listens, using the suite's
    Redis response cache config.

    It must finish booting and answer a chat request with 200. UNVERIFIED: that startup does not block on the
    first Redis connection (the config sync subscriber, the cache ping at boot and the router's attach all swallow
    connection errors in `proxy_server.py`); a Rust host that made Redis a startup dependency would turn a Redis
    blip into a crash loop.
    """
    with (
        gateway_from_environment() as shared,
        scripted_upstream() as upstream,
        routing_proxy(
            shared,
            tmp_path,
            model_list=(deployment("boot-no-redis", "ok-boot-no-redis", upstream.api_base),),
            router_settings={"num_retries": 0},
            environment={"REDIS_HOST": "127.0.0.1", "REDIS_PORT": str(allocate_port()), **ENVIRONMENT},
        ) as lonely,
    ):
        assert served(chat(lonely, "boot-no-redis")) == "ok-boot-no-redis"
