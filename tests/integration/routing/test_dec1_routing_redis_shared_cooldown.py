"""`router_settings.redis_host` and `redis_port`: cooldown state shared between proxy instances.

Two owned proxies boot from the same config, with `num_retries: 0`, `cooldown_time: 45` and a `redis_host`
and `redis_port` that point at one Redis started for this file. A cooldown is written to Redis under the key
`deployment:<deployment id>:cooldown` with a time to live of the cooldown length, and its value is a JSON
object with `exception_received`, `status_code`, `timestamp` and `cooldown_time`. Every instance reads that
key before it picks a deployment (Redis is asked again for a key that missed at most every 10 seconds), so a
deployment one instance cooled down gets no traffic from the other.
Without this, each instance in a fleet has to burn its own failed request before it stops calling a broken
deployment.

Each group pairs a failing deployment with `weight: 1` and a healthy sibling without a weight, so the failing
deployment is always picked until it is in cooldown.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Final

import pytest
from integration.routing.dec1_routing_support import (
    Upstream,
    chat,
    deployment,
    routing_proxy,
    scripted_upstream,
    statuses,
)
from pydantic import JsonValue
from redis import Redis

from tests.integration._support.client import JSON_OBJECT, Gateway, eventually, gateway_from_environment
from tests.integration._support.redis_process import OwnedRedis, owned_redis

pytestmark: Final = [pytest.mark.timeout(300), pytest.mark.shares_worker_state]

COOLDOWN_SECONDS: Final = 45
DEPLOYMENT_COOLDOWN_SECONDS: Final = 7
REDIS_READ_THROTTLE_SECONDS: Final = 11  # DEFAULT_REDIS_BATCH_CACHE_EXPIRY_SECONDS (10) plus a margin


@dataclass(frozen=True, slots=True)
class Fleet:
    first: Gateway
    second: Gateway
    upstream: Upstream
    cache: OwnedRedis

    def cooldown_key(self, deployment_id: str) -> str:
        return f"deployment:{deployment_id}:cooldown"

    def cooldown(self, deployment_id: str) -> tuple[dict[str, JsonValue] | None, int]:
        with Redis(host=self.cache.host, port=self.cache.port) as client:
            raw: Final = client.get(self.cooldown_key(deployment_id))
            ttl: Final = int(client.ttl(self.cooldown_key(deployment_id)))
        return (None if raw is None else JSON_OBJECT.validate_python(json.loads(raw)), ttl)


def _models(api_base: str) -> tuple[dict[str, JsonValue], ...]:
    return (
        deployment("redis-key", "status-429-rediskey", api_base, weight=1),
        deployment("redis-key", "ok-rediskey", api_base),
        deployment("redis-peer", "status-429-redispeer", api_base, weight=1),
        deployment("redis-peer", "ok-redispeer", api_base),
        deployment(
            "redis-own-time",
            "status-429-redisown",
            api_base,
            weight=1,
            cooldown_time=DEPLOYMENT_COOLDOWN_SECONDS,
        ),
        deployment("redis-own-time", "ok-redisown", api_base),
        deployment("redis-clean", "status-400-redisclean", api_base, weight=1),
        deployment("redis-clean", "ok-redisclean", api_base),
    )


@pytest.fixture(scope="module")
def fleet(tmp_path_factory: pytest.TempPathFactory) -> Iterator[Fleet]:
    directory: Final = tmp_path_factory.mktemp("dec1-routing-redis")
    with ExitStack() as stack:
        shared: Final = stack.enter_context(gateway_from_environment())
        upstream: Final = stack.enter_context(scripted_upstream())
        cache: Final = stack.enter_context(owned_redis(directory))
        settings: Final[dict[str, JsonValue]] = {
            "num_retries": 0,
            "cooldown_time": COOLDOWN_SECONDS,
            "redis_host": cache.host,
            "redis_port": cache.port,
        }
        first: Final = stack.enter_context(
            routing_proxy(shared, directory / "first", model_list=_models(upstream.api_base), router_settings=settings)
        )
        second: Final = stack.enter_context(
            routing_proxy(shared, directory / "second", model_list=_models(upstream.api_base), router_settings=settings)
        )
        yield Fleet(first, second, upstream, cache)


def test_a_cooldown_is_written_to_redis_with_the_status_the_router_cooldown_time_and_a_ttl(fleet: Fleet) -> None:
    """The first instance sends one request to a deployment with id `dec1-status-429-rediskey` that answers 429.

    Redis must then hold the key `deployment:dec1-status-429-rediskey:cooldown`. Its JSON value must have
    `status_code` `"429"`, `cooldown_time` 45, a numeric `timestamp` and a string `exception_received`, and
    the key's time to live must be between 1 and 45 seconds. UNVERIFIED: that the value is stored as a JSON
    string rather than another serialisation; see `Health::add_cooldown` in
    `litellm-rust/crates/router-core/src/health.rs` and the Redis cache's `set`.
    """
    response: Final = chat(fleet.first, "redis-key")
    assert response.status_code == 429, response.text
    value, ttl = eventually(
        lambda: fleet.cooldown("dec1-status-429-rediskey"), lambda found: found[0] is not None, seconds=10
    )
    assert value is not None
    assert value["status_code"] == "429"
    assert value["cooldown_time"] == COOLDOWN_SECONDS
    assert isinstance(value["timestamp"], float | int)
    assert isinstance(value["exception_received"], str)
    assert 0 < ttl <= COOLDOWN_SECONDS


def test_a_deployment_cooled_down_by_one_instance_gets_no_traffic_from_the_other(fleet: Fleet) -> None:
    """The first instance gets a 429 from the weighted deployment of group `redis-peer`.

    Once the cooldown key is in Redis and the second instance's Redis read throttle has passed, five requests
    sent to the second instance must all return 200 from the healthy sibling, and the failing upstream must
    have received exactly one request in total. The second instance never saw the failure itself; it only read
    the shared cooldown.
    The router reads cooldowns through `DualCache.async_batch_get_cache`, which asks Redis for a key that missed
    in memory at most once every `DEFAULT_REDIS_BATCH_CACHE_EXPIRY_SECONDS` (10, `litellm/constants.py:162`,
    `litellm/caching/dual_cache.py:311`). A miss the second instance recorded shortly before the cooldown landed
    therefore hides it for up to 10 seconds (run 43044106414 sent the first request to the failing deployment),
    so the test waits that window out before asking the second instance.
    """
    response: Final = chat(fleet.first, "redis-peer")
    assert response.status_code == 429, response.text
    eventually(lambda: fleet.cooldown("dec1-status-429-redispeer")[0], lambda value: value is not None, seconds=10)
    time.sleep(REDIS_READ_THROTTLE_SECONDS)
    assert statuses(tuple(chat(fleet.second, "redis-peer") for _ in range(5))) == (200,) * 5
    assert fleet.upstream.count("status-429-redispeer") == 1
    assert fleet.upstream.count("ok-redispeer") == 5


def test_a_deployments_own_cooldown_time_sets_the_redis_value_and_ttl(fleet: Fleet) -> None:
    """The failing deployment sets `litellm_params.cooldown_time: 7`; the router's `cooldown_time` is 45.

    After its 429 the Redis key `deployment:dec1-status-429-redisown:cooldown` must carry `cooldown_time` 7
    and a time to live between 1 and 7 seconds.
    """
    response: Final = chat(fleet.first, "redis-own-time")
    assert response.status_code == 429, response.text
    value, ttl = eventually(
        lambda: fleet.cooldown("dec1-status-429-redisown"), lambda found: found[0] is not None, seconds=10
    )
    assert value is not None
    assert value["cooldown_time"] == DEPLOYMENT_COOLDOWN_SECONDS
    assert 0 < ttl <= DEPLOYMENT_COOLDOWN_SECONDS


def test_a_failure_that_does_not_cool_down_writes_no_cooldown_key(fleet: Fleet) -> None:
    """The weighted deployment of group `redis-clean` answers 400.

    Three requests through the first instance all return 400, and a request through the second instance also
    returns 400 from the same deployment. Redis must hold no `deployment:dec1-status-400-redisclean:cooldown`
    key afterwards.
    """
    assert statuses(tuple(chat(fleet.first, "redis-clean") for _ in range(3))) == (400,) * 3
    assert chat(fleet.second, "redis-clean").status_code == 400
    assert fleet.cooldown("dec1-status-400-redisclean")[0] is None
    assert fleet.upstream.count("status-400-redisclean") == 4
