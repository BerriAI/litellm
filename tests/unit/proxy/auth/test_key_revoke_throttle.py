import asyncio
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final

import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import TimeoutError as RedisTimeoutError
from starlette.requests import Request

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.constants import KEY_REVOKE_MAX_REQUESTS_PER_SOURCE, KEY_REVOKE_WINDOW_SECONDS
from litellm.proxy.auth.key_revoke_throttle import KeyRevokeRefused, KeyRevokeThrottle

pytestmark = pytest.mark.asyncio


def _request(peer: str, forwarded_for: str | None = None) -> Request:
    headers: Final = () if forwarded_for is None else ((b"x-forwarded-for", forwarded_for.encode()),)
    return Request({"type": "http", "client": (peer, 40000), "headers": list(headers)})


class _CountingRedis:
    def __init__(self) -> None:
        self.counts: dict[str, int] = {}
        self.ttls: dict[str, int] = {}

    def async_register_script(self, script: str) -> Callable[[Sequence[str], Sequence[object]], Awaitable[object]]:
        async def run(keys: Sequence[str], args: Sequence[object]) -> object:
            key: Final = keys[0]
            self.counts[key] = self.counts.get(key, 0) + 1
            self.ttls.setdefault(key, int(str(args[0])))
            return [self.counts[key], self.ttls[key]]

        return run


class _FailingRedis:
    def __init__(self, outcome: object) -> None:
        self.outcome: Final = outcome

    def async_register_script(self, script: str) -> Callable[[Sequence[str], Sequence[object]], Awaitable[object]]:
        async def run(keys: Sequence[str], args: Sequence[object]) -> object:
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome

        return run


def _throttle(
    request: Request,
    settings: Mapping[str, object] | None = None,
    redis: _CountingRedis | _FailingRedis | None = None,
    counters: InMemoryCache | None = None,
) -> KeyRevokeThrottle:
    return KeyRevokeThrottle.from_request(
        request,
        general_settings=settings,
        redis_cache=redis,
        counters=counters if counters is not None else InMemoryCache(),
    )


async def _spend_allowance(throttle: KeyRevokeThrottle) -> tuple[KeyRevokeRefused | None, ...]:
    return tuple([await throttle.check() for _ in range(KEY_REVOKE_MAX_REQUESTS_PER_SOURCE)])


async def test_the_request_past_the_hourly_allowance_is_refused_for_the_whole_window():
    throttle: Final = _throttle(_request("203.0.113.7"))

    assert await _spend_allowance(throttle) == (None,) * KEY_REVOKE_MAX_REQUESTS_PER_SOURCE
    assert await throttle.check() == KeyRevokeRefused(retry_after=KEY_REVOKE_WINDOW_SECONDS)


async def test_redis_shares_one_count_across_workers_and_sets_the_retry_time():
    redis: Final = _CountingRedis()
    first_worker: Final = _throttle(_request("203.0.113.7"), redis=redis)
    second_worker: Final = _throttle(_request("203.0.113.7"), redis=redis)
    half: Final = KEY_REVOKE_MAX_REQUESTS_PER_SOURCE // 2

    admitted: Final = tuple([await worker.check() for worker in (first_worker, second_worker) for _ in range(half)])
    assert admitted == (None,) * (half * 2)
    assert await second_worker.check() == KeyRevokeRefused(retry_after=KEY_REVOKE_WINDOW_SECONDS)
    (key,) = redis.ttls
    redis.ttls[key] = 1234
    assert await first_worker.check() == KeyRevokeRefused(retry_after=1234)


@pytest.mark.parametrize(
    "outcome",
    [
        RedisConnectionError("redis is down"),
        RedisTimeoutError("redis timed out"),
        RedisCircuitBreakerOpenError("circuit open"),
        ConnectionRefusedError("refused"),
        asyncio.TimeoutError(),
        ["not", "a count"],
    ],
    ids=["connection", "timeout", "circuit-open", "socket", "asyncio-timeout", "malformed-reply"],
)
async def test_a_redis_failure_falls_back_to_this_workers_own_count(outcome):
    throttle: Final = _throttle(_request("203.0.113.7"), redis=_FailingRedis(outcome))

    assert await _spend_allowance(throttle) == (None,) * KEY_REVOKE_MAX_REQUESTS_PER_SOURCE
    assert await throttle.check() == KeyRevokeRefused(retry_after=KEY_REVOKE_WINDOW_SECONDS)


async def test_without_trusted_proxy_ranges_clients_behind_one_peer_share_a_bucket():
    counters: Final = InMemoryCache()
    first_client: Final = _throttle(_request("10.0.0.1", "198.51.100.1"), counters=counters)
    second_client: Final = _throttle(_request("10.0.0.1", "198.51.100.2"), counters=counters)

    assert await _spend_allowance(first_client) == (None,) * KEY_REVOKE_MAX_REQUESTS_PER_SOURCE
    assert await second_client.check() == KeyRevokeRefused(retry_after=KEY_REVOKE_WINDOW_SECONDS)


async def test_with_trusted_proxy_ranges_each_forwarded_client_has_its_own_bucket():
    counters: Final = InMemoryCache()
    settings: Final = {"trusted_proxy_ranges": ["10.0.0.0/8"]}
    first_client: Final = _throttle(_request("10.0.0.1", "198.51.100.1"), settings, counters=counters)
    second_client: Final = _throttle(_request("10.0.0.1", "198.51.100.2"), settings, counters=counters)

    assert await _spend_allowance(first_client) == (None,) * KEY_REVOKE_MAX_REQUESTS_PER_SOURCE
    assert await second_client.check() is None
    assert await first_client.check() == KeyRevokeRefused(retry_after=KEY_REVOKE_WINDOW_SECONDS)


async def test_ipv6_addresses_in_one_64_share_a_bucket():
    counters: Final = InMemoryCache()
    settings: Final = {"trusted_proxy_ranges": []}
    first_address: Final = _throttle(_request("2001:db8:1:2::1"), settings, counters=counters)
    same_prefix: Final = _throttle(_request("2001:db8:1:2:ffff:ffff:ffff:ffff"), settings, counters=counters)
    other_prefix: Final = _throttle(_request("2001:db8:1:3::1"), settings, counters=counters)

    assert await _spend_allowance(first_address) == (None,) * KEY_REVOKE_MAX_REQUESTS_PER_SOURCE
    assert await same_prefix.check() == KeyRevokeRefused(retry_after=KEY_REVOKE_WINDOW_SECONDS)
    assert await other_prefix.check() is None
