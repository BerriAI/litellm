import gc
import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest

from litellm.caching.redis_cache import RedisCache
from litellm.redis.store_settings import RedisBreakerSettings, RedisStoreSettings
from litellm.redis.stores.connection import RedisStores, connect
from litellm.redis.stores.sensitive_route_pins import SensitiveRoutePinStore
from litellm.rust_bridge._native import NativeRedisBreaker, NativeRedisClient, native_redis_breaker_states

_BREAKER: Final = RedisBreakerSettings(failure_threshold=5, recovery_timeout=60.0, timeout_min_duration=5.0)


def _settings(namespace: str) -> RedisStoreSettings:
    return RedisStoreSettings(
        redis_kwargs={"host": "127.0.0.1", "port": 1}, namespace=namespace, default_ttl=60, breaker=None
    )


def test_concurrent_first_connects_finish_without_deadlocking() -> None:
    namespace: Final = f"concurrent-connect-{uuid.uuid4()}"
    pool: Final = ThreadPoolExecutor(max_workers=8)
    futures: Final = tuple(pool.submit(NativeRedisClient.connect, _settings(namespace)) for _ in range(8))
    try:
        clients: Final = tuple(future.result(timeout=30) for future in futures)
    finally:
        pool.shutdown(wait=False)

    assert all(client is not None for client in clients)


def test_concurrent_lookups_of_one_redis_share_one_breaker_without_deadlocking() -> None:
    gc.collect()
    before: Final = len(native_redis_breaker_states())
    redis_kwargs: Final = {"host": f"shared-breaker-{uuid.uuid4()}.invalid", "port": 6379}
    pool: Final = ThreadPoolExecutor(max_workers=8)
    futures: Final = tuple(pool.submit(NativeRedisBreaker.shared, redis_kwargs, _BREAKER) for _ in range(8))
    try:
        breakers: Final = tuple(future.result(timeout=30) for future in futures)
    finally:
        pool.shutdown(wait=False)

    assert len(native_redis_breaker_states()) == before + 1
    for _ in range(_BREAKER.failure_threshold):
        breakers[0].failed(False)
    assert {breaker.state for breaker in breakers} == {"open"}


def _breaker_states_while_held() -> int:
    breaker: Final = NativeRedisBreaker(_BREAKER)
    assert breaker.state == "closed"
    return len(native_redis_breaker_states())


def test_a_discarded_breaker_leaves_the_process_registry() -> None:
    gc.collect()
    before: Final = len(native_redis_breaker_states())

    while_held: Final = _breaker_states_while_held()
    gc.collect()

    assert while_held == before + 1
    assert len(native_redis_breaker_states()) == before


def test_a_redis_cache_captures_what_connect_needs_at_construction() -> None:
    cache: Final = RedisCache(host="127.0.0.1", port=1, namespace="ns", socket_timeout=0.2)

    assert cache.store_settings is not None
    assert {"host": "127.0.0.1", "port": 1, "socket_timeout": 0.2}.items() <= cache.store_settings.redis_kwargs.items()
    assert cache.store_settings.namespace == "ns"
    assert cache.store_settings.default_ttl == cache.default_ttl


def test_a_redis_cache_connects_the_native_stores() -> None:
    cache: Final = RedisCache(host="127.0.0.1", port=1, namespace=f"from-cache-{uuid.uuid4()}", socket_timeout=0.2)

    assert connect(cache).client is not None


class _UnbuiltRedisCache(RedisCache):
    def __init__(self) -> None:
        pass


def test_a_redis_cache_stand_in_that_skips_the_constructor_connects_nothing() -> None:
    assert connect(_UnbuiltRedisCache()).client is None


def test_no_redis_cache_connects_nothing() -> None:
    assert connect(None).client is None


def test_caches_on_one_server_share_a_breaker_whatever_their_db_or_url() -> None:
    host: Final = f"one-server-{uuid.uuid4()}.invalid"
    by_host: Final = NativeRedisBreaker.shared({"host": host, "port": 6379, "db": 0}, _BREAKER)
    by_url: Final = NativeRedisBreaker.shared({"url": f"redis://:pw@{host.upper()}:6379/1"}, _BREAKER)
    other_port: Final = NativeRedisBreaker.shared({"host": host, "port": 6380}, _BREAKER)

    for _ in range(_BREAKER.failure_threshold):
        by_host.failed(False)

    assert (by_url.state, other_port.state) == ("open", "closed")


@pytest.mark.asyncio
async def test_a_store_admits_through_the_breaker_its_server_opened(caplog: pytest.LogCaptureFixture) -> None:
    redis_kwargs: Final = {"host": f"opened-{uuid.uuid4()}.invalid", "port": 6379}
    opened: Final = NativeRedisBreaker.shared(redis_kwargs, _BREAKER)
    for _ in range(_BREAKER.failure_threshold):
        opened.failed(False)
    client: Final = NativeRedisClient.connect(
        RedisStoreSettings(redis_kwargs={**redis_kwargs, "db": 3}, namespace=None, default_ttl=60, breaker=_BREAKER)
    )
    pins: Final = SensitiveRoutePinStore(RedisStores(client).sensitive_route_pins())

    with caplog.at_level(logging.DEBUG, logger="LiteLLM"):
        await pins.set("tenant", "session", "model", 60.0)

    assert opened.state == "open"
    assert any("circuit breaker is open" in record.getMessage() for record in caplog.records)
