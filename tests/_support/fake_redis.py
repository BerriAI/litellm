from collections.abc import Awaitable, Callable
from typing import Final, TypeVar

import fakeredis
from pydantic import TypeAdapter

import litellm
from litellm._service_logger import ServiceLogging
from litellm.caching.dual_cache import DualCache
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreaker
from litellm.redis.transport.events import RedisServiceEvents
from litellm.redis.transport.redis_py import RedisPyTransport

_T = TypeVar("_T")


def _admit_directly(call: Callable[[], Awaitable[_T]]) -> Awaitable[_T]:
    return call()


def fake_redis_cache(
    server: fakeredis.FakeServer,
    namespace: str | None = None,
    service_logger: ServiceLogging | None = None,
    client: type[fakeredis.FakeAsyncRedis] = fakeredis.FakeAsyncRedis,
) -> RedisCache:
    cache = RedisCache(
        host="127.0.0.1",
        port=1,
        namespace=namespace,
        socket_timeout=0.1,
        service_logger_obj=service_logger or ServiceLogging(),
    )
    litellm.in_memory_llm_clients_cache.set_cache(
        key=cache._get_async_client_cache_key(),  # pyright: ignore[reportPrivateUsage]  # seeds the per-loop client slot RedisCache reads
        value=client(server=server),
    )
    return cache


def fake_redis_transport(
    client: fakeredis.FakeAsyncRedis,
    *,
    failure_threshold: int,
    events: ServiceLogging | None = None,
    namespace: str | None = None,
) -> RedisPyTransport:
    return RedisPyTransport(
        client=lambda: client,
        namespace=namespace,
        breaker=RedisCircuitBreaker(failure_threshold=failure_threshold, recovery_timeout=3600),
        admit=_admit_directly,
        events=RedisServiceEvents(events or ServiceLogging()),
        multi_get=lambda keys: client.mget(keys),
        server_version=lambda: 7,
        pubsub_client=lambda: client,
        request_engine=lambda: None,
        post_call_engine=lambda: None,
    )


_FLOAT: Final = TypeAdapter(float)
_INT: Final = TypeAdapter(int)
_KEYS: Final = TypeAdapter(list[bytes])
_OBJECT: Final = TypeAdapter(object)


def set_reply_callback(cache: RedisCache, command: str, callback: Callable[..., object]) -> None:
    cache.init_async_client().set_response_callback(command, callback)  # pyright: ignore[reportUnknownMemberType, reportArgumentType]  # redis-py types the callback for decoding only, tests also raise or await in it


async def legacy_increment(cache: RedisCache, key: str, value: float, *, ttl: int | None) -> float:
    return _FLOAT.validate_python(await cache.async_increment(key, value, ttl=ttl))  # pyright: ignore[reportUnknownMemberType]  # legacy RedisCache API is untyped


def legacy_default_ttl(cache: RedisCache) -> int:
    return _INT.validate_python(cache.get_ttl(ttl=None))  # pyright: ignore[reportUnknownMemberType]  # legacy RedisCache API is untyped


async def keys_matching(client: fakeredis.FakeAsyncRedis, pattern: str) -> tuple[bytes, ...]:
    return tuple(_KEYS.validate_python(await client.keys(pattern)))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # fakeredis is untyped


def trip_breaker(cache: RedisCache, failures: int | None = None) -> None:
    breaker: Final = cache._circuit_breaker  # pyright: ignore[reportPrivateUsage]  # tests open the breaker the way RedisCache does after hard failures
    for _ in range(breaker.failure_threshold if failures is None else failures):
        breaker.record_failure()


def async_client(cache: RedisCache) -> fakeredis.FakeAsyncRedis:
    client: Final = cache.init_async_client()  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # legacy RedisCache API is untyped
    assert isinstance(client, fakeredis.FakeAsyncRedis)
    return client


def pin_async_client(cache: RedisCache) -> Callable[[], None]:
    client: Final = async_client(cache)

    def pin() -> None:
        litellm.in_memory_llm_clients_cache.set_cache(  # pyright: ignore[reportUnknownMemberType]  # legacy cache API is untyped
            key=cache._get_async_client_cache_key(),  # pyright: ignore[reportPrivateUsage]  # re-seeds the per-loop client slot RedisCache reads
            value=client,
        )

    return pin


async def legacy_set(cache: RedisCache, key: str, value: object, *, ttl: int | None = None) -> None:
    await cache.async_set_cache(key, value, ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # legacy RedisCache API is untyped


def seed_async_client(cache: RedisCache, client: object) -> None:
    litellm.in_memory_llm_clients_cache.set_cache(  # pyright: ignore[reportUnknownMemberType]  # legacy cache API is untyped
        key=cache._get_async_client_cache_key(),  # pyright: ignore[reportPrivateUsage]  # seeds the per-loop client slot RedisCache reads
        value=client,
    )


async def legacy_get(cache: RedisCache, key: str) -> object:
    return _OBJECT.validate_python(await cache.async_get_cache(key))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # legacy RedisCache API is untyped


def local_set(cache: DualCache, key: str, value: object, *, ttl: int | None = None) -> None:
    cache.in_memory_cache.set_cache(key, value, ttl=ttl)  # pyright: ignore[reportUnknownMemberType]  # legacy InMemoryCache API is untyped


def local_get(cache: DualCache, key: str) -> object:
    return _OBJECT.validate_python(cache.in_memory_cache.get_cache(key))  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # legacy InMemoryCache API is untyped
