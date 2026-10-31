from collections.abc import Awaitable, Callable
from typing import TypeVar

import fakeredis

import litellm
from litellm._service_logger import ServiceLogging
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreaker
from litellm.redis.transport.events import RedisServiceEvents
from litellm.redis.transport.redis_py import RedisPyTransport

_T = TypeVar("_T")


def _admit_directly(call: Callable[[], Awaitable[_T]]) -> Awaitable[_T]:
    return call()


def fake_redis_cache(
    server: fakeredis.FakeServer, namespace: str | None = None, service_logger: ServiceLogging | None = None
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
        value=fakeredis.FakeAsyncRedis(server=server),
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
