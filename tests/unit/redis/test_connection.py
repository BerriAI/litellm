from collections.abc import Awaitable, Callable
from typing import Final, TypeVar

import fakeredis
import pytest

from litellm._service_logger import ServiceLogging
from litellm.redis.connection import RedisConnection
from litellm.redis.health import RedisCircuitBreaker
from litellm.redis.transport.contract import RedisFailure
from litellm.redis.transport.events import RedisServiceEvents
from tests._support.fake_redis import fake_redis_cache

_T: Final = TypeVar("_T")


def _admit(call: Callable[[], Awaitable[_T]]) -> Awaitable[_T]:
    return call()


@pytest.mark.asyncio
async def test_connection_without_cache_namespaces_keys_and_records_failures(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    breaker: Final = RedisCircuitBreaker(failure_threshold=1, recovery_timeout=3600)
    events: Final = RedisServiceEvents(ServiceLogging())
    connection: Final = RedisConnection(
        client=lambda: raw,
        pubsub_client=lambda: raw,
        namespace=lambda: "connection",
        breaker=lambda: breaker,
        admit=_admit,
        events=lambda: events,
        multi_get=lambda keys: raw.mget(keys),
        server_version=lambda: 7,
        request_engine=lambda: None,
        post_call_engine=lambda: None,
    )
    transport: Final = connection.transport()
    assert await transport.get("missing") is None
    assert await transport.set("key", b"value", ttl=None) is True
    assert await raw.get("connection:key") == b"value"
    assert await raw.get("key") is None
    assert await transport.mget(("key", "missing")) == (b"value", None)
    server.connected = False
    failed: Final = await transport.get("key")
    assert isinstance(failed, RedisFailure) and failed.kind == "transport"
    assert breaker.state == RedisCircuitBreaker.OPEN
    refused: Final = await transport.get("key")
    assert isinstance(refused, RedisFailure) and refused.kind == "breaker_open"


@pytest.mark.asyncio
async def test_cache_connection_reads_namespace_at_transport_construction(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    cache: Final = fake_redis_cache(server, namespace="before")
    first: Final = cache.transport()
    assert await first.set("key", b"first", ttl=None) is True
    cache.namespace = "after"
    second: Final = cache.transport()
    assert await second.set("key", b"second", ttl=None) is True
    assert await first.get("key") == b"first"
    assert await second.get("key") == b"second"
    assert await raw.mget(("before:key", "after:key")) == [b"first", b"second"]
