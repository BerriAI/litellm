import json
from collections.abc import Iterator
from datetime import timedelta
from typing import Final
from unittest.mock import patch

import fakeredis
import pytest
from redis.exceptions import TimeoutError as RedisTimeoutError

from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.redis.stores.agent_identity_bindings import TransportAgentIdentityBindingsStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache
from tests.unit.proxy.agent_endpoints.test_identity_store import setup_store


@pytest.fixture(autouse=True)
def fixed_clock() -> Iterator[None]:
    with patch("time.time", return_value=1700000000.0):
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [b'"agent-\\u00e9"', b"'agent-\\u00e9'"])
async def test_old_binding_writer_new_caller_preserves_escaped_identity(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, wire: bytes
) -> None:
    redis: Final = fake_redis_cache(server, namespace="auth")
    cache: Final = UserApiKeyCache(redis_cache=redis, default_in_memory_ttl=19)
    store, _, identities, _ = setup_store(cache=cache)
    key: Final = 'agent_identity:["tenant-\\u00e9", "client:one"]'
    await raw.set(f"auth:{key}", wire, ex=31)

    assert await store._bound_agent_id("tenant-é", "client:one") == "agent-é"
    assert await cache.async_get_cache(key, local_only=True) == "agent-é"
    assert identities.find_unique.await_count == 0
    assert await raw.pttl(f"auth:{key}") == 31000


@pytest.mark.asyncio
async def test_new_binding_writer_old_reader_preserves_bytes_and_seconds(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    redis: Final = fake_redis_cache(server, namespace="auth")
    store, _, identities, _ = setup_store(cache=UserApiKeyCache(redis_cache=redis, default_in_memory_ttl=19.9))
    key: Final = 'agent_identity:["tenant-\\u00e9", "client:one"]'

    assert await store._bound_agent_id("tenant-é", "client:one") == "agent-one"
    assert await raw.get(f"auth:{key}") == json.dumps("agent-one").encode()
    assert await redis.async_get_cache(key) == "agent-one"
    assert await raw.pttl(f"auth:{key}") == 19000
    assert await raw.get(f"auth:auth:{key}") is None
    assert identities.find_unique.await_count == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [b"not a literal", b"42", b"null", b"{}", b"\xff"])
async def test_malformed_binding_falls_back_to_database(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis, wire: bytes
) -> None:
    redis: Final = fake_redis_cache(server, namespace="auth")
    store, _, identities, _ = setup_store(cache=UserApiKeyCache(redis_cache=redis))
    key: Final = 'agent_identity:["tenant", "client"]'
    await raw.set(f"auth:{key}", wire)

    assert await store._bound_agent_id("tenant", "client") == "agent-one"
    assert identities.find_unique.await_count == 1
    assert await redis.async_get_cache(key) == "agent-one"


@pytest.mark.asyncio
async def test_binding_outage_keeps_local_fill_and_database_authoritative(server: fakeredis.FakeServer) -> None:
    redis: Final = fake_redis_cache(server)
    store, _, identities, _ = setup_store(cache=UserApiKeyCache(redis_cache=redis))
    server.connected = False

    assert await store._bound_agent_id("tenant", "client") == "agent-one"
    assert await store._bound_agent_id("tenant", "client") == "agent-one"
    assert identities.find_unique.await_count == 1
    failure: Final = await TransportAgentIdentityBindingsStore(redis.transport()).get("tenant", "client")
    assert isinstance(failure, RedisFailure)
    assert failure.kind in ("transport", "breaker_open")


@pytest.mark.asyncio
async def test_nonpositive_binding_ttl_does_not_replace_old_redis_value(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    redis: Final = fake_redis_cache(server)
    key: Final = 'agent_identity:["tenant", "client"]'
    await raw.set(key, b'"old-agent"')
    store: Final = TransportAgentIdentityBindingsStore(redis.transport())

    failure: Final = await store.put("tenant", "client", "new-agent", ttl=timedelta(seconds=0.9))
    assert isinstance(failure, RedisFailure)
    assert failure.kind == "server"
    assert await raw.get(key) == b'"old-agent"'


@pytest.mark.asyncio
async def test_binding_unknown_applied_write_keeps_local_fill_and_cross_pod_read(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    redis: Final = fake_redis_cache(server, namespace="auth")
    store, _, identities, _ = setup_store(cache=UserApiKeyCache(redis_cache=redis, default_in_memory_ttl=19))
    key: Final = 'agent_identity:["tenant", "client"]'

    async def applied_then_timeout(*args: object, **kwargs: object) -> None:
        await raw.set(f"auth:{key}", b'"agent-one"', ex=19)
        raise RedisTimeoutError("reply lost after applying SET")

    with patch.object(redis.init_async_client(), "set", side_effect=applied_then_timeout):
        assert await store._bound_agent_id("tenant", "client") == "agent-one"

    assert await store._bound_agent_id("tenant", "client") == "agent-one"
    assert identities.find_unique.await_count == 1
    other, _, other_identities, _ = setup_store(cache=UserApiKeyCache(redis_cache=redis))
    assert await other._bound_agent_id("tenant", "client") == "agent-one"
    assert other_identities.find_unique.await_count == 0
    assert await raw.pttl(f"auth:{key}") == 19000


@pytest.mark.asyncio
async def test_missing_binding_is_not_negative_cached(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    store, _, identities, _ = setup_store(cache=UserApiKeyCache(redis_cache=fake_redis_cache(server)))
    identities.find_unique.return_value = None

    assert await store._bound_agent_id("tenant", "client") is None
    assert await store._bound_agent_id("tenant", "client") is None
    assert identities.find_unique.await_count == 2
    assert await raw.keys("*") == []
