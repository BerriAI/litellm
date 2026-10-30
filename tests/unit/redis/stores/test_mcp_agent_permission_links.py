from datetime import timedelta
from typing import Final

import fakeredis
import pytest
import pytest_asyncio
from pytest import MonkeyPatch
from redis.exceptions import ConnectionError as RedisConnectionError

from litellm.caching.redis_batch import active_request_redis_batch, request_redis_batch_scope
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    fill_agent_permission_link,
    get_agent_permission_link,
)
from litellm.redis.stores.mcp_agent_permission_links import TransportMcpAgentPermissionLinkStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache


@pytest_asyncio.fixture(loop_scope="function")
async def link_cache(monkeypatch: MonkeyPatch) -> tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache]:
    monkeypatch.setattr("time.time", lambda: 1000.0)
    server: Final = fakeredis.FakeServer()
    redis: Final = fake_redis_cache(server, namespace="links")
    cache: Final = UserApiKeyCache(redis_cache=redis, default_in_memory_ttl=37.9)
    return server, redis, cache


@pytest.mark.asyncio
@pytest.mark.parametrize("value", ['policy é "quoted"', "__agent_no_mcp_permission__", ""])
async def test_old_writer_new_reader(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache], value: str
) -> None:
    _, redis, cache = link_cache
    await redis.async_set_cache("agent_object_permission_id:u", value, ttl=37.9)
    assert await get_agent_permission_link(cache, "u") == value
    assert await cache.async_get_cache("agent_object_permission_id:u", local_only=True) == value


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "value,wire",
    [(None, b'"__agent_no_mcp_permission__"'), ('policy é "quoted"', b'"policy \\u00e9 \\"quoted\\""')],
)
@pytest.mark.parametrize("ttl", [37.9, 37.999999999])
async def test_new_writer_old_reader_and_whole_second_expiry(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache], value: str | None, wire: bytes, ttl: float
) -> None:
    server, redis, cache = link_cache
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await fill_agent_permission_link(cache, "u", value, ttl=ttl)
    assert await raw.get("links:agent_object_permission_id:u") == wire
    assert await raw.pttl("links:agent_object_permission_id:u") == 37000
    assert await redis.async_get_cache("agent_object_permission_id:u") == (value or "__agent_no_mcp_permission__")
    assert await raw.keys("*") == [b"links:agent_object_permission_id:u"]


@pytest.mark.asyncio
@pytest.mark.parametrize("wire", [b"broken", b"42", b"null", b"{}", b"[]", b"\xff"])
async def test_bad_payload_is_decode_failure_and_caller_cache_miss(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache], wire: bytes
) -> None:
    server, redis, cache = link_cache
    await fakeredis.FakeAsyncRedis(server=server).set("links:agent_object_permission_id:u", wire)
    result: Final = await TransportMcpAgentPermissionLinkStore(redis.transport()).get("u")
    assert isinstance(result, RedisFailure) and result.kind == "decode"
    assert await get_agent_permission_link(cache, "u") is None


@pytest.mark.asyncio
async def test_literal_legacy_string_and_local_non_string(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache],
) -> None:
    server, _, cache = link_cache
    await fakeredis.FakeAsyncRedis(server=server).set("links:agent_object_permission_id:u", b"'legacy'")
    assert await get_agent_permission_link(cache, "u") == "legacy"
    await cache.async_set_cache("agent_object_permission_id:u", {"invalid": True}, local_only=True)
    assert await get_agent_permission_link(cache, "u") is None


@pytest.mark.asyncio
async def test_agent_link_does_not_overwrite_user_link_or_permission_row(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache],
) -> None:
    server, redis, cache = link_cache
    await redis.async_set_cache("user_object_permission_id:u", "user-policy", ttl=60)
    await redis.async_set_cache("object_permission_id:u", {"mcp_servers": ["server"]}, ttl=60)
    await fill_agent_permission_link(cache, "u", "agent-policy", ttl=37.9)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    assert await raw.mget(
        "links:agent_object_permission_id:u", "links:user_object_permission_id:u", "links:object_permission_id:u"
    ) == [b'"agent-policy"', b'"user-policy"', b'{"mcp_servers": ["server"]}']


@pytest.mark.asyncio
async def test_outage_preserves_local_fill_and_open_breaker(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache],
) -> None:
    server, redis, cache = link_cache
    server.connected = False
    assert await get_agent_permission_link(cache, "u") is None
    await fill_agent_permission_link(cache, "u", "policy", ttl=30)
    assert await get_agent_permission_link(cache, "u") == "policy"
    cache.in_memory_cache.delete_cache("agent_object_permission_id:u")
    assert await cache.async_get_cache("agent_object_permission_id:u", local_only=True) is None
    store: Final = TransportMcpAgentPermissionLinkStore(redis.transport())
    for _ in range(10):
        assert isinstance(await store.get("u"), RedisFailure)
    result: Final = await store.get("u")
    assert isinstance(result, RedisFailure) and result.kind == "breaker_open"


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [37.9, 37.999999999])
async def test_request_fill_stays_queued_and_clears_prior_miss(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache], ttl: float
) -> None:
    server, redis, cache = link_cache
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    with request_redis_batch_scope():
        batch: Final = active_request_redis_batch(redis)
        assert batch is not None
        assert await batch.mget(("agent_object_permission_id:u",)) == {"agent_object_permission_id:u": None}
        await redis.async_set_cache("agent_object_permission_id:u", "external", ttl=60)
        assert await get_agent_permission_link(cache, "u") is None
        await fill_agent_permission_link(cache, "u", "policy", ttl=ttl)
        assert await get_agent_permission_link(cache, "u") == "policy"
        assert await raw.get("links:agent_object_permission_id:u") == b'"external"'
        await batch.flush()
        assert await raw.get("links:agent_object_permission_id:u") == b'"policy"'
        assert await raw.pttl("links:agent_object_permission_id:u") == 37000


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0.9, 0.999999999, 0, -1])
async def test_rejected_expiry_keeps_existing_redis_value_and_new_local_value(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache], ttl: float
) -> None:
    _, redis, cache = link_cache
    await redis.async_set_cache("agent_object_permission_id:u", "old", ttl=60)
    await fill_agent_permission_link(cache, "u", "new", ttl=ttl)
    assert await cache.async_get_cache("agent_object_permission_id:u", local_only=True) == ("new" if ttl >= 0 else None)
    assert await redis.async_get_cache("agent_object_permission_id:u") == "old"
    result: Final = await TransportMcpAgentPermissionLinkStore(redis.transport()).fill(
        "u", "new", ttl=timedelta(seconds=int(ttl))
    )
    assert isinstance(result, RedisFailure) and result.kind == "server"


@pytest.mark.asyncio
@pytest.mark.parametrize("ttl", [0.999999999, 0, -1])
async def test_rejected_batched_expiry_keeps_existing_value(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache], ttl: float
) -> None:
    _, redis, cache = link_cache
    await redis.async_set_cache("agent_object_permission_id:u", "old", ttl=60)
    with request_redis_batch_scope() as scope:
        await fill_agent_permission_link(cache, "u", "new", ttl=ttl)
        await scope.flush_all()
    assert await redis.async_get_cache("agent_object_permission_id:u") == "old"


@pytest.mark.asyncio
async def test_unclassified_client_failure_keeps_cache_optional(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache], monkeypatch: MonkeyPatch
) -> None:
    _, redis, cache = link_cache
    client: Final = redis.init_async_client()
    assert isinstance(client, fakeredis.FakeAsyncRedis)
    await client.ping()
    connection: Final = await client.connection_pool.get_connection()
    read_response: Final = connection.read_response

    async def unavailable_reply(**kwargs: object) -> object:
        await read_response(**kwargs)
        raise RuntimeError("client callback failed")

    monkeypatch.setattr(connection, "read_response", unavailable_reply)
    await client.connection_pool.release(connection)
    assert await cache.async_get_cache("agent_object_permission_id:a") is None
    assert await get_agent_permission_link(cache, "a") is None
    await fill_agent_permission_link(cache, "a", "policy", ttl=30)
    assert await get_agent_permission_link(cache, "a") == "policy"


@pytest.mark.asyncio
async def test_lost_set_reply_does_not_roll_back_applied_write(
    link_cache: tuple[fakeredis.FakeServer, RedisCache, UserApiKeyCache], monkeypatch: MonkeyPatch
) -> None:
    server, redis, cache = link_cache
    client: Final = redis.init_async_client()
    assert isinstance(client, fakeredis.FakeAsyncRedis)
    await client.ping()
    connection: Final = await client.connection_pool.get_connection()
    read_response: Final = connection.read_response

    async def lost_reply(**kwargs: object) -> object:
        await read_response(**kwargs)
        raise RedisConnectionError("reply lost after write")

    monkeypatch.setattr(connection, "read_response", lost_reply)
    await client.connection_pool.release(connection)
    await fill_agent_permission_link(cache, "u", "applied", ttl=37.9)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    assert await raw.get("links:agent_object_permission_id:u") == b'"applied"'
    assert await raw.pttl("links:agent_object_permission_id:u") == 37000
    assert await get_agent_permission_link(cache, "u") == "applied"
