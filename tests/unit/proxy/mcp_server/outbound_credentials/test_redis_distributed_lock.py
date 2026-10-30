"""Tests for the Redis lock: acquire NX/PX with a token, compare-and-delete release, namespacing."""

import fakeredis
import pytest

from litellm.proxy.mcp_server.outbound_credentials.redis_distributed_lock import (
    RedisDistributedLock,
)
from litellm.proxy.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)
from litellm.redis.stores.mcp_credential_refresh import RefreshLockTarget, TransportMcpRefreshLockStore
from tests._support.fake_redis import fake_redis_cache

TARGET = RefreshLockTarget(user_id="u", server_id="s")
KEY = "mcp:refresh_lock:u:s"


@pytest.fixture
def server() -> fakeredis.FakeServer:
    return fakeredis.FakeServer()


@pytest.fixture
def raw(server: fakeredis.FakeServer) -> fakeredis.FakeAsyncRedis:
    return fakeredis.FakeAsyncRedis(server=server)


def _lock(server: fakeredis.FakeServer, namespace: str | None = None) -> RedisDistributedLock:
    return RedisDistributedLock(TransportMcpRefreshLockStore(fake_redis_cache(server, namespace=namespace).transport()))


@pytest.mark.asyncio
async def test_acquire_stores_the_token_under_the_namespace_with_the_ttl(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    assert await _lock(server, namespace="ns").acquire(TARGET, "tok-1", 10.0) is LockAcquisition.ACQUIRED

    assert await raw.keys("*") == [f"ns:{KEY}".encode()]
    assert await raw.get(f"ns:{KEY}") == b"tok-1"
    assert 9_000 < await raw.pttl(f"ns:{KEY}") <= 10_000


@pytest.mark.asyncio
async def test_acquire_reports_held_when_key_already_held(server: fakeredis.FakeServer) -> None:
    lock = _lock(server)
    assert await lock.acquire(TARGET, "tok-a", 10.0) is LockAcquisition.ACQUIRED
    assert await lock.acquire(TARGET, "tok-b", 10.0) is LockAcquisition.HELD


@pytest.mark.asyncio
async def test_acquire_reports_error_on_redis_error_distinct_from_held(server: fakeredis.FakeServer) -> None:
    lock = _lock(server)
    server.connected = False
    assert await lock.acquire(TARGET, "tok", 10.0) is LockAcquisition.ERROR


@pytest.mark.asyncio
async def test_release_deletes_only_when_the_token_matches(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    lock = _lock(server)
    await lock.acquire(TARGET, "owner-B", 10.0)
    await lock.release(TARGET, "owner-A")
    assert await raw.get(KEY) == b"owner-B"
    await lock.release(TARGET, "owner-B")
    assert await raw.exists(KEY) == 0


@pytest.mark.asyncio
async def test_a_lock_the_old_writer_took_is_held_renewed_and_released_by_its_owner(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    await raw.set(f"ns:{KEY}", "owner-old", nx=True, px=10_000)
    lock = _lock(server, namespace="ns")

    assert await lock.is_held(TARGET) is True
    assert await lock.acquire(TARGET, "owner-new", 10.0) is LockAcquisition.HELD
    assert await lock.extend(TARGET, "owner-old", 30.0) is True
    assert 29_000 < await raw.pttl(f"ns:{KEY}") <= 30_000
    await lock.release(TARGET, "owner-old")
    assert await lock.is_held(TARGET) is False


@pytest.mark.asyncio
async def test_extend_refreshes_ttl_only_when_the_token_matches(
    server: fakeredis.FakeServer, raw: fakeredis.FakeAsyncRedis
) -> None:
    lock = _lock(server)
    await lock.acquire(TARGET, "owner-B", 1.0)
    assert await lock.extend(TARGET, "owner-A", 10.0) is False
    assert await raw.pttl(KEY) <= 1_000
    assert await lock.extend(TARGET, "owner-B", 10.0) is True
    assert 9_000 < await raw.pttl(KEY) <= 10_000


@pytest.mark.asyncio
async def test_extend_release_and_is_held_degrade_on_redis_error(server: fakeredis.FakeServer) -> None:
    lock = _lock(server)
    await lock.acquire(TARGET, "tok", 10.0)
    server.connected = False

    assert await lock.extend(TARGET, "tok", 10.0) is False
    await lock.release(TARGET, "tok")
    assert await lock.is_held(TARGET) is False

    server.connected = True
    assert await lock.is_held(TARGET) is True
