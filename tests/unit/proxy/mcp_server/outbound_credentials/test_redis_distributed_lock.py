"""Tests for the Redis lock: acquire NX/PX with a token, compare-and-delete release, namespacing."""

from typing import Final, Literal

import fakeredis
import pytest

from litellm.caching.redis_cache import RedisCache
from litellm.constants import REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD
from litellm.proxy.mcp_server.outbound_credentials.redis_distributed_lock import (
    RedisDistributedLock,
)
from litellm.proxy.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)
from litellm.redis.stores.mcp_credential_refresh import RefreshLockTarget, TransportMcpRefreshLockStore
from litellm.redis.transport.contract import RedisFailure
from tests._support.fake_redis import fake_redis_cache
from tests.unit.proxy.mcp_server.outbound_credentials.conftest import lost_write_reply

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


_IDENTITY: Final = RefreshLockTarget("u", "s")
_KEY: Final = "ns:mcp:refresh_lock:u:s"
_OLD_RELEASE: Final = (
    "if redis.call('get', KEYS[1]) == ARGV[1] then return redis.call('del', KEYS[1]) else return 0 end"
)


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace", (None, "ns", "mcp"))
@pytest.mark.parametrize("owner", ('owner "雪" \\ 1.25', "", "123"))
async def test_acquire_preserves_raw_owner_and_fractional_ttl_for_old_readers(
    refresh_server: fakeredis.FakeServer, refresh_raw: fakeredis.FakeAsyncRedis, namespace: str | None, owner: str
) -> None:
    lock: Final = RedisDistributedLock(
        TransportMcpRefreshLockStore(fake_redis_cache(refresh_server, namespace=namespace).transport())
    )
    key: Final = "ns:mcp:refresh_lock:u:s" if namespace == "ns" else "mcp:refresh_lock:u:s"
    assert await lock.acquire(_IDENTITY, owner, 1.2345) is LockAcquisition.ACQUIRED
    assert await refresh_raw.keys("*") == [key.encode()]
    assert await refresh_raw.get(key) == owner.encode()
    assert await refresh_raw.pttl(key) == 1235
    assert await refresh_raw.eval(_OLD_RELEASE, 1, key, owner) == 1
    assert await lock.is_held(_IDENTITY) is False


@pytest.mark.asyncio
async def test_acquire_reports_held_without_overwriting_owner_or_expiry(
    refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis
) -> None:
    assert await refresh_lock.acquire(_IDENTITY, "tok-a", 1.2345) is LockAcquisition.ACQUIRED
    assert await refresh_lock.acquire(_IDENTITY, "tok-b", 10.0) is LockAcquisition.HELD
    assert await refresh_raw.get(_KEY) == b"tok-a"
    assert await refresh_raw.pttl(_KEY) == 1235


@pytest.mark.asyncio
async def test_each_user_server_identity_has_an_independent_lease(
    refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis
) -> None:
    assert await refresh_lock.acquire(_IDENTITY, "first", 10.0) is LockAcquisition.ACQUIRED
    assert await refresh_lock.acquire(RefreshLockTarget("u2", "s"), "second", 10.0) is LockAcquisition.ACQUIRED
    assert await refresh_lock.acquire(RefreshLockTarget("u", "s2"), "third", 10.0) is LockAcquisition.ACQUIRED
    await refresh_lock.release(_IDENTITY, "first")
    assert await refresh_raw.mget(_KEY, "ns:mcp:refresh_lock:u2:s", "ns:mcp:refresh_lock:u:s2") == [
        None,
        b"second",
        b"third",
    ]


@pytest.mark.asyncio
async def test_old_writer_lease_is_held_renewed_and_released_only_by_its_owner(
    refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis
) -> None:
    owner: Final = 'old "雪" \\ 1.25'
    await refresh_raw.set(_KEY, owner, nx=True, px=1235)
    assert await refresh_lock.is_held(_IDENTITY) is True
    assert await refresh_lock.acquire(_IDENTITY, "new", 10.0) is LockAcquisition.HELD
    assert await refresh_lock.extend(_IDENTITY, "new", 30.0) is False
    await refresh_lock.release(_IDENTITY, "new")
    assert await refresh_raw.get(_KEY) == owner.encode()
    assert await refresh_raw.pttl(_KEY) == 1235
    assert await refresh_lock.extend(_IDENTITY, owner, 30.2345) is True
    assert await refresh_raw.pttl(_KEY) == 30235
    await refresh_lock.release(_IDENTITY, owner)
    assert await refresh_lock.is_held(_IDENTITY) is False


@pytest.mark.asyncio
async def test_outage_and_open_breaker_keep_acquisition_error_distinct_from_held(
    refresh_server: fakeredis.FakeServer,
    refresh_cache: RedisCache,
    refresh_lock: RedisDistributedLock,
    refresh_raw: fakeredis.FakeAsyncRedis,
) -> None:
    refresh_server.connected = False
    for _ in range(REDIS_CIRCUIT_BREAKER_FAILURE_THRESHOLD):
        assert await refresh_lock.acquire(_IDENTITY, "tok", 10.0) is LockAcquisition.ERROR
    refresh_server.connected = True
    assert await refresh_lock.acquire(_IDENTITY, "tok", 10.0) is LockAcquisition.ERROR
    failure: Final = await refresh_cache.transport().get("mcp:refresh_lock:u:s")
    assert isinstance(failure, RedisFailure) and failure.kind == "breaker_open"
    assert await refresh_raw.get(_KEY) is None


@pytest.mark.asyncio
async def test_extend_release_and_is_held_degrade_on_redis_error_with_redis_store(
    refresh_server: fakeredis.FakeServer, refresh_lock: RedisDistributedLock, refresh_raw: fakeredis.FakeAsyncRedis
) -> None:
    assert await refresh_lock.acquire(_IDENTITY, "tok", 10.0) is LockAcquisition.ACQUIRED
    refresh_server.connected = False
    assert await refresh_lock.extend(_IDENTITY, "tok", 10.0) is False
    await refresh_lock.release(_IDENTITY, "tok")
    assert await refresh_lock.is_held(_IDENTITY) is False
    refresh_server.connected = True
    assert await refresh_raw.get(_KEY) == b"tok"
    assert await refresh_lock.is_held(_IDENTITY) is True


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("acquire", "extend", "release"))
async def test_lost_write_reply_does_not_claim_the_write_was_rejected(
    refresh_cache: RedisCache,
    refresh_lock: RedisDistributedLock,
    refresh_raw: fakeredis.FakeAsyncRedis,
    operation: Literal["acquire", "extend", "release"],
) -> None:
    client: Final = refresh_cache.init_async_client()
    assert isinstance(client, fakeredis.FakeAsyncRedis)
    if operation == "acquire":
        client.set_response_callback("SET", lost_write_reply)
        assert await refresh_lock.acquire(_IDENTITY, "tok", 1.2345) is LockAcquisition.ERROR
        assert await refresh_raw.get(_KEY) == b"tok"
        assert await refresh_raw.pttl(_KEY) == 1235
        return
    await refresh_raw.set(_KEY, "tok", px=1235)
    client.set_response_callback("EVAL", lost_write_reply)
    client.set_response_callback("EVALSHA", lost_write_reply)
    if operation == "extend":
        assert await refresh_lock.extend(_IDENTITY, "tok", 30.2345) is False
        assert await refresh_raw.get(_KEY) == b"tok"
        assert await refresh_raw.pttl(_KEY) == 30235
        return
    await refresh_lock.release(_IDENTITY, "tok")
    assert await refresh_raw.get(_KEY) is None
