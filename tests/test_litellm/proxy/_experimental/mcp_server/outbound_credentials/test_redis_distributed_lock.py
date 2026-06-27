"""Tests for the Redis SET NX PX lock: acquire semantics, release, is_held, error degradation."""

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_distributed_lock import (
    RedisDistributedLock,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)


class _FakeRedis:
    def __init__(self, set_returns=True, exists_returns=1, raise_on=()):
        self._set_returns = set_returns
        self._exists_returns = exists_returns
        self._raise_on = set(raise_on)
        self.set_calls = []
        self.deleted = []

    async def set(self, name, value, *, nx=False, px=None):
        if "set" in self._raise_on:
            raise RuntimeError("redis down")
        self.set_calls.append((name, value, nx, px))
        return self._set_returns

    async def delete(self, *names):
        self.deleted.extend(names)
        return len(names)

    async def exists(self, *names):
        if "exists" in self._raise_on:
            raise RuntimeError("redis down")
        return self._exists_returns


@pytest.mark.asyncio
async def test_acquire_uses_set_nx_px_and_reports_acquired():
    redis = _FakeRedis(set_returns=True)
    lock = RedisDistributedLock(redis)
    assert await lock.acquire("k", 10.0) is LockAcquisition.ACQUIRED
    assert redis.set_calls == [("k", "1", True, 10000)]  # NX + px in milliseconds


@pytest.mark.asyncio
async def test_acquire_reports_held_when_key_already_held():
    # redis SET NX returns None when the key exists -> another worker holds it.
    assert await RedisDistributedLock(_FakeRedis(set_returns=None)).acquire("k", 10.0) is LockAcquisition.HELD


@pytest.mark.asyncio
async def test_acquire_reports_error_on_redis_error_distinct_from_held():
    # A dead backend must be distinguishable from a busy holder so the coordinator refreshes anyway
    # instead of waiting and serving a stale token.
    assert await RedisDistributedLock(_FakeRedis(raise_on=["set"])).acquire("k", 10.0) is LockAcquisition.ERROR


@pytest.mark.asyncio
async def test_release_deletes_the_key():
    redis = _FakeRedis()
    await RedisDistributedLock(redis).release("k")
    assert redis.deleted == ["k"]


@pytest.mark.asyncio
async def test_is_held_reflects_exists():
    assert await RedisDistributedLock(_FakeRedis(exists_returns=1)).is_held("k") is True
    assert await RedisDistributedLock(_FakeRedis(exists_returns=0)).is_held("k") is False


@pytest.mark.asyncio
async def test_is_held_degrades_to_false_on_redis_error():
    assert await RedisDistributedLock(_FakeRedis(raise_on=["exists"])).is_held("k") is False
