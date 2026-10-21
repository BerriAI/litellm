"""Tests for the Redis lock: acquire NX/PX with a token, compare-and-delete release, namespacing."""

import pytest

from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_distributed_lock import (
    RedisDistributedLock,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)


class _FakeRedis:
    """Models just enough Redis to exercise SET NX (token store) and the compare-and-delete EVAL."""

    def __init__(self, set_returns=True, exists_returns=1, raise_on=()):
        self._set_returns = set_returns
        self._exists_returns = exists_returns
        self._raise_on = set(raise_on)
        self.set_calls = []
        self.eval_calls = []
        self.expire_calls = []
        self.deleted = []
        self.store: dict = {}

    async def set(self, name, value, *, nx=False, px=None):
        if "set" in self._raise_on:
            raise RuntimeError("redis down")
        self.set_calls.append((name, value, nx, px))
        if self._set_returns:
            self.store[name] = value
        return self._set_returns

    async def eval(self, script, numkeys, *keys_and_args):
        if "eval" in self._raise_on:
            raise RuntimeError("redis down")
        self.eval_calls.append((numkeys, keys_and_args))
        key, token = keys_and_args[0], keys_and_args[1]
        if len(keys_and_args) == 3:
            ttl_ms = keys_and_args[2]
            if self.store.get(key) == token:
                self.expire_calls.append((key, ttl_ms))
                return 1
            return 0
        if self.store.get(key) == token:  # compare-and-delete: only the owner deletes
            del self.store[key]
            self.deleted.append(key)
            return 1
        return 0

    async def exists(self, *names):
        if "exists" in self._raise_on:
            raise RuntimeError("redis down")
        return self._exists_returns


@pytest.mark.asyncio
async def test_acquire_sets_token_with_nx_px_and_reports_acquired():
    redis = _FakeRedis(set_returns=True)
    assert await RedisDistributedLock(redis).acquire("k", "tok-1", 10.0) is (LockAcquisition.ACQUIRED)
    assert redis.set_calls == [("k", "tok-1", True, 10000)]  # token value, NX, px in ms


@pytest.mark.asyncio
async def test_acquire_reports_held_when_key_already_held():
    # redis SET NX returns None when the key exists -> another worker holds it.
    assert await RedisDistributedLock(_FakeRedis(set_returns=None)).acquire("k", "tok", 10.0) is LockAcquisition.HELD


@pytest.mark.asyncio
async def test_acquire_reports_error_on_redis_error_distinct_from_held():
    # A dead backend must be distinguishable from a busy holder so the coordinator refreshes anyway.
    assert await RedisDistributedLock(_FakeRedis(raise_on=["set"])).acquire("k", "tok", 10.0) is LockAcquisition.ERROR


@pytest.mark.asyncio
async def test_release_deletes_only_when_the_token_matches():
    # Regression: a holder whose lock PX-expired and was re-acquired by another worker must not be
    # able to delete the new holder's lock. release with a stale token is a no-op.
    redis = _FakeRedis()
    lock = RedisDistributedLock(redis)
    await lock.acquire("k", "owner-B", 10.0)  # B currently holds the lock
    await lock.release("k", "owner-A")  # A's stale token
    assert redis.deleted == [] and redis.store.get("k") == "owner-B"  # B's lock survives
    await lock.release("k", "owner-B")  # the real owner releases
    assert redis.deleted == ["k"] and "k" not in redis.store


@pytest.mark.asyncio
async def test_keys_are_namespaced_before_reaching_redis():
    redis = _FakeRedis()
    lock = RedisDistributedLock(redis, namespace_key=lambda key: f"ns:{key}")
    await lock.acquire("k", "tok", 10.0)
    await lock.extend("k", "tok", 10.0)
    await lock.release("k", "tok")
    await lock.is_held("k")
    assert redis.set_calls[0][0] == "ns:k"  # acquire namespaced
    assert redis.eval_calls[0][1][0] == "ns:k"  # extend (EVAL KEYS[1]) namespaced
    assert redis.eval_calls[1][1][0] == "ns:k"  # release (EVAL KEYS[1]) namespaced
    assert redis.deleted == ["ns:k"]


@pytest.mark.asyncio
async def test_extend_refreshes_ttl_only_when_the_token_matches():
    redis = _FakeRedis()
    lock = RedisDistributedLock(redis)
    await lock.acquire("k", "owner-B", 10.0)
    assert await lock.extend("k", "owner-A", 10.0) is False
    assert await lock.extend("k", "owner-B", 10.0) is True
    assert redis.expire_calls == [("k", "10000")]


@pytest.mark.asyncio
async def test_extend_degrades_to_false_on_redis_error():
    assert await RedisDistributedLock(_FakeRedis(raise_on=["eval"])).extend("k", "tok", 10.0) is False


@pytest.mark.asyncio
async def test_is_held_reflects_exists():
    assert await RedisDistributedLock(_FakeRedis(exists_returns=1)).is_held("k") is True
    assert await RedisDistributedLock(_FakeRedis(exists_returns=0)).is_held("k") is False


@pytest.mark.asyncio
async def test_is_held_degrades_to_false_on_redis_error():
    assert await RedisDistributedLock(_FakeRedis(raise_on=["exists"])).is_held("k") is False
