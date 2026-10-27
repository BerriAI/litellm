"""Refresh lock policy over the real cache facade with Redis I/O isolated."""

import asyncio
from collections.abc import Iterator
from typing import Final, Literal
from unittest.mock import ANY, AsyncMock, call, patch

import pytest

from litellm.caching.redis_cache import RedisCache
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_distributed_lock import (
    RedisDistributedLock,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    LockAcquisition,
)


@pytest.fixture
def backend() -> Iterator[tuple[RedisDistributedLock, AsyncMock]]:
    with (
        patch("redis.client.Redis.execute_command", return_value=True),
        patch("redis.asyncio.client.Redis.execute_command", new_callable=AsyncMock) as transport,
    ):
        yield RedisDistributedLock(RedisCache(host="127.0.0.1", port=6379, namespace="ns")), transport


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("reply", "expected"),
    (
        (b"OK", LockAcquisition.ACQUIRED),
        (None, LockAcquisition.HELD),
        (False, LockAcquisition.HELD),
        ("invalid", LockAcquisition.ERROR),
    ),
)
async def test_acquire_distinguishes_ownership_contention_and_invalid_replies(
    backend: tuple[RedisDistributedLock, AsyncMock], reply: object, expected: LockAcquisition
) -> None:
    lock, transport = backend
    transport.return_value = reply
    assert await lock.acquire("k", "tok-1", 10.125) is expected
    transport.assert_awaited_once_with("EVALSHA", ANY, 1, "ns:k", "tok-1", 10125)


@pytest.mark.asyncio
async def test_acquire_reports_error_on_redis_error_distinct_from_held(
    backend: tuple[RedisDistributedLock, AsyncMock],
) -> None:
    lock, transport = backend
    transport.side_effect = OSError("redis down")
    assert await lock.acquire("k", "tok", 10.0) is LockAcquisition.ERROR


@pytest.mark.asyncio
async def test_release_passes_each_holders_token_to_the_atomic_operation(
    backend: tuple[RedisDistributedLock, AsyncMock],
) -> None:
    lock, transport = backend
    transport.side_effect = (0, 1)
    assert await lock.release("k", "stale-owner") is None
    assert await lock.release("k", "current-owner") is None
    assert transport.await_args_list == [
        call("EVALSHA", ANY, 1, "ns:k", "stale-owner"),
        call("EVALSHA", ANY, 1, "ns:k", "current-owner"),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("key", ("k", "ns:k"))
async def test_all_lease_operations_apply_the_namespace_once(
    backend: tuple[RedisDistributedLock, AsyncMock], key: str
) -> None:
    lock, transport = backend
    transport.side_effect = (b"OK", 1, 1, 0)
    assert await lock.acquire(key, "tok", 10.0) is LockAcquisition.ACQUIRED
    assert await lock.extend(key, "tok", 10.0) is True
    assert await lock.release(key, "tok") is None
    assert await lock.is_held(key) is False
    assert transport.await_args_list == [
        call("EVALSHA", ANY, 1, "ns:k", "tok", 10000),
        call("EVALSHA", ANY, 1, "ns:k", "tok", 10000),
        call("EVALSHA", ANY, 1, "ns:k", "tok"),
        call("EVALSHA", ANY, 1, "ns:k"),
    ]


@pytest.mark.asyncio
async def test_extend_preserves_owner_token_and_millisecond_ttl(
    backend: tuple[RedisDistributedLock, AsyncMock],
) -> None:
    lock, transport = backend
    transport.side_effect = (0, 1)
    assert await lock.extend("k", "stale-owner", 10.125) is False
    assert await lock.extend("k", "current-owner", 10.125) is True
    assert transport.await_args_list == [
        call("EVALSHA", ANY, 1, "ns:k", "stale-owner", 10125),
        call("EVALSHA", ANY, 1, "ns:k", "current-owner", 10125),
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize(("reply", "held"), ((1, True), (0, False), ("invalid", False)))
async def test_is_held_maps_existence_and_invalid_replies(
    backend: tuple[RedisDistributedLock, AsyncMock], reply: object, held: bool
) -> None:
    lock, transport = backend
    transport.return_value = reply
    assert await lock.is_held("k") is held


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("extend", "release", "is_held"))
async def test_lease_outages_do_not_block_refresh_or_reread(
    backend: tuple[RedisDistributedLock, AsyncMock], operation: Literal["extend", "release", "is_held"]
) -> None:
    lock, transport = backend
    transport.side_effect = OSError("redis down")
    match operation:
        case "extend":
            assert await lock.extend("k", "tok", 10.0) is False
        case "release":
            assert await lock.release("k", "tok") is None
        case "is_held":
            assert await lock.is_held("k") is False


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("acquire", "extend", "release", "is_held"))
async def test_lease_cancellation_is_not_translated_to_an_outage(
    backend: tuple[RedisDistributedLock, AsyncMock], operation: Literal["acquire", "extend", "release", "is_held"]
) -> None:
    lock, transport = backend
    transport.side_effect = asyncio.CancelledError

    async def invoke() -> None:
        match operation:
            case "acquire":
                await lock.acquire("k", "tok", 10.0)
            case "extend":
                await lock.extend("k", "tok", 10.0)
            case "release":
                await lock.release("k", "tok")
            case "is_held":
                await lock.is_held("k")

    with pytest.raises(asyncio.CancelledError):
        await invoke()


@pytest.mark.asyncio
async def test_lease_apis_warn_as_experimental_at_the_lock_callsite(
    backend: tuple[RedisDistributedLock, AsyncMock],
) -> None:
    lock, transport = backend
    transport.return_value = b"OK"
    with pytest.warns(FutureWarning, match="async_acquire_lease is experimental and may change") as seen:
        assert await lock.acquire("k", "tok", 10.0) is LockAcquisition.ACQUIRED
    warning: Final = seen[0]
    assert warning.filename.endswith("/outbound_credentials/redis_distributed_lock.py")
