"""The service-logging tasks RedisCache fires must be strongly referenced.

`asyncio.create_task` / `loop.create_task` hand the task to the event loop,
which keeps only a *weak* reference to it. A task whose only referent was the
`create_task(...)` call can be garbage collected while it is suspended, and
the service-log event it was going to emit is then silently lost.

`RedisCache` fires these from every cache operation, on both the success and
the failure path. These tests drive each of those methods with the redis client
mocked out and assert the task lands in `_service_logging_tasks`, and that the
entry is removed once it completes so holding it cannot become a leak.
"""

import asyncio
import contextlib
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# litellm's package __getattr__ only resolves names it lists explicitly, so the
# submodule has to be imported before patch() can find it by path.
import litellm._redis  # noqa: F401  # imported for its side effect, see above
from litellm.caching.redis_cache import RedisCache


@pytest.fixture
def cache() -> RedisCache:
    """A RedisCache with every real redis connection stubbed out."""
    # RedisCache imports these from litellm._redis inside __init__, so they have
    # to be patched at the source module rather than on redis_cache.
    with (
        patch("litellm._redis.get_redis_client", return_value=MagicMock()),
        patch("litellm._redis.get_redis_connection_pool", return_value=MagicMock()),
    ):
        redis_cache = RedisCache(host="localhost", port=6379)

    # The hooks are what get scheduled; make them awaitable no-ops so the tasks
    # complete immediately instead of touching a real logging backend.
    redis_cache.service_logger_obj.async_service_success_hook = AsyncMock()
    redis_cache.service_logger_obj.async_service_failure_hook = AsyncMock()
    return redis_cache


async def drain(cache: RedisCache) -> None:
    """Let every scheduled task run and every done-callback be delivered."""
    # add_done_callback goes through call_soon, so the discard lands on the
    # tick after the task itself finishes.
    for _ in range(4):
        await asyncio.sleep(0)


def failing_client() -> MagicMock:
    """An async redis client whose every operation raises.

    The client itself has to construct successfully: several of these methods
    call `init_async_client()` *outside* their try block, so making that raise
    would propagate before any service-log task is ever scheduled. The failure
    has to come from the redis operation instead.
    """
    client = MagicMock()
    for op in ("get", "set", "mget", "sadd", "incrbyfloat", "expire", "rpush", "lpop", "scan_iter", "ping"):
        setattr(client, op, AsyncMock(side_effect=ConnectionError("redis is down")))
    return client


# Each entry drives one RedisCache method against a client that fails, which is
# the path to the service-failure hook these methods fire.
FAILING_CALLS = (
    ("async_set_cache", ("k", "v")),
    ("async_get_cache", ("k",)),
    ("async_batch_get_cache", (("k1", "k2"),)),
    ("async_increment", ("k", 1.0)),
    ("async_rpush", ("k", ("v",))),
    ("async_lpop", ("k",)),
    ("ping", ()),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name, args", FAILING_CALLS, ids=tuple(n for n, _ in FAILING_CALLS))
async def test_failure_path_holds_the_service_log_task(cache: RedisCache, method_name: str, args: tuple) -> None:
    method = getattr(cache, method_name)

    with patch.object(cache, "init_async_client", return_value=failing_client()):
        # Several of these re-raise after logging and several swallow; either
        # way the scheduled task is what this asserts on.
        with contextlib.suppress(ConnectionError):
            await method(*args)

        assert cache._service_logging_tasks, (
            f"{method_name} scheduled a service-log task without keeping a "
            "reference to it, so the loop's weak reference is the only one"
        )

    await drain(cache)
    assert not cache._service_logging_tasks, f"{method_name} left its finished task in the registry"


@pytest.mark.asyncio
async def test_success_path_holds_the_service_log_task(cache: RedisCache) -> None:
    """The success hook is fired from a task too, and needs the same reference."""
    client = MagicMock()
    client.set = AsyncMock(return_value=True)

    with patch.object(cache, "init_async_client", return_value=client):
        await cache.async_set_cache("k", "v")
        assert cache._service_logging_tasks

    await drain(cache)
    assert not cache._service_logging_tasks


@pytest.mark.asyncio
async def test_ping_error_handlers_hold_their_tasks(cache: RedisCache) -> None:
    """_handle_async_ping_error / _handle_sync_ping_error each fire one task."""
    error = ConnectionError("redis is down")

    cache._handle_async_ping_error(error)
    assert len(cache._service_logging_tasks) == 1

    cache._handle_sync_ping_error(error)
    assert len(cache._service_logging_tasks) == 2

    await drain(cache)
    assert not cache._service_logging_tasks


@pytest.mark.asyncio
async def test_registry_starts_empty(cache: RedisCache) -> None:
    """Nothing is scheduled just by constructing the cache."""
    assert len(cache._service_logging_tasks) == 0
