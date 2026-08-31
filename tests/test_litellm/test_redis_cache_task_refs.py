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
        patch("litellm._redis.get_redis_client", return_value=MagicMock()),  # test-quality-ok: connection factory, no HTTP boundary to fake
        patch("litellm._redis.get_redis_connection_pool", return_value=MagicMock()),  # test-quality-ok: connection factory, no HTTP boundary to fake
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
    for op in ("get", "set", "mget", "sadd", "incrbyfloat", "expire", "ttl", "rpush", "lpop", "ping"):
        setattr(client, op, AsyncMock(side_effect=ConnectionError("redis is down")))
    # scan_iter is not awaited, it is iterated, so it has to fail on the call
    # itself rather than on an await that never happens.
    client.scan_iter = MagicMock(side_effect=ConnectionError("redis is down"))
    client.pipeline = MagicMock(return_value=_pipeline_cm())
    return client


class _AsyncIter:
    """`scan_iter` is consumed with `async for`, which a plain AsyncMock is not."""

    def __init__(self, items):
        self._items = list(items)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self._items:
            raise StopAsyncIteration
        return self._items.pop(0)


def _pipeline_cm() -> MagicMock:
    """`pipeline(transaction=False)` is entered as an async context manager."""
    pipe = MagicMock()
    pipe.execute = AsyncMock(return_value=[])
    cm = MagicMock()
    cm.__aenter__ = AsyncMock(return_value=pipe)
    cm.__aexit__ = AsyncMock(return_value=False)
    return cm


def working_client() -> MagicMock:
    """An async redis client whose operations all succeed.

    Return values are the benign ones (a cache miss, a zero-length list): the
    success hook fires either way, and that hook is what these tests are about.
    """
    client = MagicMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock(return_value=True)
    client.mget = AsyncMock(return_value=[None, None])
    client.sadd = AsyncMock(return_value=1)
    client.incrbyfloat = AsyncMock(return_value=1.0)
    client.expire = AsyncMock(return_value=True)
    client.ttl = AsyncMock(return_value=60)
    client.rpush = AsyncMock(return_value=1)
    client.lpop = AsyncMock(return_value=None)
    client.ping = AsyncMock(return_value=True)
    client.scan_iter = MagicMock(return_value=_AsyncIter([]))
    client.pipeline = MagicMock(return_value=_pipeline_cm())
    return client


# The pipeline methods delegate the actual redis work to a helper, so that is
# the seam to drive them from rather than the individual redis commands.
PIPELINE_HELPERS = {
    "async_set_cache_pipeline": "_pipeline_helper",
    "async_increment_pipeline": "_pipeline_increment_helper",
    "async_rpush_pipeline": "_pipeline_rpush_helper",
    "async_lpop_pipeline": "_pipeline_lpop_helper",
}


# Every RedisCache method that fires a service-log task, with arguments that
# get it past its own early-return guards. Both hooks are exercised per method.
CALLS = (
    ("async_set_cache", ("k", "v")),
    ("async_get_cache", ("k",)),
    ("async_batch_get_cache", (("k1", "k2"),)),
    ("async_increment", ("k", 1.0)),
    ("async_rpush", ("k", ("v",))),
    ("async_lpop", ("k",)),
    ("ping", ()),
    ("async_scan_iter", ("pattern",)),
    ("async_set_cache_sadd", ("k", ["v"], None)),
    ("async_set_cache_pipeline", ([("k", "v")],)),
    ("async_increment_pipeline", ([{"key": "k", "increment_value": 1.0, "ttl": 60}],)),
    ("async_rpush_pipeline", ([{"key": "k", "values": ["v"]}],)),
    ("async_lpop_pipeline", ([{"key": "k", "count": 1}],)),
)
CALL_IDS = tuple(n for n, _ in CALLS)


@contextlib.contextmanager
def driven(cache: RedisCache, method_name: str, *, failing: bool):
    """Point one RedisCache method at a client that either works or breaks.

    The pipeline methods reach redis through a helper rather than through the
    client's own commands, so for those the helper is the seam.
    """
    client = failing_client() if failing else working_client()
    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(cache, "init_async_client", return_value=client))

    helper = PIPELINE_HELPERS.get(method_name)
    if helper is not None:
        stack.enter_context(
            patch.object(
                cache,
                helper,
                AsyncMock(side_effect=ConnectionError("redis is down") if failing else None, return_value=[]),
            )
        )
    with stack:
        yield


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name, args", CALLS, ids=CALL_IDS)
async def test_failure_path_holds_the_service_log_task(cache: RedisCache, method_name: str, args: tuple) -> None:
    """The service-failure hook is fired from a task that nothing else holds."""
    with driven(cache, method_name, failing=True):
        # Several of these re-raise after logging and several swallow; either
        # way the scheduled task is what this asserts on.
        with contextlib.suppress(ConnectionError):
            await getattr(cache, method_name)(*args)

        assert cache._service_logging_tasks, (
            f"{method_name} scheduled a service-log task without keeping a "
            "reference to it, so the loop's weak reference is the only one"
        )

    await drain(cache)
    assert not cache._service_logging_tasks, f"{method_name} left its finished task in the registry"


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name, args", CALLS, ids=CALL_IDS)
async def test_success_path_holds_the_service_log_task(cache: RedisCache, method_name: str, args: tuple) -> None:
    """The success hook is fired from a task too, and needs the same reference."""
    with driven(cache, method_name, failing=False):
        await getattr(cache, method_name)(*args)

        assert cache._service_logging_tasks, (
            f"{method_name} scheduled its success-log task without keeping a reference to it"
        )

    await drain(cache)
    assert not cache._service_logging_tasks, f"{method_name} left its finished task in the registry"


# These wrap init_async_client() in their own try, so a client that cannot even
# be built is a separate logged path from a redis command that fails.
CLIENT_INIT_CALLS = (
    ("async_set_cache", ("k", "v")),
    ("async_set_cache_sadd", ("k", ["v"], None)),
)


@pytest.mark.asyncio
@pytest.mark.parametrize("method_name, args", CLIENT_INIT_CALLS, ids=tuple(n for n, _ in CLIENT_INIT_CALLS))
async def test_client_init_failure_holds_the_service_log_task(cache: RedisCache, method_name: str, args: tuple) -> None:
    with patch.object(cache, "init_async_client", side_effect=ConnectionError("no client")):
        with contextlib.suppress(ConnectionError):
            await getattr(cache, method_name)(*args)

        assert cache._service_logging_tasks, f"{method_name} logged the client-init failure from a task it did not keep"

    await drain(cache)
    assert not cache._service_logging_tasks


@pytest.mark.asyncio
async def test_health_ping_setup_holds_its_task(cache: RedisCache) -> None:
    """_setup_health_pings fires the async ping from a task of its own."""
    with patch.object(cache, "ping", AsyncMock(return_value=True)):
        cache._setup_health_pings()
        assert cache._service_logging_tasks, "the async health ping task is not kept anywhere"

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
