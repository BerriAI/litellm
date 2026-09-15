import asyncio
from types import SimpleNamespace

import pytest

from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.utils import BudgetConfig

_SPEND_KEY = "provider_spend:openai:1d"


def _increment(increment_value: float) -> RedisPipelineIncrementOperation:
    return RedisPipelineIncrementOperation(key=_SPEND_KEY, increment_value=increment_value, ttl=86400)


class _ObservedLock(asyncio.Lock):
    def __init__(self) -> None:
        super().__init__()
        self.waiter_started = asyncio.Event()

    async def acquire(self) -> bool:
        if self.locked():
            self.waiter_started.set()
        return await super().acquire()


class _MockRedisCache:
    def __init__(
        self,
        initial_values: dict[str, float],
        pipeline_started: asyncio.Event | None = None,
        allow_pipeline_to_complete: asyncio.Event | None = None,
        should_fail_pipeline: bool = False,
        pipeline_completed: asyncio.Event | None = None,
        read_started: asyncio.Event | None = None,
        allow_read_to_complete: asyncio.Event | None = None,
    ) -> None:
        self.values = initial_values
        self.events: list[str] = []
        self.pipeline_started = pipeline_started
        self.allow_pipeline_to_complete = allow_pipeline_to_complete
        self.should_fail_pipeline = should_fail_pipeline
        self.pipeline_completed = pipeline_completed
        self.read_started = read_started
        self.allow_read_to_complete = allow_read_to_complete

    async def async_increment_pipeline(
        self, increment_list: list[RedisPipelineIncrementOperation], **kwargs: object
    ) -> None:
        self.events.append("increment_pipeline:start")
        if self.pipeline_started is not None:
            self.pipeline_started.set()
        if self.allow_pipeline_to_complete is not None:
            await self.allow_pipeline_to_complete.wait()
        if self.should_fail_pipeline:
            raise RuntimeError("redis down")
        for op in increment_list:
            key = op["key"]
            current = float(self.values.get(key, 0.0) or 0.0)
            self.values[key] = current + float(op["increment_value"])
        self.events.append("increment_pipeline:done")
        if self.pipeline_completed is not None:
            self.pipeline_completed.set()

    async def async_batch_get_cache(self, key_list: list[str], **kwargs: object) -> dict[str, float | None]:
        self.events.append("batch_get")
        snapshot = {key: self.values.get(key) for key in key_list}
        if self.read_started is not None:
            self.read_started.set()
        if self.allow_read_to_complete is not None:
            await self.allow_read_to_complete.wait()
        return snapshot


class _MockInMemoryCache:
    def __init__(self, initial_values: dict[str, float]) -> None:
        self.values = initial_values

    async def async_increment(self, key: str, value: float, ttl: int, **kwargs: object) -> float:
        current = float(self.values.get(key, 0.0) or 0.0)
        self.values[key] = current + float(value)
        return self.values[key]

    async def async_set_cache(self, key: str, value: float, **kwargs: object) -> None:
        self.values[key] = float(value)


def _new_router_budget_limiter(
    *,
    redis_cache: object,
    queue_lock: asyncio.Lock | None = None,
    in_memory_cache: object | None = None,
    redis_increment_operation_queue: list[RedisPipelineIncrementOperation] | None = None,
    provider_budget_config: dict[str, BudgetConfig] | None = None,
) -> RouterBudgetLimiting:
    budget_limiter = RouterBudgetLimiting.__new__(RouterBudgetLimiting)
    budget_limiter.dual_cache = SimpleNamespace(
        redis_cache=redis_cache,
        in_memory_cache=in_memory_cache if in_memory_cache is not None else SimpleNamespace(),
    )
    budget_limiter.provider_budget_config = provider_budget_config
    budget_limiter.deployment_budget_config = None
    budget_limiter.tag_budget_config = None
    budget_limiter.redis_increment_operation_queue = (
        list(redis_increment_operation_queue) if redis_increment_operation_queue is not None else []
    )
    budget_limiter._redis_increment_queue_lock = queue_lock if queue_lock is not None else asyncio.Lock()
    budget_limiter._redis_increment_flush_lock = asyncio.Lock()
    budget_limiter._detached_increment_operations = None
    return budget_limiter


@pytest.mark.asyncio
async def test_should_await_redis_pipeline_before_sync_reads() -> None:
    pipeline_started = asyncio.Event()
    allow_pipeline_to_complete = asyncio.Event()
    redis_cache = _MockRedisCache(
        initial_values={_SPEND_KEY: 100.0},
        pipeline_started=pipeline_started,
        allow_pipeline_to_complete=allow_pipeline_to_complete,
    )
    in_memory_cache = _MockInMemoryCache(initial_values={_SPEND_KEY: 160.0})
    budget_limiter = _new_router_budget_limiter(
        redis_cache=redis_cache,
        in_memory_cache=in_memory_cache,
        redis_increment_operation_queue=[_increment(60.0)],
        provider_budget_config={"openai": BudgetConfig(time_period="1d", budget_limit=500.0)},
    )

    sync_task = asyncio.create_task(budget_limiter._sync_in_memory_spend_with_redis())
    await asyncio.wait_for(pipeline_started.wait(), timeout=1)
    assert "batch_get" not in redis_cache.events
    allow_pipeline_to_complete.set()
    await sync_task

    assert redis_cache.values[_SPEND_KEY] == 160.0
    assert in_memory_cache.values[_SPEND_KEY] == 160.0
    assert budget_limiter.redis_increment_operation_queue == []
    assert redis_cache.events == [
        "increment_pipeline:start",
        "increment_pipeline:done",
        "batch_get",
    ]


@pytest.mark.asyncio
async def test_should_requeue_increments_when_redis_pipeline_fails() -> None:
    redis_cache = _MockRedisCache(initial_values={}, should_fail_pipeline=True)
    budget_limiter = _new_router_budget_limiter(
        redis_cache=redis_cache,
        redis_increment_operation_queue=[_increment(10.0)],
    )

    flush_succeeded = await budget_limiter._push_in_memory_increments_to_redis()

    assert flush_succeeded is False
    assert budget_limiter.redis_increment_operation_queue == [_increment(10.0)]
    assert budget_limiter._detached_increment_operations is None


@pytest.mark.asyncio
async def test_should_keep_new_increments_when_pipeline_flush_fails() -> None:
    pipeline_started = asyncio.Event()
    allow_pipeline_to_complete = asyncio.Event()
    redis_cache = _MockRedisCache(
        initial_values={},
        pipeline_started=pipeline_started,
        allow_pipeline_to_complete=allow_pipeline_to_complete,
        should_fail_pipeline=True,
    )
    in_memory_cache = _MockInMemoryCache(initial_values={_SPEND_KEY: 0.0})
    budget_limiter = _new_router_budget_limiter(
        redis_cache=redis_cache,
        in_memory_cache=in_memory_cache,
        redis_increment_operation_queue=[_increment(10.0)],
    )

    push_task = asyncio.create_task(budget_limiter._push_in_memory_increments_to_redis())
    await asyncio.wait_for(pipeline_started.wait(), timeout=1)
    await budget_limiter._increment_spend_in_current_window(spend_key=_SPEND_KEY, response_cost=20.0, ttl=86400)
    allow_pipeline_to_complete.set()
    await push_task

    assert budget_limiter.redis_increment_operation_queue == [_increment(10.0), _increment(20.0)]


@pytest.mark.asyncio
async def test_should_keep_in_memory_spend_when_redis_pipeline_fails() -> None:
    redis_cache = _MockRedisCache(initial_values={_SPEND_KEY: 100.0}, should_fail_pipeline=True)
    in_memory_cache = _MockInMemoryCache(initial_values={_SPEND_KEY: 160.0})
    budget_limiter = _new_router_budget_limiter(
        redis_cache=redis_cache,
        in_memory_cache=in_memory_cache,
        redis_increment_operation_queue=[_increment(60.0)],
        provider_budget_config={"openai": BudgetConfig(time_period="1d", budget_limit=500.0)},
    )

    await budget_limiter._sync_in_memory_spend_with_redis()

    assert in_memory_cache.values[_SPEND_KEY] == 160.0
    assert redis_cache.values[_SPEND_KEY] == 100.0
    assert budget_limiter.redis_increment_operation_queue == [_increment(60.0)]
    assert "batch_get" not in redis_cache.events


@pytest.mark.asyncio
async def test_should_keep_increments_when_flush_is_cancelled_after_success() -> None:
    pipeline_started = asyncio.Event()
    allow_pipeline_to_complete = asyncio.Event()
    redis_cache = _MockRedisCache(
        initial_values={_SPEND_KEY: 0.0},
        pipeline_started=pipeline_started,
        allow_pipeline_to_complete=allow_pipeline_to_complete,
    )
    budget_limiter = _new_router_budget_limiter(
        redis_cache=redis_cache,
        redis_increment_operation_queue=[_increment(10.0)],
    )

    push_task = asyncio.create_task(budget_limiter._push_in_memory_increments_to_redis())
    await asyncio.wait_for(pipeline_started.wait(), timeout=1)
    push_task.cancel()
    allow_pipeline_to_complete.set()
    with pytest.raises(asyncio.CancelledError):
        await push_task

    assert redis_cache.values[_SPEND_KEY] == 10.0
    assert budget_limiter.redis_increment_operation_queue == []
    assert budget_limiter._detached_increment_operations is None


@pytest.mark.asyncio
async def test_empty_flush_does_not_block_later_increment_sync() -> None:
    redis_cache = _MockRedisCache(initial_values={_SPEND_KEY: 100.0})
    in_memory_cache = _MockInMemoryCache(initial_values={_SPEND_KEY: 100.0})
    budget_limiter = _new_router_budget_limiter(
        redis_cache=redis_cache,
        in_memory_cache=in_memory_cache,
        provider_budget_config={"openai": BudgetConfig(time_period="1d", budget_limit=500.0)},
    )

    empty_flush_succeeded = await budget_limiter._push_in_memory_increments_to_redis()
    await budget_limiter._increment_spend_in_current_window(spend_key=_SPEND_KEY, response_cost=20.0, ttl=86400)
    await budget_limiter._sync_in_memory_spend_with_redis()

    assert empty_flush_succeeded is True
    assert budget_limiter._detached_increment_operations is None
    assert budget_limiter.redis_increment_operation_queue == []
    assert redis_cache.values[_SPEND_KEY] == 120.0
    assert in_memory_cache.values[_SPEND_KEY] == 120.0
    assert redis_cache.events == [
        "increment_pipeline:start",
        "increment_pipeline:done",
        "batch_get",
    ]


@pytest.mark.asyncio
async def test_should_requeue_increments_when_flush_is_cancelled_and_redis_fails() -> None:
    pipeline_started = asyncio.Event()
    allow_pipeline_to_complete = asyncio.Event()
    redis_cache = _MockRedisCache(
        initial_values={_SPEND_KEY: 0.0},
        pipeline_started=pipeline_started,
        allow_pipeline_to_complete=allow_pipeline_to_complete,
        should_fail_pipeline=True,
    )
    budget_limiter = _new_router_budget_limiter(
        redis_cache=redis_cache,
        redis_increment_operation_queue=[_increment(10.0)],
    )

    push_task = asyncio.create_task(budget_limiter._push_in_memory_increments_to_redis())
    await asyncio.wait_for(pipeline_started.wait(), timeout=1)
    push_task.cancel()
    allow_pipeline_to_complete.set()
    with pytest.raises(asyncio.CancelledError):
        await push_task

    assert redis_cache.values[_SPEND_KEY] == 0.0
    assert budget_limiter.redis_increment_operation_queue == [_increment(10.0)]
    assert budget_limiter._detached_increment_operations is None


@pytest.mark.asyncio
@pytest.mark.parametrize("pause_during", ["write", "read"])
async def test_sync_preserves_spend_recorded_during_redis_io(pause_during: str) -> None:
    io_started = asyncio.Event()
    allow_io_to_complete = asyncio.Event()
    redis_cache = _MockRedisCache(
        initial_values={_SPEND_KEY: 100.0},
        pipeline_started=io_started if pause_during == "write" else None,
        allow_pipeline_to_complete=allow_io_to_complete if pause_during == "write" else None,
        read_started=io_started if pause_during == "read" else None,
        allow_read_to_complete=allow_io_to_complete if pause_during == "read" else None,
    )
    in_memory_cache = _MockInMemoryCache(initial_values={_SPEND_KEY: 160.0})
    budget_limiter = _new_router_budget_limiter(
        redis_cache=redis_cache,
        in_memory_cache=in_memory_cache,
        redis_increment_operation_queue=[_increment(60.0)],
        provider_budget_config={"openai": BudgetConfig(time_period="1d", budget_limit=175.0)},
    )

    sync_task = asyncio.create_task(budget_limiter._sync_in_memory_spend_with_redis())
    await asyncio.wait_for(io_started.wait(), timeout=1)
    await budget_limiter._increment_spend_in_current_window(_SPEND_KEY, 20.0, 86400)
    allow_io_to_complete.set()
    await sync_task

    assert in_memory_cache.values[_SPEND_KEY] == 180.0
    assert redis_cache.values[_SPEND_KEY] == 160.0
    assert budget_limiter.redis_increment_operation_queue == [_increment(20.0)]

    await budget_limiter._sync_in_memory_spend_with_redis()

    assert in_memory_cache.values[_SPEND_KEY] == 180.0
    assert redis_cache.values[_SPEND_KEY] == 180.0
    assert budget_limiter.redis_increment_operation_queue == []


@pytest.mark.asyncio
@pytest.mark.parametrize("cancellations", [1, 2])
async def test_cancelled_flush_does_not_requeue_an_applied_batch(cancellations: int) -> None:
    pipeline_started = asyncio.Event()
    pipeline_completed = asyncio.Event()
    allow_pipeline = asyncio.Event()
    redis_cache = _MockRedisCache(
        initial_values={_SPEND_KEY: 0.0},
        pipeline_started=pipeline_started,
        pipeline_completed=pipeline_completed,
        allow_pipeline_to_complete=allow_pipeline,
    )
    queue_lock = _ObservedLock()
    limiter = _new_router_budget_limiter(
        redis_cache=redis_cache, queue_lock=queue_lock, redis_increment_operation_queue=[_increment(10.0)]
    )
    push_task = asyncio.create_task(limiter._push_in_memory_increments_to_redis())
    await asyncio.wait_for(pipeline_started.wait(), timeout=1)
    async with limiter._redis_increment_queue_lock:
        allow_pipeline.set()
        await asyncio.wait_for(pipeline_completed.wait(), timeout=1)
        await asyncio.wait_for(queue_lock.waiter_started.wait(), timeout=1)
        for _ in range(cancellations):
            push_task.cancel()
            await asyncio.sleep(0)
        assert not push_task.done()
    with pytest.raises(asyncio.CancelledError):
        await push_task
    await limiter._push_in_memory_increments_to_redis()
    assert redis_cache.values[_SPEND_KEY] == 10.0
    assert limiter.redis_increment_operation_queue == []
    assert limiter._detached_increment_operations is None
