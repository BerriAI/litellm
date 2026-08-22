import asyncio
from types import SimpleNamespace

import pytest

from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.utils import BudgetConfig

_SPEND_KEY = "provider_spend:openai:1d"


def _increment(increment_value: float) -> RedisPipelineIncrementOperation:
    return RedisPipelineIncrementOperation(key=_SPEND_KEY, increment_value=increment_value, ttl=86400)


class _MockRedisCache:
    def __init__(
        self,
        initial_values: dict[str, float],
        pipeline_started: asyncio.Event | None = None,
        allow_pipeline_to_complete: asyncio.Event | None = None,
        should_fail_pipeline: bool = False,
    ) -> None:
        self.values = initial_values
        self.events: list[str] = []
        self.pipeline_started = pipeline_started
        self.allow_pipeline_to_complete = allow_pipeline_to_complete
        self.should_fail_pipeline = should_fail_pipeline

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

    async def async_batch_get_cache(self, key_list: list[str], **kwargs: object) -> dict[str, float | None]:
        self.events.append("batch_get")
        return {key: self.values.get(key) for key in key_list}


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
    budget_limiter._redis_increment_queue_lock = asyncio.Lock()
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
