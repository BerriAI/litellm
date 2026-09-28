import asyncio
import gc
import logging
from types import SimpleNamespace
from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

import litellm
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache, RedisCircuitBreakerOpenError
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.router import LiteLLM_Params
from litellm.types.utils import BudgetConfig


@pytest.fixture
def disable_budget_sync(monkeypatch):
    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_dict_does_not_require_litellm_params_instantiation(
    disable_budget_sync, monkeypatch
):
    class RaiseOnInit:
        def __init__(self, *args, **kwargs):
            raise AssertionError("LiteLLM_Params should not be instantiated in hot path")

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.LiteLLM_Params",
        RaiseOnInit,
    )

    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    deployment = {"litellm_params": {"model": "openai/gpt-4o-mini"}}
    provider = provider_budget._get_llm_provider_for_deployment(deployment)

    assert provider == "openai"


@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_dict_view_supports_mapping_and_attr_access(
    disable_budget_sync, monkeypatch
):
    observed = {}

    def _future_style_get_llm_provider(
        model,
        custom_llm_provider=None,
        api_base=None,
        api_key=None,
        litellm_params=None,
    ):
        assert litellm_params is not None
        observed["model_attr"] = litellm_params.model
        observed["provider_get"] = litellm_params.get("custom_llm_provider")
        observed["api_base_item"] = litellm_params["api_base"]
        observed["has_api_key"] = "api_key" in litellm_params
        observed["model_dump"] = litellm_params.model_dump()
        return model, "openai", None, None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.litellm.get_llm_provider",
        _future_style_get_llm_provider,
    )

    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    deployment = {
        "litellm_params": {
            "model": "openai/gpt-4o-mini",
            "custom_llm_provider": "openai",
            "api_base": "https://api.openai.com/v1",
        }
    }
    provider = provider_budget._get_llm_provider_for_deployment(deployment)

    assert provider == "openai"
    assert observed["model_attr"] == "openai/gpt-4o-mini"
    assert observed["provider_get"] == "openai"
    assert observed["api_base_item"] == "https://api.openai.com/v1"
    assert observed["has_api_key"] is False
    assert observed["model_dump"]["model"] == "openai/gpt-4o-mini"


@pytest.mark.asyncio
async def test_async_filter_deployments_resolves_provider_once_per_deployment(disable_budget_sync, monkeypatch):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(budget_duration="1d", max_budget=100.0),
        },
    )

    healthy_deployments = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "deployment-1"},
        },
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "openai/gpt-4o-mini"},
            "model_info": {"id": "deployment-2"},
        },
    ]

    provider_resolution_calls = 0

    def _count_provider_calls(deployment):
        nonlocal provider_resolution_calls
        provider_resolution_calls += 1
        return "openai"

    monkeypatch.setattr(
        provider_budget,
        "_get_llm_provider_for_deployment",
        _count_provider_calls,
    )

    filtered_deployments = await provider_budget.async_filter_deployments(
        model="gpt-4o-mini",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={},
        parent_otel_span=None,
    )

    assert len(filtered_deployments) == len(healthy_deployments)
    assert provider_resolution_calls == len(healthy_deployments)


@pytest.mark.asyncio
async def test_async_filter_deployments_does_not_recompute_provider_when_resolved_none(
    disable_budget_sync, monkeypatch
):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(budget_duration="1d", max_budget=100.0),
        },
        model_list=[
            {
                "model_name": "gpt-4o-mini",
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "max_budget": 100.0,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "deployment-1"},
            }
        ],
    )

    healthy_deployments = [
        {
            "model_name": "gpt-4o-mini",
            "litellm_params": {"model": "unknown-provider/model"},
            "model_info": {"id": "deployment-1"},
        }
    ]

    provider_resolution_calls = 0

    def _provider_returns_none(deployment):
        nonlocal provider_resolution_calls
        provider_resolution_calls += 1
        return None

    monkeypatch.setattr(
        provider_budget,
        "_get_llm_provider_for_deployment",
        _provider_returns_none,
    )

    filtered_deployments = await provider_budget.async_filter_deployments(
        model="gpt-4o-mini",
        healthy_deployments=healthy_deployments,
        messages=[],
        request_kwargs={},
        parent_otel_span=None,
    )

    assert len(filtered_deployments) == len(healthy_deployments)
    assert provider_resolution_calls == len(healthy_deployments)


def _legacy_provider_resolution(deployment):
    """
    Reference implementation used before hot-path optimization.
    """
    try:
        _litellm_params = LiteLLM_Params(**deployment.get("litellm_params", {"model": ""}))
        _, custom_llm_provider, _, _ = litellm.get_llm_provider(
            model=_litellm_params.model,
            litellm_params=_litellm_params,
        )
    except Exception:
        return None
    return custom_llm_provider


@pytest.mark.parametrize(
    "deployment",
    [
        {"litellm_params": {"model": "openai/gpt-4o-mini"}},
        {"litellm_params": {"model": "gpt-4o-mini", "custom_llm_provider": "openai"}},
        {"litellm_params": {"model": "unknown-provider/model"}},
    ],
)
@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_matches_legacy_behavior(disable_budget_sync, deployment):
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )

    current_provider = provider_budget._get_llm_provider_for_deployment(deployment)
    legacy_provider = _legacy_provider_resolution(deployment)

    assert current_provider == legacy_provider


def test_register_deployment_budget_for_runtime_added_deployment(disable_budget_sync, monkeypatch):
    import asyncio

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)
    budget_limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={},
    )
    model_id = "dynamic-deployment-id"
    budget_limiter.register_deployment_budget(
        deployment={
            "model_name": "dynamic-budget-model",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "max_budget": 0.000000000001,
                "budget_duration": "1d",
            },
            "model_info": {"id": model_id},
        }
    )

    config = budget_limiter._get_budget_config_for_deployment(model_id)
    assert config is not None
    assert config.max_budget == 0.000000000001
    assert config.budget_duration == "1d"

    budget_limiter.unregister_deployment_budget(model_id=model_id)
    assert budget_limiter._get_budget_config_for_deployment(model_id) is None


def test_router_add_deployment_registers_deployment_budget(disable_budget_sync, monkeypatch):
    import asyncio

    from litellm import Router
    from litellm.types.router import Deployment, LiteLLM_Params, ModelInfo

    monkeypatch.setattr(asyncio, "create_task", lambda coro: None)

    router = Router(
        model_list=[],
        optional_pre_call_checks=[],
    )

    router.add_deployment(
        deployment=Deployment(
            model_name="dynamic-budget-model",
            litellm_params=LiteLLM_Params(
                model="openai/gpt-4o-mini",
                api_key="fake-key",
                max_budget=0.000000000001,
                budget_duration="1d",
            ),
            model_info=ModelInfo(id="runtime-budget-deployment"),
        )
    )

    budget_limiter = router._get_router_deployment_budget_limiter()
    assert budget_limiter is not None
    config = budget_limiter._get_budget_config_for_deployment("runtime-budget-deployment")
    assert config is not None
    assert config.max_budget == 0.000000000001


@pytest.mark.asyncio
async def test_sync_refused_by_the_open_circuit_breaker_is_quiet_and_leaks_no_task(disable_budget_sync, caplog):
    """The budget sync runs every second, so an open breaker must not add an error line or an unretrieved task exception per cycle."""
    refused = RedisCircuitBreakerOpenError("Redis circuit breaker is open - skipping async_increment_pipeline")
    redis_cache = MagicMock(spec=RedisCache)
    redis_cache.async_increment_pipeline = AsyncMock(side_effect=refused)
    redis_cache.async_batch_get_cache = AsyncMock(side_effect=refused)
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(redis_cache=redis_cache),
        provider_budget_config={"openai": BudgetConfig(max_budget=1.0, budget_duration="1d")},
    )
    await asyncio.gather(*(task for task in asyncio.all_tasks() if task is not asyncio.current_task()))
    limiter.redis_increment_operation_queue = [{"key": "provider_spend:openai:1d", "increment_value": 0.5, "ttl": 60}]
    loop = asyncio.get_running_loop()
    unretrieved = MagicMock()
    loop.set_exception_handler(unretrieved)

    try:
        with caplog.at_level(logging.ERROR):
            await limiter._sync_in_memory_spend_with_redis()
            await asyncio.sleep(0)
            gc.collect()
    finally:
        loop.set_exception_handler(None)

    assert caplog.records == []
    unretrieved.assert_not_called()
    assert limiter.redis_increment_operation_queue == [
        {"key": "provider_spend:openai:1d", "increment_value": 0.5, "ttl": 60}
    ]
    assert redis_cache.async_increment_pipeline.await_count == 1


async def _limiter_with_redis(redis_cache: MagicMock) -> RouterBudgetLimiting:
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(redis_cache=redis_cache),
        provider_budget_config={"openai": BudgetConfig(max_budget=1.0, budget_duration="1d")},
    )
    await asyncio.gather(*(task for task in asyncio.all_tasks() if task is not asyncio.current_task()))
    limiter.redis_increment_operation_queue = [{"key": "provider_spend:openai:1d", "increment_value": 0.5, "ttl": 60}]
    return limiter


@pytest.mark.asyncio
async def test_push_waits_for_redis_before_completing(disable_budget_sync):
    redis_started = asyncio.Event()
    redis_answered = asyncio.Event()

    async def wait_for_redis(**_: object) -> None:
        redis_started.set()
        await redis_answered.wait()

    redis_cache = MagicMock(spec=RedisCache)
    redis_cache.async_increment_pipeline = AsyncMock(side_effect=wait_for_redis)
    limiter = await _limiter_with_redis(redis_cache)

    push_task = asyncio.create_task(limiter._push_in_memory_increments_to_redis())
    await asyncio.wait_for(redis_started.wait(), timeout=1)
    assert not push_task.done()
    redis_answered.set()
    assert await asyncio.wait_for(push_task, timeout=1) is True
    assert redis_cache.async_increment_pipeline.await_count == 1
    assert limiter.redis_increment_operation_queue == []


@pytest.mark.asyncio
async def test_push_task_failure_is_logged_once_and_not_leaked(disable_budget_sync, caplog):
    """A real Redis failure on the background push must surface as one error line, never as an unretrieved task exception."""
    redis_cache = MagicMock(spec=RedisCache)
    redis_cache.async_increment_pipeline = AsyncMock(
        side_effect=ConnectionError("Error 61 connecting to 127.0.0.1:6379")
    )
    limiter = await _limiter_with_redis(redis_cache)
    loop = asyncio.get_running_loop()
    unretrieved = MagicMock()
    loop.set_exception_handler(unretrieved)

    try:
        with caplog.at_level(logging.ERROR):
            await limiter._push_in_memory_increments_to_redis()
            await asyncio.sleep(0)
            gc.collect()
    finally:
        loop.set_exception_handler(None)

    assert [record.getMessage() for record in caplog.records] == [
        "Error syncing in-memory cache with Redis: Error 61 connecting to 127.0.0.1:6379"
    ]
    unretrieved.assert_not_called()


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

    assert budget_limiter.redis_increment_operation_queue == [_increment(30.0)]


@pytest.mark.asyncio
async def test_failed_redis_flushes_coalesce_spend_by_key() -> None:
    other_spend_key: Final = "provider_spend:other:1d"
    redis_cache: Final = _MockRedisCache(
        initial_values={_SPEND_KEY: 0.0, other_spend_key: 0.0}, should_fail_pipeline=True
    )
    in_memory_cache: Final = _MockInMemoryCache(initial_values={_SPEND_KEY: 0.0, other_spend_key: 0.0})
    budget_limiter: Final = _new_router_budget_limiter(redis_cache=redis_cache, in_memory_cache=in_memory_cache)

    for spend_key, response_cost, ttl in (
        (_SPEND_KEY, 10.0, 90),
        (other_spend_key, 4.0, 50),
        (_SPEND_KEY, 20.0, 80),
        (_SPEND_KEY, 30.0, 70),
    ):
        await budget_limiter._increment_spend_in_current_window(spend_key, response_cost, ttl)
        assert await budget_limiter._push_in_memory_increments_to_redis() is False

    queued: Final = {operation["key"]: operation for operation in budget_limiter.redis_increment_operation_queue}
    assert len(budget_limiter.redis_increment_operation_queue) == 2
    assert queued[_SPEND_KEY] == RedisPipelineIncrementOperation(key=_SPEND_KEY, increment_value=60.0, ttl=70)
    assert queued[other_spend_key] == RedisPipelineIncrementOperation(key=other_spend_key, increment_value=4.0, ttl=50)

    redis_cache.should_fail_pipeline = False
    assert await budget_limiter._push_in_memory_increments_to_redis() is True
    assert redis_cache.values == {_SPEND_KEY: 60.0, other_spend_key: 4.0}
    assert budget_limiter.redis_increment_operation_queue == []


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


@pytest.mark.asyncio
async def test_cancelled_push_waiting_for_flush_lock_still_writes_spend() -> None:
    redis_cache = _MockRedisCache(initial_values={_SPEND_KEY: 0.0})
    budget_limiter = _new_router_budget_limiter(
        redis_cache=redis_cache,
        redis_increment_operation_queue=[_increment(10.0)],
    )
    flush_lock = _ObservedLock()
    budget_limiter._redis_increment_flush_lock = flush_lock

    async with flush_lock:
        push_task = asyncio.create_task(budget_limiter._push_in_memory_increments_to_redis())
        await asyncio.wait_for(flush_lock.waiter_started.wait(), timeout=1)
        push_task.cancel()
        await asyncio.sleep(0)
        assert not push_task.done()

    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(push_task, timeout=1)

    assert redis_cache.values[_SPEND_KEY] == 10.0
    assert budget_limiter.redis_increment_operation_queue == []


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
