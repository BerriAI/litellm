from typing import Final
from unittest.mock import AsyncMock, MagicMock

import pytest

from litellm import Router
from litellm._internal_context import current_service_target
from litellm.caching.redis_cache import RedisCache
from litellm.redis.stores.router_usage import RouterUsageStore, UsageCounter


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ("flush_cache", "reset"))
async def test_router_reset_clears_usage_and_failure_counters(operation: str) -> None:
    router: Final = Router(model_list=[])
    usage: Final = router._core_usage_store
    cooldowns: Final = router._core_cooldown_cache.cooldown_store
    counters: Final = (UsageCounter("dep", "model", "12-34", "tpm"), UsageCounter("dep", "model", "12-34", "rpm"))
    await usage.increment_deployment("dep", "model", "12-34", 7, 1, None)
    assert await usage.async_batch_get(counters) == [7, 1]
    assert cooldowns.increment_allowed_fails("dep", None, 30.0) == 1
    getattr(router, operation)()
    assert await usage.async_batch_get(counters) == [None, None]
    assert cooldowns.increment_allowed_fails("dep", None, 30.0) == 1


@pytest.mark.asyncio
async def test_deployment_increment_preserves_pipeline_ttl_and_integer_mirror() -> None:
    redis: Final = MagicMock(spec=RedisCache)
    redis.async_increment_pipeline = AsyncMock(return_value=[7.75, 2.0])
    store: Final = RouterUsageStore(redis)
    await store.increment_deployment("dep", "anthropic/model", "12-34", 7.75, 2, None)
    redis.async_increment_pipeline.assert_awaited_once_with(
        increment_list=[
            {"key": "global_router:dep:anthropic/model:tpm:12-34", "increment_value": 7.75, "ttl": 60},
            {"key": "global_router:dep:anthropic/model:rpm:12-34", "increment_value": 2, "ttl": 60},
        ],
        parent_otel_span=None,
    )
    values: Final = await store.async_batch_get(
        (UsageCounter("dep", "anthropic/model", "12-34", "tpm"), UsageCounter("dep", "anthropic/model", "12-34", "rpm"))
    )
    assert values == [7, 2]
    assert all(type(value) is int for value in values)
    redis.async_batch_get_cache.assert_not_called()
    assert current_service_target() is None


@pytest.mark.asyncio
async def test_memory_only_usage_increments_accumulate_without_redis() -> None:
    store: Final = RouterUsageStore(None)
    await store.increment_deployment("dep", "model", "12-34", 2, 1, None)
    await store.increment_deployment("dep", "model", "12-34", 5, 1, None)
    assert await store.async_batch_get(
        (UsageCounter("dep", "model", "12-34", "tpm"), UsageCounter("dep", "model", "12-34", "rpm"))
    ) == [7, 2]


@pytest.mark.asyncio
async def test_pre_call_rpm_increment_uses_same_namespace_and_service_target() -> None:
    redis: Final = MagicMock(spec=RedisCache)
    seen: Final[list[str | None]] = []

    async def increment(*args: object, **kwargs: object) -> float:
        seen.append(current_service_target())
        return 4.0

    redis.async_increment = AsyncMock(side_effect=increment)
    store: Final = RouterUsageStore(None)
    store.attach_redis(redis)
    counter: Final = UsageCounter("dep", "model", "12-34", "rpm")
    assert await store.async_increment(counter, value=1, ttl=60) == 4.0
    assert seen == ["router_usage"]
    redis.async_increment.assert_awaited_once_with(
        "global_router:dep:model:rpm:12-34", 1, parent_otel_span=None, ttl=60, refresh_ttl=False
    )


@pytest.mark.asyncio
async def test_failed_usage_pipeline_keeps_the_memory_increment_and_is_readable() -> None:
    redis: Final = MagicMock(spec=RedisCache)
    redis.async_increment_pipeline = AsyncMock(side_effect=ConnectionError("offline"))
    store: Final = RouterUsageStore(redis)
    await store.increment_deployment("dep", "model", "12-34", 5, 1, None)
    assert await store.async_batch_get(
        (UsageCounter("dep", "model", "12-34", "tpm"), UsageCounter("dep", "model", "12-34", "rpm"))
    ) == [5, 1]
