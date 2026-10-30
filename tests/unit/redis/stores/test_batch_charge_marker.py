from collections.abc import Iterator
from datetime import timedelta
from typing import Final, Protocol, cast
from unittest.mock import patch

import fakeredis
import pytest
from redis.exceptions import ConnectionError as RedisConnectionError
from redis.exceptions import ResponseError

from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.hooks.model_max_budget_limiter import PROXY_VirtualKeyModelMaxBudgetLimiter
from litellm.redis.stores.batch_charge_marker import TransportBatchChargeMarkerStore
from litellm.redis.transport.contract import RedisFailure
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting, budget_window_seconds
from litellm.types.utils import LiteLLMBatch
from tests._support.fake_redis import fake_redis_cache

NOW: Final = 1800000000.0
PROVIDER_SPEND: Final = "provider_spend:openai:1d"
KEY_SPEND: Final = "virtual_key_spend:key-hash-batch:batch-qa-primary:1d"
BATCH_COST: Final = 0.25


class LegacyIncrement(Protocol):
    async def async_increment(
        self, key: str, value: float, ttl: int | None = None, refresh_ttl: bool = False
    ) -> float: ...


@pytest.fixture(autouse=True)
def clock() -> Iterator[None]:
    with patch("time.time", return_value=NOW):
        yield


def _batch(batch_id: str) -> LiteLLMBatch:
    return LiteLLMBatch(
        id=batch_id,
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id="file-batch",
        object="batch",
        status="completed",
    )


def _router_limiter(cache: RedisCache) -> RouterBudgetLimiting:
    return RouterBudgetLimiting(
        dual_cache=DualCache(redis_cache=cache),
        provider_budget_config={"openai": {"budget_limit": 100, "time_period": "1d"}},
        clock=lambda: NOW,
    )


async def _router_poll(limiter: RouterBudgetLimiting, batch_id: str) -> None:
    kwargs: Final = {
        "call_type": "aretrieve_batch",
        "litellm_params": {"model": "openai/gpt-4o"},
        "standard_logging_object": {
            "response_cost": BATCH_COST,
            "model_id": "deployment-1",
            "custom_llm_provider": "openai",
        },
    }
    await limiter.async_log_success_event(kwargs, _batch(batch_id), None, None)


def _model_limiter(cache: RedisCache) -> PROXY_VirtualKeyModelMaxBudgetLimiter:
    return PROXY_VirtualKeyModelMaxBudgetLimiter(dual_cache=DualCache(redis_cache=cache))


async def _model_poll(limiter: PROXY_VirtualKeyModelMaxBudgetLimiter, batch_id: str) -> None:
    budget: Final = {"batch-qa-primary": {"budget_limit": 100, "time_period": "1d"}}
    event: Final = {
        "call_type": "aretrieve_batch",
        "standard_logging_object": {
            "call_type": "aretrieve_batch",
            "response_cost": BATCH_COST,
            "model": "openai/gpt-5.4-mini",
            "model_group": "batch-qa-primary",
            "metadata": {"user_api_key_hash": "key-hash-batch"},
        },
        "litellm_params": {"metadata": {"user_api_key_model_max_budget": budget}},
    }
    await limiter.async_log_success_event(event, response_obj=_batch(batch_id), start_time=None, end_time=None)


async def _shared_spend(cache: RedisCache, key: str, *limiters: RouterBudgetLimiting) -> float:
    _: Final = [
        await limiter._push_in_memory_increments_to_redis()  # pyright: ignore[reportPrivateUsage]  # flushes queued spend the way the periodic sync does
        for limiter in limiters
    ]
    spend: Final = await cache.transport().get(key)
    assert not isinstance(spend, RedisFailure)
    return 0.0 if spend is None else float(spend)


def _local_spend(limiter: PROXY_VirtualKeyModelMaxBudgetLimiter, key: str) -> float:
    return limiter.dual_cache.in_memory_cache.get_cache(key=key) or 0.0


@pytest.mark.asyncio
async def test_router_marker_bytes_ttl_and_namespace_are_shared_by_workers(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server, namespace="tenant")
    first: Final = _router_limiter(cache)
    second: Final = _router_limiter(cache)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    marker: Final = "tenant:" + PROVIDER_SPEND + ":batch:batch-shared"
    await _router_poll(first, "batch-shared")
    assert await raw.get(marker) == b"1"
    assert await raw.pttl(marker) == budget_window_seconds(NOW, "1d") * 1000
    await _router_poll(second, "batch-shared")
    assert await raw.get(marker) == b"2"
    assert await _shared_spend(cache, PROVIDER_SPEND, first, second) == BATCH_COST
    await _router_poll(second, "batch-other")
    assert await _shared_spend(cache, PROVIDER_SPEND, second) == 2 * BATCH_COST


@pytest.mark.asyncio
async def test_old_and_new_pods_see_each_others_markers(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    legacy: Final = cast(LegacyIncrement, cache)  # cast-ok: rolling-deploy writer on the same keyspace
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await legacy.async_increment(PROVIDER_SPEND + ":batch:batch-old", 1, ttl=86400, refresh_ttl=True)
    limiter: Final = _router_limiter(cache)
    await _router_poll(limiter, "batch-old")
    assert await _shared_spend(cache, PROVIDER_SPEND, limiter) == 0.0
    await _router_poll(limiter, "batch-new")
    assert await _shared_spend(cache, PROVIDER_SPEND, limiter) == BATCH_COST
    assert await legacy.async_increment(PROVIDER_SPEND + ":batch:batch-new", 1, ttl=86400, refresh_ttl=True) == 2.0
    assert await raw.get(PROVIDER_SPEND + ":batch:batch-new") == b"2"


@pytest.mark.asyncio
async def test_router_marker_failures_still_raise_and_charge_nothing(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    limiter: Final = _router_limiter(cache)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await raw.set(PROVIDER_SPEND + ":batch:batch-bad", b"not-a-number")
    with pytest.raises(ResponseError):
        await _router_poll(limiter, "batch-bad")
    server.connected = False
    with pytest.raises(RedisConnectionError):
        await _router_poll(limiter, "batch-down")
    server.connected = True
    assert await _shared_spend(cache, PROVIDER_SPEND, limiter) == 0.0


@pytest.mark.asyncio
async def test_model_budget_marker_is_shared_and_falls_back_to_the_local_marker(server: fakeredis.FakeServer) -> None:
    cache: Final = fake_redis_cache(server)
    first: Final = _model_limiter(cache)
    second: Final = _model_limiter(cache)
    raw: Final = fakeredis.FakeAsyncRedis(server=server)
    await _model_poll(first, "batch-shared")
    await _model_poll(second, "batch-shared")
    assert _local_spend(first, KEY_SPEND) + _local_spend(second, KEY_SPEND) == BATCH_COST
    assert await raw.get(KEY_SPEND + ":batch:batch-shared") == b"2"
    assert await raw.pttl(KEY_SPEND + ":batch:batch-shared") == 86400000
    await raw.set(KEY_SPEND + ":batch:batch-bad", b"not-a-number")
    await _model_poll(second, "batch-bad")
    await _model_poll(second, "batch-bad")
    assert _local_spend(first, KEY_SPEND) + _local_spend(second, KEY_SPEND) == 2 * BATCH_COST
    assert await raw.get(KEY_SPEND + ":batch:batch-bad") == b"not-a-number"


@pytest.mark.asyncio
async def test_marker_outage_is_a_transport_failure_and_model_budgets_fall_back_locally() -> None:
    server: Final = fakeredis.FakeServer()
    cache: Final = fake_redis_cache(server)
    store: Final = TransportBatchChargeMarkerStore(cache.transport())
    server.connected = False
    outage: Final = await store.claim_batch(KEY_SPEND, "batch-down", ttl=timedelta(days=1))
    assert isinstance(outage, RedisFailure) and outage.kind == "transport"
    limiter: Final = _model_limiter(cache)
    await _model_poll(limiter, "batch-down")
    await _model_poll(limiter, "batch-down")
    assert _local_spend(limiter, KEY_SPEND) == BATCH_COST
