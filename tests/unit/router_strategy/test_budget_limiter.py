"""
Spend tracking in RouterBudgetLimiting.async_log_success_event.

Only chat completions puts custom_llm_provider into litellm_params. The responses,
anthropic_messages, embedding and rerank surfaces leave it unset, which used to make
the callback raise before any spend was recorded, so those budgets never moved.
"""

from typing import Final

import litellm
import pytest

from litellm.caching.caching import DualCache
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.utils import BudgetConfig
import sys, os, asyncio, time, random


@pytest.fixture
def disable_budget_sync(monkeypatch):
    async def noop(*args, **kwargs):
        return None

    monkeypatch.setattr(
        "litellm.router_strategy.budget_limiter.RouterBudgetLimiting.periodic_sync_in_memory_spend_with_redis",
        noop,
    )


def _success_kwargs(
    *,
    provider_in_litellm_params: str | None,
    provider_in_payload: str | None,
    call_type: str = "aresponses",
    response_cost: float = 0.25,
    model_id: str = "deployment-1",
) -> dict[str, object]:
    provider_params: Final[dict[str, str]] = (
        {} if provider_in_litellm_params is None else {"custom_llm_provider": provider_in_litellm_params}
    )
    litellm_params: Final[dict[str, str]] = {"model": "openai/gpt-4o", **provider_params}

    return {
        "call_type": call_type,
        "litellm_params": litellm_params,
        "standard_logging_object": {
            "response_cost": response_cost,
            "model_id": model_id,
            "custom_llm_provider": provider_in_payload,
        },
    }


async def _log_success(limiter: RouterBudgetLimiting, kwargs: dict[str, object]) -> None:
    await limiter.async_log_success_event(kwargs=kwargs, response_obj=None, start_time=None, end_time=None)


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ["aresponses", "anthropic_messages", "aembedding", "arerank"])
async def test_provider_spend_tracked_when_litellm_params_omits_provider(disable_budget_sync, call_type):
    """Non-chat surfaces carry the provider only on the standard logging payload."""
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={"openai": {"budget_limit": 10.0, "time_period": "1d"}},
    )

    await _log_success(
        limiter,
        _success_kwargs(
            provider_in_litellm_params=None,
            provider_in_payload="openai",
            call_type=call_type,
        ),
    )

    assert await limiter.dual_cache.async_get_cache("provider_spend:openai:1d") == 0.25


@pytest.mark.asyncio
async def test_chat_completions_spend_still_tracked(disable_budget_sync):
    """Chat completions fills in both sources and must keep accumulating."""
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={"openai": {"budget_limit": 10.0, "time_period": "1d"}},
    )

    await _log_success(
        limiter,
        _success_kwargs(
            provider_in_litellm_params="openai",
            provider_in_payload="openai",
            call_type="acompletion",
        ),
    )

    assert await limiter.dual_cache.async_get_cache("provider_spend:openai:1d") == 0.25


@pytest.mark.asyncio
async def test_budget_of_other_provider_is_untouched(disable_budget_sync):
    """A provider without its own budget must not bleed into a configured one."""
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={"openai": {"budget_limit": 10.0, "time_period": "1d"}},
    )

    await _log_success(
        limiter,
        _success_kwargs(provider_in_litellm_params=None, provider_in_payload="anthropic"),
    )

    assert await limiter.dual_cache.async_get_cache("provider_spend:openai:1d") in (None, 0.0)


@pytest.mark.asyncio
async def test_deployment_budget_tracked_when_provider_is_unresolvable(disable_budget_sync):
    """An unresolvable provider must not abort the deployment and tag budgets that follow it."""
    limiter = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config=None,
        model_list=[
            {
                "model_name": "some-model",
                "litellm_params": {
                    "model": "openai/gpt-4o",
                    "max_budget": 10.0,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "deployment-1"},
            }
        ],
    )

    await _log_success(
        limiter,
        _success_kwargs(provider_in_litellm_params=None, provider_in_payload=None),
    )

    assert await limiter.dual_cache.async_get_cache("deployment_spend:deployment-1:1d") == 0.25


@pytest.mark.asyncio
async def test_get_budget_config_for_provider():
    """
    Test the _get_budget_config_for_provider helper method

    """
    cleanup_redis()
    config = {
        "openai": BudgetConfig(budget_duration="1d", max_budget=100),
        "anthropic": BudgetConfig(budget_duration="7d", max_budget=500),
    }

    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config=config
    )

    # Test existing providers
    openai_config = provider_budget._get_budget_config_for_provider("openai")
    assert openai_config is not None
    assert openai_config.budget_duration == "1d"
    assert openai_config.max_budget == 100

    anthropic_config = provider_budget._get_budget_config_for_provider("anthropic")
    assert anthropic_config is not None
    assert anthropic_config.budget_duration == "7d"
    assert anthropic_config.max_budget == 500

    # Test non-existent provider
    assert provider_budget._get_budget_config_for_provider("unknown") is None


@pytest.mark.asyncio
async def test_get_current_provider_spend():
    """
    Test _get_current_provider_spend helper method

    Scenarios:
    1. Provider with no budget config returns None
    2. Provider with budget config but no spend returns 0.0
    3. Provider with budget config and spend returns correct value
    """
    cleanup_redis()
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(),
        provider_budget_config={
            "openai": BudgetConfig(time_period="1d", budget_limit=100),
        },
    )

    # Test provider with no budget config
    spend = await provider_budget._get_current_provider_spend("anthropic")
    assert spend is None

    # Test provider with budget config but no spend
    spend = await provider_budget._get_current_provider_spend("openai")
    assert spend == 0.0

    # Test provider with budget config and spend
    spend_key = "provider_spend:openai:1d"
    await provider_budget.dual_cache.async_set_cache(key=spend_key, value=50.5)

    spend = await provider_budget._get_current_provider_spend("openai")
    assert spend == 50.5


def cleanup_redis():
    """Cleanup Redis cache before each test"""
    try:
        import redis

        print("cleaning up redis..")

        redis_client = redis.Redis(
            host=os.getenv("REDIS_HOST"),
            port=int(os.getenv("REDIS_PORT")),
            password=os.getenv("REDIS_PASSWORD"),
        )
        print("scan iter result", redis_client.scan_iter("provider_spend:*"))
        # Delete all provider spend keys
        for key in redis_client.scan_iter("provider_spend:*"):
            print("deleting key", key)
            redis_client.delete(key)
        for key in redis_client.scan_iter("deployment_spend:*"):
            print("deleting key", key)
            redis_client.delete(key)
        for key in redis_client.scan_iter("tag_spend:*"):
            print("deleting key", key)
            redis_client.delete(key)
    except Exception as e:
        print(f"Error cleaning up Redis: {str(e)}")


def _completed_budget_batch(batch_id: str):
    from litellm.types.utils import LiteLLMBatch

    return LiteLLMBatch(
        id=batch_id,
        completion_window="24h",
        created_at=1,
        endpoint="/v1/chat/completions",
        input_file_id="file-budget",
        object="batch",
        status="completed",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("call_type", ("aretrieve_batch", "retrieve_batch"))
async def test_batch_polls_charge_each_router_budget_once(monkeypatch, call_type):
    monkeypatch.setattr(litellm, "tag_budget_config", {"release": {"budget_limit": 100, "time_period": "1d"}})
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    cache: Final = DualCache()
    limiter: Final = RouterBudgetLimiting(
        dual_cache=cache,
        provider_budget_config={"openai": {"budget_limit": 100, "time_period": "1d"}},
        model_list=[
            {
                "model_name": "batch-budget",
                "litellm_params": {"model": "openai/batch-budget", "max_budget": 100, "budget_duration": "1d"},
                "model_info": {"id": "deployment-1"},
            }
        ],
    )
    kwargs: Final = {
        **_success_kwargs(provider_in_litellm_params=None, provider_in_payload="openai", call_type=call_type),
        "metadata": {"tags": ["release"]},
    }
    response: Final = _completed_budget_batch("batch-first")
    await asyncio.gather(*(limiter.async_log_success_event(kwargs, response, None, None) for _ in range(10)))
    keys: Final = ("provider_spend:openai:1d", "deployment_spend:deployment-1:1d", "tag_spend:release:1d")
    assert tuple([await cache.async_get_cache(key) for key in keys]) == (0.25, 0.25, 0.25)
    await limiter.async_log_success_event(kwargs, _completed_budget_batch("batch-second"), None, None)
    await limiter.async_log_success_event({**kwargs, "call_type": "acompletion"}, response, None, None)
    await limiter.async_log_success_event({**kwargs, "call_type": "acompletion"}, response, None, None)
    assert tuple([await cache.async_get_cache(key) for key in keys]) == (1.0, 1.0, 1.0)
    await limiter.async_log_success_event({**kwargs, "call_type": "_arealtime"}, response, None, None)
    assert tuple([await cache.async_get_cache(key) for key in keys]) == (1.0, 1.0, 1.0)


@pytest.mark.asyncio
async def test_batch_charge_is_shared_across_workers_and_requires_redis_confirmation():
    from fakeredis import FakeServer, FakeRedis as FakeSyncRedis
    from redis.exceptions import ConnectionError as RedisConnectionError
    from fakeredis.aioredis import FakeRedis
    from litellm.caching.redis_cache import RedisCache

    server: Final = FakeServer()
    client: Final = FakeRedis(server=server)
    sync_client: Final = FakeSyncRedis(server=server)
    redis_cache: Final = RedisCache(host="router-batch-fixture", connection_pool=sync_client.connection_pool)
    litellm.in_memory_llm_clients_cache.set_cache(key=redis_cache._get_async_client_cache_key(), value=client)
    first: Final = RouterBudgetLimiting(
        dual_cache=DualCache(redis_cache=redis_cache),
        provider_budget_config={"openai": {"budget_limit": 100, "time_period": "1d"}},
    )
    second: Final = RouterBudgetLimiting(
        dual_cache=DualCache(redis_cache=redis_cache),
        provider_budget_config={"openai": {"budget_limit": 100, "time_period": "1d"}},
    )
    kwargs: Final = _success_kwargs(
        provider_in_litellm_params=None, provider_in_payload="openai", call_type="aretrieve_batch"
    )
    response: Final = _completed_budget_batch("batch-shared")
    await asyncio.gather(
        first.async_log_success_event(kwargs, response, None, None),
        second.async_log_success_event(kwargs, response, None, None),
    )
    await first._push_in_memory_increments_to_redis()
    await second._push_in_memory_increments_to_redis()
    assert float(await client.get("provider_spend:openai:1d")) == 0.25
    server.connected = False
    with pytest.raises(RedisConnectionError):
        await second.async_log_success_event(kwargs, _completed_budget_batch("batch-unconfirmed"), None, None)
    server.connected = True
    await second.async_log_success_event(kwargs, _completed_budget_batch("batch-unconfirmed"), None, None)
    await second._push_in_memory_increments_to_redis()
    assert float(await client.get("provider_spend:openai:1d")) == 0.5
    await client.aclose()
