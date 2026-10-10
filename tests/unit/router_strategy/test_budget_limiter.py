"""
Spend tracking in RouterBudgetLimiting.async_log_success_event.

Only chat completions puts custom_llm_provider into litellm_params. The responses,
anthropic_messages, embedding and rerank surfaces leave it unset, which used to make
the callback raise before any spend was recorded, so those budgets never moved.
"""

from typing import Final

import litellm
import pytest

from litellm import Router
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
@pytest.mark.parametrize(
    "call_type", ["aresponses", "_aresponses_websocket", "anthropic_messages", "aembedding", "arerank"]
)
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
async def test_tag_spend_is_incremented_for_request_metadata(
    disable_budget_sync: None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(litellm, "tag_budget_config", {"release": {"budget_limit": 5.0, "time_period": "1h"}})
    monkeypatch.setattr("litellm.proxy.proxy_server.premium_user", True)
    limiter = RouterBudgetLimiting(dual_cache=DualCache(), provider_budget_config=None)
    kwargs: Final = {
        **_success_kwargs(
            provider_in_litellm_params=None,
            provider_in_payload=None,
            response_cost=0.73,
        ),
        "metadata": {"tags": ["release"]},
    }

    await _log_success(limiter, kwargs)

    assert await limiter.dual_cache.async_get_cache("tag_spend:release:1h") == 0.73


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


@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment_resolves_provider_prefixes() -> None:
    provider_budget: Final = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    assert (
        provider_budget._get_llm_provider_for_deployment(
            {"litellm_params": {"model": "openai/gpt-4o"}}
        )
        == "openai"
    )
    assert (
        provider_budget._get_llm_provider_for_deployment(
            {"litellm_params": {"model": "azure/gpt-4o", "api_base": "https://example.azure.com"}}
        )
        == "azure"
    )
    assert provider_budget._get_llm_provider_for_deployment({}) is None


@pytest.mark.asyncio
async def test_budget_start_time_is_created_once() -> None:
    provider_budget: Final = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    first_start: Final = await provider_budget._get_or_set_budget_start_time(
        start_time_key="provider_budget_start_time:openai",
        current_time=1000.0,
        ttl_seconds=86400,
    )
    later_start: Final = await provider_budget._get_or_set_budget_start_time(
        start_time_key="provider_budget_start_time:openai",
        current_time=2000.0,
        ttl_seconds=86400,
    )

    assert first_start == 1000.0
    assert later_start == 1000.0


@pytest.mark.asyncio
async def test_new_budget_window_sets_spend_and_start_time() -> None:
    provider_budget: Final = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    start_time: Final = await provider_budget._handle_new_budget_window(
        spend_key="provider_spend:openai:1d",
        start_time_key="provider_budget_start_time:openai",
        current_time=1000.0,
        response_cost=0.5,
        ttl_seconds=86400,
    )

    assert start_time == 1000.0
    assert await provider_budget.dual_cache.async_get_cache("provider_spend:openai:1d") == 0.5
    assert (
        await provider_budget.dual_cache.async_get_cache(
            "provider_budget_start_time:openai"
        )
        == 1000.0
    )


@pytest.mark.asyncio
async def test_spend_increment_updates_memory_and_queues_redis_operation() -> None:
    provider_budget: Final = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )
    await provider_budget.dual_cache.async_set_cache(
        key="provider_spend:openai:1d", value=1.0, ttl=86400
    )

    await provider_budget._increment_spend_in_current_window(
        spend_key="provider_spend:openai:1d",
        response_cost=0.5,
        ttl=86400,
    )

    assert (
        await provider_budget.dual_cache.async_get_cache("provider_spend:openai:1d")
        == 1.5
    )
    assert provider_budget.redis_increment_operation_queue == [
        {
            "key": "provider_spend:openai:1d",
            "increment_value": 0.5,
            "ttl": 86400,
        }
    ]


@pytest.mark.asyncio
async def test_deployment_budget_filter_keeps_unspent_deployment(
    disable_budget_sync: None,
) -> None:
    deployments: Final = [
        {
            "model_name": "shared",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "mock_response": "spent deployment",
                "max_budget": 1.0,
                "budget_duration": "1d",
            },
            "model_info": {"id": "spent"},
        },
        {
            "model_name": "shared",
            "litellm_params": {
                "model": "openai/gpt-4o-mini",
                "mock_response": "available deployment",
                "max_budget": 10.0,
                "budget_duration": "1d",
            },
            "model_info": {"id": "available"},
        },
    ]
    router: Final = Router(
        model_list=deployments,
        provider_budget_config=None,
        num_retries=0,
    )
    await router.cache.async_set_cache(key="deployment_spend:spent:1d", value=1.0)

    response: Final = await router.acompletion(
        model="shared",
        messages=[{"role": "user", "content": "budget check"}],
    )

    assert response.choices[0].message.content == "available deployment"
    assert response._hidden_params["model_id"] == "available"


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
