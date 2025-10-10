import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, copy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system-path
import pytest
from litellm import Router
from litellm.router_strategy.budget_limiter import RouterBudgetLimiting
from litellm.types.router import (
    RoutingStrategy,
)
from litellm.types.utils import GenericBudgetConfigType, BudgetConfig
from litellm.caching.caching import DualCache, RedisCache
import logging
from litellm._logging import verbose_router_logger
import litellm
from datetime import timezone, timedelta

verbose_router_logger.setLevel(logging.DEBUG)


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


@pytest.mark.flaky(retries=6, delay=2)
@pytest.mark.asyncio
async def test_provider_budgets_e2e_test():
    """
    Expected behavior:
    - First request forced to OpenAI
    - Hit OpenAI budget limit
    - Next 3 requests all go to Azure

    """
    cleanup_redis()
    # Modify for test
    provider_budget_config: GenericBudgetConfigType = {
        "openai": BudgetConfig(time_period="1d", budget_limit=0.000000000001),
        "azure": BudgetConfig(time_period="1d", budget_limit=100),
    }

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/gpt-4.1-nano",
                    "api_key": os.getenv("AZURE_API_KEY"),
                    "api_version": os.getenv("AZURE_API_VERSION"),
                    "api_base": os.getenv("AZURE_API_BASE"),
                },
                "model_info": {"id": "azure-model-id"},
            },
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                },
                "model_info": {"id": "openai-model-id"},
            },
        ],
        provider_budget_config=provider_budget_config,
        redis_host=os.getenv("REDIS_HOST"),
        redis_port=int(os.getenv("REDIS_PORT")),
        redis_password=os.getenv("REDIS_PASSWORD"),
    )

    response = await router.acompletion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        model="openai/gpt-4o-mini",
    )
    print(response)

    await asyncio.sleep(2.5)

    for _ in range(3):
        response = await router.acompletion(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model="gpt-3.5-turbo",
        )
        print(response)

        print("response.hidden_params", response._hidden_params)

        await asyncio.sleep(1)

        assert response._hidden_params.get("custom_llm_provider") == "azure"


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=2)
async def test_provider_budgets_e2e_test_expect_to_fail():
    """
    Expected behavior:
    - first request passes, all subsequent requests fail

    """
    cleanup_redis()

    # Note: We intentionally use a dictionary with string keys for budget_limit and time_period
    # we want to test that the router can handle type conversion, since the proxy config yaml passes these values as a dictionary
    provider_budget_config = {
        "anthropic": {
            "budget_limit": 0.000000000001,
            "time_period": "1d",
        }
    }

    router = Router(
        model_list=[
            {
                "model_name": "anthropic/*",  # openai model name
                "litellm_params": {
                    "model": "anthropic/*",
                },
            },
        ],
        redis_host=os.getenv("REDIS_HOST"),
        redis_port=int(os.getenv("REDIS_PORT")),
        redis_password=os.getenv("REDIS_PASSWORD"),
        provider_budget_config=provider_budget_config,
    )

    response = await router.acompletion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        model="anthropic/claude-3-5-sonnet-20240620",
    )
    print(response)

    await asyncio.sleep(2.5)

    for _ in range(3):
        with pytest.raises(Exception) as exc_info:
            response = await router.acompletion(
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="anthropic/claude-3-5-sonnet-20240620",
            )
            print(response)
            print("response.hidden_params", response._hidden_params)

        await asyncio.sleep(0.5)
        # Verify the error is related to budget exceeded

        assert "Exceeded budget for provider" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_llm_provider_for_deployment():
    """
    Test the _get_llm_provider_for_deployment helper method

    """
    cleanup_redis()
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    # Test OpenAI deployment
    openai_deployment = {"litellm_params": {"model": "openai/gpt-4"}}
    assert (
        provider_budget._get_llm_provider_for_deployment(openai_deployment) == "openai"
    )

    # Test Azure deployment
    azure_deployment = {
        "litellm_params": {
            "model": "azure/gpt-4",
            "api_key": "test",
            "api_base": "test",
        }
    }
    assert provider_budget._get_llm_provider_for_deployment(azure_deployment) == "azure"

    # should not raise error for unknown deployment
    unknown_deployment = {}
    assert provider_budget._get_llm_provider_for_deployment(unknown_deployment) is None


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
async def test_handle_new_budget_window():
    """
    Test _handle_new_budget_window helper method

    Current
    """
    cleanup_redis()
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    spend_key = "provider_spend:openai:7d"
    start_time_key = "provider_budget_start_time:openai"
    current_time = 1000.0
    response_cost = 0.5
    ttl_seconds = 86400  # 1 day

    # Test handling new budget window
    new_start_time = await provider_budget._handle_new_budget_window(
        spend_key=spend_key,
        start_time_key=start_time_key,
        current_time=current_time,
        response_cost=response_cost,
        ttl_seconds=ttl_seconds,
    )

    assert new_start_time == current_time

    # Verify the spend was set correctly
    spend = await provider_budget.dual_cache.async_get_cache(spend_key)
    print("spend in cache for key", spend_key, "is", spend)
    assert float(spend) == response_cost

    # Verify start time was set correctly
    start_time = await provider_budget.dual_cache.async_get_cache(start_time_key)
    print("start time in cache for key", start_time_key, "is", start_time)
    assert float(start_time) == current_time


@pytest.mark.asyncio
async def test_get_or_set_budget_start_time():
    """
    Test _get_or_set_budget_start_time helper method

    scenario 1: no existing start time in cache, should return current time
    scenario 2: existing start time in cache, should return existing start time
    """
    cleanup_redis()
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    start_time_key = "test_start_time"
    current_time = 1000.0
    ttl_seconds = 86400  # 1 day

    # When there is no existing start time, we should set it to the current time
    start_time = await provider_budget._get_or_set_budget_start_time(
        start_time_key=start_time_key,
        current_time=current_time,
        ttl_seconds=ttl_seconds,
    )
    print("budget start time when no existing start time is in cache", start_time)
    assert start_time == current_time

    # When there is an existing start time, we should return it even if the current time is later
    new_current_time = 2000.0
    existing_start_time = await provider_budget._get_or_set_budget_start_time(
        start_time_key=start_time_key,
        current_time=new_current_time,
        ttl_seconds=ttl_seconds,
    )
    print(
        "budget start time when existing start time is in cache, but current time is later",
        existing_start_time,
    )
    assert existing_start_time == current_time  # Should return the original start time


@pytest.mark.asyncio
async def test_increment_spend_in_current_window():
    """
    Test _increment_spend_in_current_window helper method

    Expected behavior:
    - Increment the spend in memory cache
    - Queue the increment operation to Redis
    """
    cleanup_redis()
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(), provider_budget_config={}
    )

    spend_key = "provider_spend:openai:1d"
    response_cost = 0.5
    ttl = 86400  # 1 day

    # Set initial spend
    await provider_budget.dual_cache.async_set_cache(key=spend_key, value=1.0, ttl=ttl)

    # Test incrementing spend
    await provider_budget._increment_spend_in_current_window(
        spend_key=spend_key,
        response_cost=response_cost,
        ttl=ttl,
    )

    # Verify the spend was incremented correctly in memory
    spend = await provider_budget.dual_cache.async_get_cache(spend_key)
    assert float(spend) == 1.5

    # Verify the increment operation was queued for Redis
    print(
        "redis_increment_operation_queue",
        provider_budget.redis_increment_operation_queue,
    )
    assert len(provider_budget.redis_increment_operation_queue) == 1
    queued_op = provider_budget.redis_increment_operation_queue[0]
    assert queued_op["key"] == spend_key
    assert queued_op["increment_value"] == response_cost
    assert queued_op["ttl"] == ttl


@pytest.mark.asyncio
async def test_sync_in_memory_spend_with_redis():
    """
    Test _sync_in_memory_spend_with_redis helper method

    Expected behavior:
    - Push all provider spend increments to Redis
    - Fetch all current provider spend from Redis to update in-memory cache
    """
    cleanup_redis()
    provider_budget_config = {
        "openai": BudgetConfig(time_period="1d", budget_limit=100),
        "anthropic": BudgetConfig(time_period="1d", budget_limit=200),
    }

    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(
            redis_cache=RedisCache(
                host=os.getenv("REDIS_HOST"),
                port=int(os.getenv("REDIS_PORT")),
                password=os.getenv("REDIS_PASSWORD"),
            )
        ),
        provider_budget_config=provider_budget_config,
    )

    # Set some values in Redis
    spend_key_openai = "provider_spend:openai:1d"
    spend_key_anthropic = "provider_spend:anthropic:1d"

    await provider_budget.dual_cache.redis_cache.async_set_cache(
        key=spend_key_openai, value=50.0
    )
    await provider_budget.dual_cache.redis_cache.async_set_cache(
        key=spend_key_anthropic, value=75.0
    )

    # Test syncing with Redis
    await provider_budget._sync_in_memory_spend_with_redis()

    # Verify in-memory cache was updated
    openai_spend = await provider_budget.dual_cache.in_memory_cache.async_get_cache(
        spend_key_openai
    )
    anthropic_spend = await provider_budget.dual_cache.in_memory_cache.async_get_cache(
        spend_key_anthropic
    )

    assert float(openai_spend) == 50.0
    assert float(anthropic_spend) == 75.0


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


@pytest.mark.flaky(retries=6, delay=2)
@pytest.mark.asyncio
async def test_get_current_provider_budget_reset_at():
    """
    Test _get_current_provider_budget_reset_at helper method

    Scenarios:
    1. Provider with no budget config returns None
    2. Provider with budget config but no TTL returns None
    3. Provider with budget config and TTL returns correct ISO timestamp
    """
    cleanup_redis()
    provider_budget = RouterBudgetLimiting(
        dual_cache=DualCache(
            redis_cache=RedisCache(
                host=os.getenv("REDIS_HOST"),
                port=int(os.getenv("REDIS_PORT")),
                password=os.getenv("REDIS_PASSWORD"),
            )
        ),
        provider_budget_config={
            "openai": BudgetConfig(budget_duration="1d", max_budget=100),
            "vertex_ai": BudgetConfig(budget_duration="1h", max_budget=100),
        },
    )

    await asyncio.sleep(2)

    # Test provider with no budget config
    reset_at = await provider_budget._get_current_provider_budget_reset_at("anthropic")
    assert reset_at is None

    # Test provider with budget config but no TTL
    reset_at = await provider_budget._get_current_provider_budget_reset_at("openai")
    assert reset_at is not None
    reset_time = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
    expected_time = datetime.now(timezone.utc) + timedelta(seconds=(24 * 60 * 60))
    time_difference = abs((reset_time - expected_time).total_seconds())
    assert time_difference < 5

    # Test provider with budget config and TTL
    reset_at = await provider_budget._get_current_provider_budget_reset_at("vertex_ai")
    assert reset_at is not None

    # Verify the timestamp format and approximate time
    reset_time = datetime.fromisoformat(reset_at.replace("Z", "+00:00"))
    expected_time = datetime.now(timezone.utc) + timedelta(seconds=3600)

    # Allow for small time differences (within 5 seconds)
    time_difference = abs((reset_time - expected_time).total_seconds())
    assert time_difference < 5


@pytest.mark.asyncio
async def test_deployment_budget_limits_e2e_test():
    """
    Expected behavior:
    - First request forced to openai/gpt-4o
    - Hit budget limit for openai/gpt-4o
    - Next 3 requests all go to openai/gpt-4o-mini

    """
    litellm.set_verbose = True
    cleanup_redis()
    # Modify for test

    router = Router(
        model_list=[
            {
                "model_name": "gpt-4o",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "openai/gpt-4o",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "max_budget": 0.000000000001,
                    "budget_duration": "1d",
                },
                "model_info": {"id": "openai-gpt-4o"},
            },
            {
                "model_name": "gpt-4o",  # openai model name
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "api_key": os.getenv("OPENAI_API_KEY"),
                    "max_budget": 10,
                    "budget_duration": "20d",
                },
                "model_info": {"id": "openai-gpt-4o-mini"},
            },
        ],
    )

    response = await router.acompletion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        model="openai-gpt-4o",
    )
    print(response)

    await asyncio.sleep(2.5)

    for _ in range(3):
        response = await router.acompletion(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model="gpt-4o",
        )
        print(response)
        await asyncio.sleep(1)

        print("response.hidden_params", response._hidden_params)
        assert response._hidden_params.get("model_id") == "openai-gpt-4o-mini"


@pytest.mark.asyncio
async def test_deployment_budgets_e2e_test_expect_to_fail():
    """
    Expected behavior:
    - first request passes, all subsequent requests fail

    """
    cleanup_redis()

    router = Router(
        model_list=[
            {
                "model_name": "openai/gpt-4o-mini",  # openai model name
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                    "max_budget": 0.000000000001,
                    "budget_duration": "1d",
                },
            },
        ],
        redis_host=os.getenv("REDIS_HOST"),
        redis_port=int(os.getenv("REDIS_PORT")),
        redis_password=os.getenv("REDIS_PASSWORD"),
    )

    response = await router.acompletion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        model="openai/gpt-4o-mini",
    )
    print(response)

    await asyncio.sleep(2.5)

    for _ in range(3):
        with pytest.raises(Exception) as exc_info:
            response = await router.acompletion(
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="openai/gpt-4o-mini",
            )
            print(response)
            print("response.hidden_params", response._hidden_params)

        await asyncio.sleep(0.5)
        # Verify the error is related to budget exceeded

        assert "Exceeded budget for deployment" in str(exc_info.value)

@pytest.mark.flaky(retries=6, delay=2)
@pytest.mark.asyncio
async def test_tag_budgets_e2e_test_expect_to_fail():
    """
    Expected behavior:
    - first request passes, all subsequent requests fail

    """
    cleanup_redis()
    TAG_NAME = "product:chat-bot"
    TAG_NAME_2 = "product:chat-bot-2"
    litellm.tag_budget_config = {
        TAG_NAME: BudgetConfig(max_budget=0.000000000001, budget_duration="1d"),
        TAG_NAME_2: BudgetConfig(max_budget=100, budget_duration="1d"),
    }

    router = Router(
        model_list=[
            {
                "model_name": "openai/gpt-4o-mini",  # openai model name
                "litellm_params": {
                    "model": "openai/gpt-4o-mini",
                },
            },
        ],
        redis_host=os.getenv("REDIS_HOST"),
        redis_port=int(os.getenv("REDIS_PORT")),
        redis_password=os.getenv("REDIS_PASSWORD"),
    )

    response = await router.acompletion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        model="openai/gpt-4o-mini",
        metadata={"tags": [TAG_NAME]},
    )
    print(response)

    await asyncio.sleep(2.5)

    for _ in range(3):
        with pytest.raises(Exception) as exc_info:
            response = await router.acompletion(
                messages=[{"role": "user", "content": "Hello, how are you?"}],
                model="openai/gpt-4o-mini",
                metadata={"tags": [TAG_NAME]},
            )
            print(response)
            print("response.hidden_params", response._hidden_params)

        await asyncio.sleep(0.5)
        # Verify the error is related to budget exceeded

        assert f"Exceeded budget for tag='{TAG_NAME}'" in str(exc_info.value)

    # test with tag-2 expect to pass
    for _ in range(2):
        response = await router.acompletion(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model="openai/gpt-4o-mini",
            metadata={"tags": [TAG_NAME_2]},
        )
        print(response)
