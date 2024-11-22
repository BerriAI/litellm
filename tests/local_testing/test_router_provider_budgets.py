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
from litellm.router_strategy.provider_budgets import ProviderBudgetLimiting
from litellm.types.router import (
    RoutingStrategy,
    ProviderBudgetConfigType,
    ProviderBudgetInfo,
)
from litellm.caching.caching import DualCache
import logging
from litellm._logging import verbose_router_logger
import litellm

verbose_router_logger.setLevel(logging.DEBUG)


@pytest.mark.asyncio
async def test_provider_budgets_e2e_test():
    """
    Expected behavior:
    - First request forced to OpenAI
    - Hit OpenAI budget limit
    - Next 3 requests all go to Azure

    """
    provider_budget_config: ProviderBudgetConfigType = {
        "openai": ProviderBudgetInfo(time_period="1d", budget_limit=0.000000000001),
        "azure": ProviderBudgetInfo(time_period="1d", budget_limit=100),
    }

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
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

    await asyncio.sleep(0.5)

    for _ in range(3):
        response = await router.acompletion(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model="gpt-3.5-turbo",
        )
        print(response)

        print("response.hidden_params", response._hidden_params)

        await asyncio.sleep(0.5)

        assert response._hidden_params.get("custom_llm_provider") == "azure"


@pytest.mark.asyncio
async def test_provider_budgets_e2e_test_expect_to_fail():
    """
    Expected behavior:
    - first request passes, all subsequent requests fail

    """

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

    await asyncio.sleep(0.5)

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


def test_get_ttl_seconds():
    """
    Test the get_ttl_seconds helper method"

    """
    provider_budget = ProviderBudgetLimiting(
        router_cache=DualCache(), provider_budget_config={}
    )

    assert provider_budget.get_ttl_seconds("1d") == 86400  # 1 day in seconds
    assert provider_budget.get_ttl_seconds("7d") == 604800  # 7 days in seconds
    assert provider_budget.get_ttl_seconds("30d") == 2592000  # 30 days in seconds

    with pytest.raises(ValueError, match="Unsupported time period format"):
        provider_budget.get_ttl_seconds("1h")


def test_get_llm_provider_for_deployment():
    """
    Test the _get_llm_provider_for_deployment helper method

    """
    provider_budget = ProviderBudgetLimiting(
        router_cache=DualCache(), provider_budget_config={}
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


def test_get_budget_config_for_provider():
    """
    Test the _get_budget_config_for_provider helper method

    """
    config = {
        "openai": ProviderBudgetInfo(time_period="1d", budget_limit=100),
        "anthropic": ProviderBudgetInfo(time_period="7d", budget_limit=500),
    }

    provider_budget = ProviderBudgetLimiting(
        router_cache=DualCache(), provider_budget_config=config
    )

    # Test existing providers
    openai_config = provider_budget._get_budget_config_for_provider("openai")
    assert openai_config is not None
    assert openai_config.time_period == "1d"
    assert openai_config.budget_limit == 100

    anthropic_config = provider_budget._get_budget_config_for_provider("anthropic")
    assert anthropic_config is not None
    assert anthropic_config.time_period == "7d"
    assert anthropic_config.budget_limit == 500

    # Test non-existent provider
    assert provider_budget._get_budget_config_for_provider("unknown") is None


@pytest.mark.asyncio
async def test_prometheus_metric_tracking():
    """
    Test that the Prometheus metric for provider budget is tracked correctly
    """
    from unittest.mock import MagicMock
    from litellm.integrations.prometheus import PrometheusLogger

    # Create a mock PrometheusLogger
    mock_prometheus = MagicMock(spec=PrometheusLogger)

    # Setup provider budget limiting
    provider_budget = ProviderBudgetLimiting(
        router_cache=DualCache(),
        provider_budget_config={
            "openai": ProviderBudgetInfo(time_period="1d", budget_limit=100)
        },
    )

    litellm._async_success_callback = [mock_prometheus]

    provider_budget_config: ProviderBudgetConfigType = {
        "openai": ProviderBudgetInfo(time_period="1d", budget_limit=0.000000000001),
        "azure": ProviderBudgetInfo(time_period="1d", budget_limit=100),
    }

    router = Router(
        model_list=[
            {
                "model_name": "gpt-3.5-turbo",  # openai model name
                "litellm_params": {  # params for litellm completion/embedding call
                    "model": "azure/chatgpt-v-2",
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

    try:
        response = await router.acompletion(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model="openai/gpt-4o-mini",
            mock_response="hi",
        )
        print(response)
    except Exception as e:
        print("error", e)

    await asyncio.sleep(0.5)

    # Verify the mock was called correctly
    mock_prometheus.track_provider_remaining_budget.assert_called_once()
