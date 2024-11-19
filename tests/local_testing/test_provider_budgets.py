import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os, copy

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
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
        routing_strategy="provider-budget-routing",
        routing_strategy_args=provider_budget_config,
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
