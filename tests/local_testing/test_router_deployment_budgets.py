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
from litellm.router_strategy.provider_budgets import RouterBudgetLimiting
from litellm.types.router import (
    RoutingStrategy,
    ProviderBudgetConfigType,
    ProviderBudgetInfo,
)
from litellm.caching.caching import DualCache, RedisCache
import logging
from litellm._logging import verbose_router_logger
import litellm
from datetime import timezone, timedelta
from test_provider_budgets import cleanup_redis

verbose_router_logger.setLevel(logging.DEBUG)


@pytest.mark.flaky(retries=3, delay=2)
@pytest.mark.asyncio
async def test_deployment_budget_limits_e2e_test():
    """
    Expected behavior:
    - First request forced to openai/gpt-4o
    - Hit budget limit for openai/gpt-4o
    - Next 3 requests all go to openai/gpt-4o-mini

    """
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
        redis_host=os.getenv("REDIS_HOST"),
        redis_port=int(os.getenv("REDIS_PORT")),
        redis_password=os.getenv("REDIS_PASSWORD"),
    )

    response = await router.acompletion(
        messages=[{"role": "user", "content": "Hello, how are you?"}],
        model="gpt-4o",
    )
    print(response)

    await asyncio.sleep(2.5)

    for _ in range(3):
        response = await router.acompletion(
            messages=[{"role": "user", "content": "Hello, how are you?"}],
            model="gpt-4o",
        )
        print(response)

        print("response.hidden_params", response._hidden_params)

        await asyncio.sleep(1)

        assert response._hidden_params.get("custom_llm_provider") == "azure"
