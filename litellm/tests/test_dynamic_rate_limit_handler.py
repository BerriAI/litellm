# What is this?
## Unit tests for 'dynamic_rate_limiter.py`
import asyncio
import os
import random
import sys
import time
import traceback
from datetime import datetime
from typing import Tuple

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest

import litellm
from litellm import DualCache, Router
from litellm.proxy.hooks.dynamic_rate_limiter import (
    _PROXY_DynamicRateLimitHandler as DynamicRateLimitHandler,
)

"""
Basic test cases:

- If 1 'active' project => give all tpm
- If 2 'active' projects => divide tpm in 2
"""


@pytest.fixture
def dynamic_rate_limit_handler() -> DynamicRateLimitHandler:
    internal_cache = DualCache()
    return DynamicRateLimitHandler(internal_usage_cache=internal_cache)


@pytest.mark.parametrize("num_projects", [1, 2, 100])
@pytest.mark.asyncio
async def test_available_tpm(num_projects, dynamic_rate_limit_handler):
    model = "my-fake-model"
    ## SET CACHE W/ ACTIVE PROJECTS
    await dynamic_rate_limit_handler.internal_usage_cache.async_increment_cache(
        model=model, value=num_projects
    )

    model_tpm = 100
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "tpm": model_tpm,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    ## CHECK AVAILABLE TPM PER PROJECT

    availability = await dynamic_rate_limit_handler.check_available_tpm(model=model)

    expected_availability = int(model_tpm / num_projects)

    assert availability == expected_availability
