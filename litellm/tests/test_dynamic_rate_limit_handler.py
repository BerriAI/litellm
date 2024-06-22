# What is this?
## Unit tests for 'dynamic_rate_limiter.py`
import asyncio
import os
import random
import sys
import time
import traceback
import uuid
from datetime import datetime
from typing import Optional, Tuple

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


@pytest.fixture
def mock_response() -> litellm.ModelResponse:
    return litellm.ModelResponse(
        **{
            "id": "chatcmpl-abc123",
            "object": "chat.completion",
            "created": 1699896916,
            "model": "gpt-3.5-turbo-0125",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_abc123",
                                "type": "function",
                                "function": {
                                    "name": "get_current_weather",
                                    "arguments": '{\n"location": "Boston, MA"\n}',
                                },
                            }
                        ],
                    },
                    "logprobs": None,
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        }
    )


@pytest.mark.parametrize("num_projects", [1, 2, 100])
@pytest.mark.asyncio
async def test_available_tpm(num_projects, dynamic_rate_limit_handler):
    model = "my-fake-model"
    ## SET CACHE W/ ACTIVE PROJECTS
    projects = [str(uuid.uuid4()) for _ in range(num_projects)]

    await dynamic_rate_limit_handler.internal_usage_cache.async_set_cache_sadd(
        model=model, value=projects
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

    availability, _, _ = await dynamic_rate_limit_handler.check_available_tpm(
        model=model
    )

    expected_availability = int(model_tpm / num_projects)

    assert availability == expected_availability


@pytest.mark.asyncio
async def test_base_case(dynamic_rate_limit_handler, mock_response):
    """
    If just 1 active project

    it should get all the quota

    = allow request to go through
    - update token usage
    - exhaust all tpm with just 1 project
    """
    model = "my-fake-model"
    model_tpm = 50
    setattr(
        mock_response,
        "usage",
        litellm.Usage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
    )

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "tpm": model_tpm,
                    "mock_response": mock_response,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    prev_availability: Optional[int] = None
    for _ in range(5):
        # check availability
        availability, _, _ = await dynamic_rate_limit_handler.check_available_tpm(
            model=model
        )

        ## assert availability updated
        if prev_availability is not None and availability is not None:
            assert availability == prev_availability - 10

        print(
            "prev_availability={}, availability={}".format(
                prev_availability, availability
            )
        )

        prev_availability = availability

        # make call
        await llm_router.acompletion(
            model=model, messages=[{"role": "user", "content": "hey!"}]
        )

        await asyncio.sleep(3)
