# What is this?
## Unit tests for 'dynamic_rate_limiter.py`
import asyncio
import os
import random
import sys
import time
import traceback
from litellm._uuid import uuid
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
from litellm.proxy._types import UserAPIKeyAuth
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


@pytest.fixture
def user_api_key_auth() -> UserAPIKeyAuth:
    return UserAPIKeyAuth()


@pytest.mark.parametrize("num_projects", [1, 2, 100])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
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

    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    availability = resp[0]

    expected_availability = int(model_tpm / num_projects)

    assert availability == expected_availability


@pytest.mark.parametrize("num_projects", [1, 2, 100])
@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_available_rpm(num_projects, dynamic_rate_limit_handler):
    model = "my-fake-model"
    ## SET CACHE W/ ACTIVE PROJECTS
    projects = [str(uuid.uuid4()) for _ in range(num_projects)]

    await dynamic_rate_limit_handler.internal_usage_cache.async_set_cache_sadd(
        model=model, value=projects
    )

    model_rpm = 100
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "rpm": model_rpm,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    ## CHECK AVAILABLE rpm PER PROJECT

    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    availability = resp[1]

    expected_availability = int(model_rpm / num_projects)

    assert availability == expected_availability


@pytest.mark.parametrize("usage", ["rpm", "tpm"])
@pytest.mark.asyncio
async def test_rate_limit_raised(dynamic_rate_limit_handler, user_api_key_auth, usage):
    """
    Unit test. Tests if rate limit error raised when quota exhausted.
    """
    from fastapi import HTTPException

    model = "my-fake-model"
    ## SET CACHE W/ ACTIVE PROJECTS
    projects = [str(uuid.uuid4())]

    await dynamic_rate_limit_handler.internal_usage_cache.async_set_cache_sadd(
        model=model, value=projects
    )

    model_usage = 0
    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    usage: model_usage,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    ## CHECK AVAILABLE TPM PER PROJECT

    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    if usage == "tpm":
        availability = resp[0]
    else:
        availability = resp[1]

    expected_availability = 0

    assert availability == expected_availability

    ## CHECK if exception raised

    try:
        await dynamic_rate_limit_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_auth,
            cache=DualCache(),
            data={"model": model},
            call_type="completion",
        )
        pytest.fail("Expected this to raise HTTPexception")
    except HTTPException as e:
        assert e.status_code == 429  # check if rate limit error raised
        pass


@pytest.mark.asyncio
async def test_base_case(dynamic_rate_limit_handler, mock_response):
    """
    If just 1 active project

    it should get all the quota

    = allow request to go through
    - update token usage
    - exhaust all tpm with just 1 project
    - assert ratelimiterror raised at 100%+1 tpm
    """
    model = "my-fake-model"
    ## model tpm - 50
    model_tpm = 50
    ## tpm per request - 10
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
    allowed_fails = 1
    for _ in range(2):
        try:
            # check availability
            resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

            availability = resp[0]

            print(
                "prev_availability={}, availability={}".format(
                    prev_availability, availability
                )
            )

            ## assert availability updated
            if prev_availability is not None and availability is not None:
                assert availability == prev_availability - 10

            prev_availability = availability

            # make call
            await llm_router.acompletion(
                model=model, messages=[{"role": "user", "content": "hey!"}]
            )

            await asyncio.sleep(3)
        except Exception:
            if allowed_fails > 0:
                allowed_fails -= 1
            else:
                raise


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_update_cache(
    dynamic_rate_limit_handler, mock_response, user_api_key_auth
):
    """
    Check if active project correctly updated
    """
    model = "my-fake-model"
    model_tpm = 50

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

    ## INITIAL ACTIVE PROJECTS - ASSERT NONE
    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    active_projects = resp[-1]

    assert active_projects is None

    ## MAKE CALL
    await dynamic_rate_limit_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_auth,
        cache=DualCache(),
        data={"model": model},
        call_type="completion",
    )

    await asyncio.sleep(2)
    ## INITIAL ACTIVE PROJECTS - ASSERT 1
    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    active_projects = resp[-1]

    assert active_projects == 1


@pytest.mark.skip(
    reason="Unstable on ci/cd due to curr minute changes. Refactor to handle minute changing"
)
@pytest.mark.parametrize("num_projects", [2])
@pytest.mark.asyncio
async def test_multiple_projects(
    dynamic_rate_limit_handler, mock_response, num_projects
):
    """
    If 2 active project

    it should split 50% each

    - assert available tpm is 0 after 50%+1 tpm calls
    """
    model = "my-fake-model"
    model_tpm = 50
    total_tokens_per_call = 10
    step_tokens_per_call_per_project = total_tokens_per_call / num_projects

    available_tpm_per_project = int(model_tpm / num_projects)

    ## SET CACHE W/ ACTIVE PROJECTS
    projects = [str(uuid.uuid4()) for _ in range(num_projects)]
    await dynamic_rate_limit_handler.internal_usage_cache.async_set_cache_sadd(
        model=model, value=projects
    )

    expected_runs = int(available_tpm_per_project / step_tokens_per_call_per_project)

    setattr(
        mock_response,
        "usage",
        litellm.Usage(
            prompt_tokens=5, completion_tokens=5, total_tokens=total_tokens_per_call
        ),
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

    print("expected_runs: {}".format(expected_runs))

    for i in range(expected_runs + 1):
        # check availability

        resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

        availability = resp[0]

        ## assert availability updated
        if prev_availability is not None and availability is not None:
            assert (
                availability == prev_availability - step_tokens_per_call_per_project
            ), "Current Availability: Got={}, Expected={}, Step={}, Tokens per step={}, Initial model tpm={}".format(
                availability,
                prev_availability - 10,
                i,
                step_tokens_per_call_per_project,
                model_tpm,
            )

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

    # check availability
    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    availability = resp[0]

    assert availability == 0


@pytest.mark.parametrize("num_projects", [1, 2, 100])
@pytest.mark.asyncio
async def test_priority_reservation(num_projects, dynamic_rate_limit_handler):
    """
    If reservation is set + `mock_testing_reservation` passed in

    assert correct rpm is reserved
    """
    model = "my-fake-model"
    ## SET CACHE W/ ACTIVE PROJECTS
    projects = [str(uuid.uuid4()) for _ in range(num_projects)]

    await dynamic_rate_limit_handler.internal_usage_cache.async_set_cache_sadd(
        model=model, value=projects
    )

    litellm.priority_reservation = {"dev": 0.1, "prod": 0.9}

    model_usage = 100

    llm_router = Router(
        model_list=[
            {
                "model_name": model,
                "litellm_params": {
                    "model": "gpt-3.5-turbo",
                    "api_key": "my-key",
                    "api_base": "my-base",
                    "rpm": model_usage,
                },
            }
        ]
    )
    dynamic_rate_limit_handler.update_variables(llm_router=llm_router)

    ## CHECK AVAILABLE TPM PER PROJECT

    resp = await dynamic_rate_limit_handler.check_available_usage(
        model=model, priority="prod"
    )

    availability = resp[1]

    expected_availability = int(
        model_usage * litellm.priority_reservation["prod"] / num_projects
    )

    assert availability == expected_availability


@pytest.mark.skip(
    reason="Unstable on ci/cd due to curr minute changes. Refactor to handle minute changing"
)
@pytest.mark.parametrize("num_projects", [2])
@pytest.mark.asyncio
async def test_multiple_projects_e2e(
    dynamic_rate_limit_handler, mock_response, num_projects
):
    """
    2 parallel calls with different keys, same model

    If 2 active project

    it should split 50% each

    - assert available tpm is 0 after 50%+1 tpm calls
    """
    model = "my-fake-model"
    model_tpm = 50
    total_tokens_per_call = 10
    step_tokens_per_call_per_project = total_tokens_per_call / num_projects

    available_tpm_per_project = int(model_tpm / num_projects)

    ## SET CACHE W/ ACTIVE PROJECTS
    projects = [str(uuid.uuid4()) for _ in range(num_projects)]
    await dynamic_rate_limit_handler.internal_usage_cache.async_set_cache_sadd(
        model=model, value=projects
    )

    expected_runs = int(available_tpm_per_project / step_tokens_per_call_per_project)

    setattr(
        mock_response,
        "usage",
        litellm.Usage(
            prompt_tokens=5, completion_tokens=5, total_tokens=total_tokens_per_call
        ),
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

    print("expected_runs: {}".format(expected_runs))
    for i in range(expected_runs + 1):
        # check availability
        resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

        availability = resp[0]

        ## assert availability updated
        if prev_availability is not None and availability is not None:
            assert (
                availability == prev_availability - step_tokens_per_call_per_project
            ), "Current Availability: Got={}, Expected={}, Step={}, Tokens per step={}, Initial model tpm={}".format(
                availability,
                prev_availability - 10,
                i,
                step_tokens_per_call_per_project,
                model_tpm,
            )

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

    # check availability
    resp = await dynamic_rate_limit_handler.check_available_usage(model=model)

    availability = resp[0]
    assert availability == 0
