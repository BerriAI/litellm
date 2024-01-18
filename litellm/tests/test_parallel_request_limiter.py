# What this tests?
## Unit Tests for the max parallel request limiter for the proxy

import sys, os, asyncio, time, random
from datetime import datetime
import traceback
from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
import pytest
import litellm
from litellm import Router
from litellm.proxy.utils import ProxyLogging
from litellm.proxy._types import UserAPIKeyAuth
from litellm.caching import DualCache
from litellm.proxy.hooks.parallel_request_limiter import MaxParallelRequestsHandler

## On Request received
## On Request success
## On Request failure


@pytest.mark.asyncio
async def test_pre_call_hook():
    """
    Test if cache updated on call being received
    """
    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler()

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    print(
        parallel_request_handler.user_api_key_cache.get_cache(
            key=f"{_api_key}_request_count"
        )
    )
    assert (
        parallel_request_handler.user_api_key_cache.get_cache(
            key=f"{_api_key}_request_count"
        )
        == 1
    )


@pytest.mark.asyncio
async def test_success_call_hook():
    """
    Test if on success, cache correctly decremented
    """
    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler()

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    assert (
        parallel_request_handler.user_api_key_cache.get_cache(
            key=f"{_api_key}_request_count"
        )
        == 1
    )

    kwargs = {"litellm_params": {"metadata": {"user_api_key": _api_key}}}

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs, response_obj="", start_time="", end_time=""
    )

    assert (
        parallel_request_handler.user_api_key_cache.get_cache(
            key=f"{_api_key}_request_count"
        )
        == 0
    )


@pytest.mark.asyncio
async def test_failure_call_hook():
    """
    Test if on failure, cache correctly decremented
    """
    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler()

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    assert (
        parallel_request_handler.user_api_key_cache.get_cache(
            key=f"{_api_key}_request_count"
        )
        == 1
    )

    kwargs = {
        "litellm_params": {"metadata": {"user_api_key": _api_key}},
        "exception": Exception(),
    }

    await parallel_request_handler.async_log_failure_event(
        kwargs=kwargs, response_obj="", start_time="", end_time=""
    )

    assert (
        parallel_request_handler.user_api_key_cache.get_cache(
            key=f"{_api_key}_request_count"
        )
        == 0
    )


"""
Test with Router 
- normal call 
- streaming call 
- bad call 
"""


@pytest.mark.asyncio
async def test_normal_router_call():
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    assert (
        parallel_request_handler.user_api_key_cache.get_cache(
            key=f"{_api_key}_request_count"
        )
        == 1
    )

    # normal call
    response = await router.acompletion(
        model="azure-model",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        metadata={"user_api_key": _api_key},
    )
    await asyncio.sleep(1)  # success is done in a separate thread
    print(f"response: {response}")
    value = parallel_request_handler.user_api_key_cache.get_cache(
        key=f"{_api_key}_request_count"
    )
    print(f"cache value: {value}")

    assert value == 0


@pytest.mark.asyncio
async def test_streaming_router_call():
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    assert (
        parallel_request_handler.user_api_key_cache.get_cache(
            key=f"{_api_key}_request_count"
        )
        == 1
    )

    # streaming call
    response = await router.acompletion(
        model="azure-model",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        stream=True,
        metadata={"user_api_key": _api_key},
    )
    async for chunk in response:
        continue
    await asyncio.sleep(1)  # success is done in a separate thread
    value = parallel_request_handler.user_api_key_cache.get_cache(
        key=f"{_api_key}_request_count"
    )
    print(f"cache value: {value}")

    assert value == 0


@pytest.mark.asyncio
async def test_bad_router_call():
    model_list = [
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-turbo",
                "api_key": "os.environ/AZURE_FRANCE_API_KEY",
                "api_base": "https://openai-france-1234.openai.azure.com",
                "rpm": 1440,
            },
            "model_info": {"id": 1},
        },
        {
            "model_name": "azure-model",
            "litellm_params": {
                "model": "azure/gpt-35-turbo",
                "api_key": "os.environ/AZURE_EUROPE_API_KEY",
                "api_base": "https://my-endpoint-europe-berri-992.openai.azure.com",
                "rpm": 6,
            },
            "model_info": {"id": 2},
        },
    ]
    router = Router(
        model_list=model_list,
        set_verbose=False,
        num_retries=3,
    )  # type: ignore

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    assert (
        parallel_request_handler.user_api_key_cache.get_cache(
            key=f"{_api_key}_request_count"
        )
        == 1
    )

    # bad streaming call
    try:
        response = await router.acompletion(
            model="azure-model",
            messages=[{"role": "user2", "content": "Hey, how's it going?"}],
            stream=True,
            metadata={"user_api_key": _api_key},
        )
    except:
        pass
    value = parallel_request_handler.user_api_key_cache.get_cache(
        key=f"{_api_key}_request_count"
    )
    print(f"cache value: {value}")

    assert value == 0
