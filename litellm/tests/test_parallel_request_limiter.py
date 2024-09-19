# What this tests?
## Unit Tests for the max parallel request limiter for the proxy

import asyncio
import os
import random
import sys
import time
import traceback
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
import os

sys.path.insert(
    0, os.path.abspath("../..")
)  # Adds the parent directory to the system path
from datetime import datetime

import pytest

import litellm
from litellm import Router
from litellm.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter import (
    _PROXY_MaxParallelRequestsHandler as MaxParallelRequestsHandler,
)
from litellm.proxy.utils import ProxyLogging, hash_token

## On Request received
## On Request success
## On Request failure


@pytest.mark.asyncio
async def test_global_max_parallel_requests():
    """
    Test if ParallelRequestHandler respects 'global_max_parallel_requests'

    data["metadata"]["global_max_parallel_requests"]
    """
    global_max_parallel_requests = 0
    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=100)
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler(
        internal_usage_cache=local_cache
    )

    for _ in range(3):
        try:
            await parallel_request_handler.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=local_cache,
                data={
                    "metadata": {
                        "global_max_parallel_requests": global_max_parallel_requests
                    }
                },
                call_type="",
            )
            pytest.fail("Expected call to fail")
        except Exception as e:
            pass


@pytest.mark.asyncio
async def test_pre_call_hook():
    """
    Test if cache updated on call being received
    """
    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler(
        internal_usage_cache=local_cache
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"

    print(
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
    )
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 1
    )


@pytest.mark.asyncio
async def test_pre_call_hook_rpm_limits():
    """
    Test if error raised on hitting rpm limits
    """
    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key, max_parallel_requests=1, tpm_limit=9, rpm_limit=1
    )
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler(
        internal_usage_cache=local_cache
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    kwargs = {"litellm_params": {"metadata": {"user_api_key": _api_key}}}

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs,
        response_obj="",
        start_time="",
        end_time="",
    )

    ## Expected cache val: {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}

    try:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={},
            call_type="",
        )

        pytest.fail(f"Expected call to fail")
    except Exception as e:
        assert e.status_code == 429


@pytest.mark.asyncio
async def test_pre_call_hook_rpm_limits_retry_after():
    """
    Test if rate limit error, returns 'retry_after'
    """
    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key, max_parallel_requests=1, tpm_limit=9, rpm_limit=1
    )
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler(
        internal_usage_cache=local_cache
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    kwargs = {"litellm_params": {"metadata": {"user_api_key": _api_key}}}

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs,
        response_obj="",
        start_time="",
        end_time="",
    )

    ## Expected cache val: {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}

    try:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={},
            call_type="",
        )

        pytest.fail(f"Expected call to fail")
    except Exception as e:
        assert e.status_code == 429
        assert hasattr(e, "headers")
        assert "retry-after" in e.headers


@pytest.mark.asyncio
async def test_pre_call_hook_team_rpm_limits():
    """
    Test if error raised on hitting team rpm limits
    """
    litellm.set_verbose = True
    _api_key = "sk-12345"
    _team_id = "unique-team-id"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        max_parallel_requests=1,
        tpm_limit=9,
        rpm_limit=10,
        team_rpm_limit=1,
        team_id=_team_id,
    )
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler(
        internal_usage_cache=local_cache
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    kwargs = {
        "litellm_params": {
            "metadata": {"user_api_key": _api_key, "user_api_key_team_id": _team_id}
        }
    }

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs,
        response_obj="",
        start_time="",
        end_time="",
    )

    print(f"local_cache: {local_cache}")

    ## Expected cache val: {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}

    try:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={},
            call_type="",
        )

        pytest.fail(f"Expected call to fail")
    except Exception as e:
        assert e.status_code == 429


@pytest.mark.asyncio
async def test_pre_call_hook_tpm_limits():
    """
    Test if error raised on hitting tpm limits
    """
    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key, max_parallel_requests=1, tpm_limit=9, rpm_limit=10
    )
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler(
        internal_usage_cache=local_cache
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    kwargs = {"litellm_params": {"metadata": {"user_api_key": _api_key}}}

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=litellm.ModelResponse(usage=litellm.Usage(total_tokens=10)),
        start_time="",
        end_time="",
    )

    ## Expected cache val: {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}

    try:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={},
            call_type="",
        )

        pytest.fail(f"Expected call to fail")
    except Exception as e:
        assert e.status_code == 429


@pytest.mark.asyncio
async def test_pre_call_hook_user_tpm_limits():
    """
    Test if error raised on hitting tpm limits
    """
    local_cache = DualCache()
    # create user with tpm/rpm limits
    user_id = "test-user"
    user_obj = {"tpm_limit": 9, "rpm_limit": 10}

    local_cache.set_cache(key=user_id, value=user_obj)

    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        user_id=user_id,
    )
    res = dict(user_api_key_dict)
    print("dict user", res)

    parallel_request_handler = MaxParallelRequestsHandler(
        internal_usage_cache=local_cache
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    kwargs = {
        "litellm_params": {
            "metadata": {"user_api_key_user_id": user_id, "user_api_key": "gm"}
        }
    }

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=litellm.ModelResponse(usage=litellm.Usage(total_tokens=10)),
        start_time="",
        end_time="",
    )

    ## Expected cache val: {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}

    try:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={},
            call_type="",
        )

        pytest.fail(f"Expected call to fail")
    except Exception as e:
        assert e.status_code == 429


@pytest.mark.asyncio
async def test_success_call_hook():
    """
    Test if on success, cache correctly decremented
    """
    _api_key = "sk-12345"
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler(
        internal_usage_cache=local_cache
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 1
    )

    kwargs = {"litellm_params": {"metadata": {"user_api_key": _api_key}}}

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs, response_obj="", start_time="", end_time=""
    )

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 0
    )


@pytest.mark.asyncio
async def test_failure_call_hook():
    """
    Test if on failure, cache correctly decremented
    """
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    parallel_request_handler = MaxParallelRequestsHandler(
        internal_usage_cache=local_cache
    )

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
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
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
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
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 1
    )

    # normal call
    response = await router.acompletion(
        model="azure-model",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        metadata={"user_api_key": _api_key},
        mock_response="hello",
    )
    await asyncio.sleep(1)  # success is done in a separate thread
    print(f"response: {response}")

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 0
    )


@pytest.mark.asyncio
async def test_normal_router_tpm_limit():
    import logging

    from litellm._logging import verbose_proxy_logger

    verbose_proxy_logger.setLevel(level=logging.DEBUG)
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
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key, max_parallel_requests=10, tpm_limit=10
    )
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"
    print("Test: Checking current_requests for precise_minute=", precise_minute)

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 1
    )

    # normal call
    response = await router.acompletion(
        model="azure-model",
        messages=[{"role": "user", "content": "Write me a paragraph on the moon"}],
        metadata={"user_api_key": _api_key},
        mock_response="hello",
    )
    await asyncio.sleep(1)  # success is done in a separate thread
    print(f"response: {response}")

    try:
        assert (
            parallel_request_handler.internal_usage_cache.get_cache(
                key=request_count_api_key
            )["current_tpm"]
            > 0
        )

    except Exception as e:
        print("Exception on test_normal_router_tpm_limit", e)
        assert e.status_code == 429


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
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 1
    )

    # streaming call
    response = await router.acompletion(
        model="azure-model",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        stream=True,
        metadata={"user_api_key": _api_key},
        mock_response="hello",
    )
    async for chunk in response:
        continue
    await asyncio.sleep(1)  # success is done in a separate thread
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 0
    )


@pytest.mark.asyncio
async def test_streaming_router_tpm_limit():
    litellm.set_verbose = True
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
    _api_key = hash_token("sk-12345")
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key, max_parallel_requests=10, tpm_limit=10
    )
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 1
    )

    # normal call
    response = await router.acompletion(
        model="azure-model",
        messages=[{"role": "user", "content": "Write me a paragraph on the moon"}],
        stream=True,
        metadata={"user_api_key": _api_key},
        mock_response="hello",
    )
    async for chunk in response:
        continue
    await asyncio.sleep(5)  # success is done in a separate thread

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_tpm"]
        > 0
    )


@pytest.mark.asyncio
@pytest.mark.flaky(retries=3, delay=1)
async def test_bad_router_call():
    litellm.set_verbose = True
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
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, max_parallel_requests=1)
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(  # type: ignore
            key=request_count_api_key
        )["current_requests"]
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
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(  # type: ignore
            key=request_count_api_key
        )["current_requests"]
        == 0
    )


@pytest.mark.asyncio
async def test_bad_router_tpm_limit():
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
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key, max_parallel_requests=10, tpm_limit=10
    )
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 1
    )

    # bad call
    try:
        response = await router.acompletion(
            model="azure-model",
            messages=[{"role": "user2", "content": "Write me a paragraph on the moon"}],
            stream=True,
            metadata={"user_api_key": _api_key},
        )
    except:
        pass
    await asyncio.sleep(1)  # success is done in a separate thread

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_tpm"]
        == 0
    )


@pytest.mark.asyncio
async def test_bad_router_tpm_limit_per_model():
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
    _api_key = hash_token(_api_key)
    model = "azure-model"

    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        max_parallel_requests=10,
        tpm_limit=10,
        metadata={
            "model_rpm_limit": {model: 5},
            "model_tpm_limit": {model: 5},
        },
    )
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": model},
        call_type="",
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{model}::{precise_minute}::request_count"

    print(
        "internal usage cache: ",
        parallel_request_handler.internal_usage_cache.in_memory_cache.cache_dict,
    )

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 1
    )

    # bad call
    try:
        response = await router.acompletion(
            model=model,
            messages=[{"role": "user2", "content": "Write me a paragraph on the moon"}],
            stream=True,
            metadata={
                "user_api_key": _api_key,
                "user_api_key_metadata": {
                    "model_rpm_limit": {model: 5},
                    "model_tpm_limit": {model: 5},
                },
            },
        )
    except:
        pass
    await asyncio.sleep(1)  # success is done in a separate thread

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_tpm"]
        == 0
    )


@pytest.mark.asyncio
async def test_pre_call_hook_rpm_limits_per_model():
    """
    Test if error raised on hitting rpm limits for a given model
    """
    import logging

    from litellm._logging import (
        verbose_logger,
        verbose_proxy_logger,
        verbose_router_logger,
    )

    verbose_logger.setLevel(logging.DEBUG)
    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_router_logger.setLevel(logging.DEBUG)

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        max_parallel_requests=100,
        tpm_limit=900000,
        rpm_limit=100000,
        metadata={
            "model_rpm_limit": {"azure-model": 1},
        },
    )
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    model = "azure-model"

    kwargs = {
        "model": model,
        "litellm_params": {
            "metadata": {
                "user_api_key": _api_key,
                "model_group": model,
                "user_api_key_metadata": {"model_rpm_limit": {"azure-model": 1}},
            },
        },
    }

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs,
        response_obj="",
        start_time="",
        end_time="",
    )

    ## Expected cache val: {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}

    try:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": model},
            call_type="",
        )

        pytest.fail(f"Expected call to fail")
    except Exception as e:
        assert e.status_code == 429
        print("got error=", e)
        assert (
            "limit reached Hit RPM limit for model: azure-model on api_key: c11e7177eb60c80cf983ddf8ca98f2dc1272d4c612204ce9bedd2460b18939cc"
            in str(e)
        )


@pytest.mark.asyncio
async def test_pre_call_hook_tpm_limits_per_model():
    """
    Test if error raised on hitting tpm limits for a given model
    """
    import logging

    from litellm._logging import (
        verbose_logger,
        verbose_proxy_logger,
        verbose_router_logger,
    )

    verbose_logger.setLevel(logging.DEBUG)
    verbose_proxy_logger.setLevel(logging.DEBUG)
    verbose_router_logger.setLevel(logging.DEBUG)

    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(
        api_key=_api_key,
        max_parallel_requests=100,
        tpm_limit=900000,
        rpm_limit=100000,
        metadata={
            "model_tpm_limit": {"azure-model": 1},
            "model_rpm_limit": {"azure-model": 100},
        },
    )
    local_cache = DualCache()
    pl = ProxyLogging(user_api_key_cache=local_cache)
    pl._init_litellm_callbacks()
    print(f"litellm callbacks: {litellm.callbacks}")
    parallel_request_handler = pl.max_parallel_request_limiter
    model = "azure-model"

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": model},
        call_type="",
    )

    kwargs = {
        "model": model,
        "litellm_params": {
            "metadata": {
                "user_api_key": _api_key,
                "model_group": model,
                "user_api_key_metadata": {
                    "model_tpm_limit": {"azure-model": 1},
                    "model_rpm_limit": {"azure-model": 100},
                },
            }
        },
    }

    await parallel_request_handler.async_log_success_event(
        kwargs=kwargs,
        response_obj=litellm.ModelResponse(usage=litellm.Usage(total_tokens=11)),
        start_time="",
        end_time="",
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{model}::{precise_minute}::request_count"

    print(
        "internal usage cache: ",
        parallel_request_handler.internal_usage_cache.in_memory_cache.cache_dict,
    )

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_tpm"]
        == 11
    )

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_rpm"]
        == 1
    )

    ## Expected cache val: {"current_requests": 0, "current_tpm": 11, "current_rpm": "1"}

    try:
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": model},
            call_type="",
        )

        pytest.fail(f"Expected call to fail")
    except Exception as e:
        assert e.status_code == 429
        print("got error=", e)
        assert (
            "request limit reached Hit TPM limit for model: azure-model on api_key"
            in str(e)
        )
