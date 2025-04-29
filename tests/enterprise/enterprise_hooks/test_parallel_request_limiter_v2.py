"""
Unit Tests for the max parallel request limiter v2 for the proxy
"""
import asyncio
import os
import sys
from datetime import datetime

import pytest

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.utils import InternalUsageCache, ProxyLogging, hash_token
from enterprise.enterprise_hooks.parallel_request_limiter_v2 import _PROXY_MaxParallelRequestsHandler

@pytest.mark.asyncio
async def test_normal_router_call_v2(monkeypatch):
    """
    Test normal router call with parallel request limiter v2
    """
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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(local_cache))
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"
    await asyncio.sleep(1)
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
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

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
        == 0
    )

@pytest.mark.asyncio
async def test_streaming_router_call_v2(monkeypatch):
    """
    Test streaming router call with parallel request limiter v2
    """

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
    
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(local_cache))
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"
    await asyncio.sleep(1)
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
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
    await asyncio.sleep(3)  # success is done in a separate thread
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
        == 0
    )

@pytest.mark.asyncio
async def test_bad_router_call_v2(monkeypatch):
    """
    Test bad router call with parallel request limiter v2
    """
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
    
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(internal_usage_cache=InternalUsageCache(local_cache))
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_date = datetime.now().strftime("%Y-%m-%d")
    current_hour = datetime.now().strftime("%H")
    current_minute = datetime.now().strftime("%M")
    precise_minute = f"{current_date}-{current_hour}-{current_minute}"
    request_count_api_key = f"{_api_key}::{precise_minute}::request_count"
    await asyncio.sleep(1)
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
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
    except Exception:
        pass
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )["current_requests"]
        == 0
    )
