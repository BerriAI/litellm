"""
Unit Tests for the max parallel request limiter v2 for the proxy
"""
import asyncio
import os
import sys
from datetime import datetime

import pytest
from fastapi import HTTPException

import litellm
from litellm import Router
from litellm.caching.caching import DualCache
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.parallel_request_limiter_v2 import (
    _PROXY_MaxParallelRequestsHandler_v2 as _PROXY_MaxParallelRequestsHandler,
)
from litellm.proxy.utils import InternalUsageCache, ProxyLogging, hash_token


@pytest.mark.flaky(reruns=3)
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
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    current_slot = (
        current_time.second // 15
    )  # This gives us 0-3 for the current 15s slot
    slot_key = f"{current_time.strftime('%Y-%m-%d')}-{current_hour:02d}-{current_minute:02d}-{current_slot}"
    print(f"slot_key: {slot_key}")
    request_count_api_key = parallel_request_handler._get_current_usage_key(
        user_api_key_dict=user_api_key_dict,
        precise_minute=slot_key,
        model=None,
        rate_limit_type="key",
        group="request_count",
    )
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

    print(f"local_cache in normal call: {local_cache}")
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
        == 0
    )


@pytest.mark.parametrize(
    "rate_limit_object",
    [
        "key",
        "model_per_key",
        "user",
        "customer",
        "team",
    ],
)
@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_normal_router_call_tpm(monkeypatch, rate_limit_object):
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
    if rate_limit_object == "key":
        user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, tpm_limit=10)
    elif rate_limit_object == "user":
        user_api_key_dict = UserAPIKeyAuth(user_id="12345", user_tpm_limit=10)
    elif rate_limit_object == "team":
        user_api_key_dict = UserAPIKeyAuth(team_id="12345", team_tpm_limit=10)
    elif rate_limit_object == "customer":
        user_api_key_dict = UserAPIKeyAuth(end_user_id="12345", end_user_tpm_limit=10)
    elif rate_limit_object == "model_per_key":
        user_api_key_dict = UserAPIKeyAuth(
            api_key=_api_key,
            metadata={"model_tpm_limit": {"azure-model": 10}},
        )
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": "azure-model"},
        call_type="",
    )

    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    current_slot = (
        current_time.second // 15
    )  # This gives us 0-3 for the current 15s slot
    slot_key = f"{current_time.strftime('%Y-%m-%d')}-{current_hour:02d}-{current_minute:02d}-{current_slot}"
    print(f"slot_key: {slot_key}")
    request_count_api_key = parallel_request_handler._get_current_usage_key(
        user_api_key_dict=user_api_key_dict,
        precise_minute=slot_key,
        model="azure-model",
        rate_limit_type=rate_limit_object,
        group="tpm",
    )
    print(f"request_count_api_key: {request_count_api_key}")
    await asyncio.sleep(1)
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
        == 0
    )

    # normal call
    response = await router.acompletion(
        model="azure-model",
        messages=[{"role": "user", "content": "Hey, how's it going?"}],
        metadata={
            "user_api_key": _api_key,
            "user_api_key_user_id": user_api_key_dict.user_id,
            "user_api_key_team_id": user_api_key_dict.team_id,
            "user_api_key_end_user_id": user_api_key_dict.end_user_id,
        },
        mock_response="hello",
    )
    await asyncio.sleep(1)  # success is done in a separate thread

    print(f"request_count_api_key: {request_count_api_key}")

    next_slot_key = f"{current_time.strftime('%Y-%m-%d')}-{current_hour:02d}-{current_minute:02d}-{current_slot + 1 if current_slot < 3 else 0}"
    request_count_api_key_next_slot = parallel_request_handler._get_current_usage_key(
        user_api_key_dict=user_api_key_dict,
        precise_minute=next_slot_key,
        model="azure-model",
        rate_limit_type=rate_limit_object,
        group="tpm",
    )

    ## check if current slot matches response.usage.total_tokens else next slot
    current_slot_get_cache = parallel_request_handler.internal_usage_cache.get_cache(
        key=request_count_api_key
    )
    next_slot_get_cache = parallel_request_handler.internal_usage_cache.get_cache(
        key=request_count_api_key_next_slot
    )

    assert (
        current_slot_get_cache == response.usage.total_tokens
        or next_slot_get_cache == response.usage.total_tokens
    )


@pytest.mark.parametrize(
    "rate_limit_object",
    [
        "key",
        "model_per_key",
        "user",
        "customer",
        "team",
    ],
)
@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_normal_router_call_rpm(monkeypatch, rate_limit_object):
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
    if rate_limit_object == "key":
        user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=1)
    elif rate_limit_object == "user":
        user_api_key_dict = UserAPIKeyAuth(user_id="12345", user_rpm_limit=1)
    elif rate_limit_object == "team":
        user_api_key_dict = UserAPIKeyAuth(team_id="12345", team_rpm_limit=1)
    elif rate_limit_object == "customer":
        user_api_key_dict = UserAPIKeyAuth(end_user_id="12345", end_user_rpm_limit=1)
    elif rate_limit_object == "model_per_key":
        user_api_key_dict = UserAPIKeyAuth(
            api_key=_api_key,
            metadata={"model_rpm_limit": {"azure-model": 1}},
        )
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict,
        cache=local_cache,
        data={"model": "azure-model"},
        call_type="",
    )

    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    current_slot = (
        current_time.second // 15
    )  # This gives us 0-3 for the current 15s slot
    slot_key = f"{current_time.strftime('%Y-%m-%d')}-{current_hour:02d}-{current_minute:02d}-{current_slot}"
    request_count_api_key = parallel_request_handler._get_current_usage_key(
        user_api_key_dict=user_api_key_dict,
        precise_minute=slot_key,
        model="azure-model",
        rate_limit_type=rate_limit_object,
        group="rpm",
    )
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
        metadata={
            "user_api_key": _api_key,
            "user_api_key_user_id": user_api_key_dict.user_id,
            "user_api_key_team_id": user_api_key_dict.team_id,
            "user_api_key_end_user_id": user_api_key_dict.end_user_id,
        },
        mock_response="hello",
    )
    await asyncio.sleep(1)  # success is done in a separate thread

    print(f"request_count_api_key: {request_count_api_key}")

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
        == 1
    )

    with pytest.raises(HTTPException):
        await parallel_request_handler.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={"model": "azure-model"},
            call_type="",
        )


@pytest.mark.flaky(reruns=3)
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

    print(f"litellm callbacks pre-set: {litellm.callbacks}")
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    current_slot = (
        current_time.second // 15
    )  # This gives us 0-3 for the current 15s slot
    slot_key = f"{current_time.strftime('%Y-%m-%d')}-{current_hour:02d}-{current_minute:02d}-{current_slot}"

    request_count_api_key = parallel_request_handler._get_current_usage_key(
        user_api_key_dict=user_api_key_dict,
        precise_minute=slot_key,
        model=None,
        rate_limit_type="key",
        group="request_count",
    )
    await asyncio.sleep(1)
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
        == 1
    )

    # streaming call
    print(f"litellm callbacks: {litellm.callbacks}")
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
    print(f"local_cache in streaming call: {local_cache}")
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
        == 0
    )


@pytest.mark.parametrize(
    "rate_limit_object",
    [
        "key",
        # "model_per_key",
        "user",
        # "customer",
        "team",
    ],
)
@pytest.mark.flaky(reruns=3)
@pytest.mark.asyncio
async def test_bad_router_call_v2(monkeypatch, rate_limit_object):
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
    if rate_limit_object == "key":
        user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=1)
    elif rate_limit_object == "user":
        user_api_key_dict = UserAPIKeyAuth(user_id="12345", user_rpm_limit=1)
    elif rate_limit_object == "team":
        user_api_key_dict = UserAPIKeyAuth(team_id="12345", team_rpm_limit=1)
    local_cache = DualCache()

    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )
    monkeypatch.setattr(litellm, "callbacks", [parallel_request_handler])

    await parallel_request_handler.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    current_time = datetime.now()
    current_hour = current_time.hour
    current_minute = current_time.minute
    current_slot = (
        current_time.second // 15
    )  # This gives us 0-3 for the current 15s slot
    slot_key = f"{current_time.strftime('%Y-%m-%d')}-{current_hour:02d}-{current_minute:02d}-{current_slot}"
    request_count_api_key = parallel_request_handler._get_current_usage_key(
        user_api_key_dict=user_api_key_dict,
        precise_minute=slot_key,
        model=None,
        rate_limit_type=rate_limit_object,
        group="rpm",
    )
    await asyncio.sleep(1)
    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
        == 1
    )

    # bad streaming call
    await parallel_request_handler.async_post_call_failure_hook(
        request_data={},
        original_exception=Exception("test"),
        user_api_key_dict=user_api_key_dict,
    )

    assert (
        parallel_request_handler.internal_usage_cache.get_cache(
            key=request_count_api_key
        )
        == 1
    )


@pytest.mark.asyncio
async def test_check_key_in_limits_v2_sliding_window():
    """
    Test the check_key_in_limits_v2 function with sliding window logic
    """
    print("Starting test")
    _api_key = "sk-12345"
    _api_key = hash_token(_api_key)
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, rpm_limit=2)
    local_cache = DualCache()
    parallel_request_handler = _PROXY_MaxParallelRequestsHandler(
        internal_usage_cache=InternalUsageCache(local_cache)
    )

    print("Created handler")
    # Get current time and calculate slots
    current_time = datetime.now()
    current_slot = (current_time.minute * 60 + current_time.second) // 15
    current_slot_key = (
        f"{current_time.strftime('%Y-%m-%d')}-{current_time.hour:02d}-{current_slot}"
    )
    print(f"Current slot key: {current_slot_key}")

    print("Making first request")
    # Test 1: First request should succeed
    await parallel_request_handler.check_key_in_limits_v2(
        user_api_key_dict=user_api_key_dict,
        data={},
        max_parallel_requests=None,
        precise_minute=current_slot_key,
        tpm_limit=None,
        rpm_limit=3,
        rate_limit_type="key",
    )
    print("First request completed")

    print("Making second request")
    # Test 2: Second request should succeed
    await parallel_request_handler.check_key_in_limits_v2(
        user_api_key_dict=user_api_key_dict,
        data={},
        max_parallel_requests=None,
        precise_minute=current_slot_key,
        tpm_limit=None,
        rpm_limit=3,
        rate_limit_type="key",
    )
    print("Second request completed")

    print("Verifying cache")
    # Make third request - should fail
    with pytest.raises(HTTPException):
        await parallel_request_handler.check_key_in_limits_v2(
            user_api_key_dict=user_api_key_dict,
            data={},
            max_parallel_requests=None,
            precise_minute=current_slot_key,
            tpm_limit=None,
            rpm_limit=2,
            rate_limit_type="key",
        )
