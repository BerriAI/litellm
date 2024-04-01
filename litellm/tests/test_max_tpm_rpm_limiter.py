# What is this?
## Unit tests for the max tpm / rpm limiter hook for proxy

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
from litellm.caching import DualCache, RedisCache
from litellm.proxy.hooks.tpm_rpm_limiter import _PROXY_MaxTPMRPMLimiter
from datetime import datetime


@pytest.mark.asyncio
async def test_pre_call_hook_rpm_limits():
    """
    Test if error raised on hitting rpm limits
    """
    litellm.set_verbose = True
    _api_key = "sk-12345"
    user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, tpm_limit=9, rpm_limit=1)
    local_cache = DualCache()
    # redis_usage_cache = RedisCache()

    local_cache.set_cache(
        key=_api_key, value={"api_key": _api_key, "tpm_limit": 9, "rpm_limit": 1}
    )

    tpm_rpm_limiter = _PROXY_MaxTPMRPMLimiter(redis_usage_cache=None)

    await tpm_rpm_limiter.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    kwargs = {"litellm_params": {"metadata": {"user_api_key": _api_key}}}

    await tpm_rpm_limiter.async_log_success_event(
        kwargs=kwargs,
        response_obj="",
        start_time="",
        end_time="",
    )

    ## Expected cache val: {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}

    try:
        await tpm_rpm_limiter.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={},
            call_type="",
        )

        pytest.fail(f"Expected call to fail")
    except Exception as e:
        assert e.status_code == 429


@pytest.mark.asyncio
async def test_pre_call_hook_team_rpm_limits():
    """
    Test if error raised on hitting team rpm limits
    """
    litellm.set_verbose = True
    _api_key = "sk-12345"
    _team_id = "unique-team-id"
    _user_api_key_dict = {
        "api_key": _api_key,
        "max_parallel_requests": 1,
        "tpm_limit": 9,
        "rpm_limit": 10,
        "team_rpm_limit": 1,
        "team_id": _team_id,
    }
    user_api_key_dict = UserAPIKeyAuth(**_user_api_key_dict)
    local_cache = DualCache()
    local_cache.set_cache(key=_api_key, value=_user_api_key_dict)
    tpm_rpm_limiter = _PROXY_MaxTPMRPMLimiter(redis_usage_cache=None)

    await tpm_rpm_limiter.async_pre_call_hook(
        user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
    )

    kwargs = {
        "litellm_params": {
            "metadata": {"user_api_key": _api_key, "user_api_key_team_id": _team_id}
        }
    }

    await tpm_rpm_limiter.async_log_success_event(
        kwargs=kwargs,
        response_obj="",
        start_time="",
        end_time="",
    )

    print(f"local_cache: {local_cache}")

    ## Expected cache val: {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}

    try:
        await tpm_rpm_limiter.async_pre_call_hook(
            user_api_key_dict=user_api_key_dict,
            cache=local_cache,
            data={},
            call_type="",
        )

        pytest.fail(f"Expected call to fail")
    except Exception as e:
        assert e.status_code == 429
