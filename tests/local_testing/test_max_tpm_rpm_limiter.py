### REPLACED BY 'test_parallel_request_limiter.py' ###
# What is this?
## Unit tests for the max tpm / rpm limiter hook for proxy

# import sys, os, asyncio, time, random
# from datetime import datetime
# import traceback
# from dotenv import load_dotenv
# from typing import Optional

# load_dotenv()
# import os

# sys.path.insert(
#     0, os.path.abspath("../..")
# )  # Adds the parent directory to the system path
# import pytest
# import litellm
# from litellm import Router
# from litellm.proxy.utils import ProxyLogging, hash_token
# from litellm.proxy._types import UserAPIKeyAuth
# from litellm.caching import DualCache, RedisCache
# from litellm.proxy.hooks.tpm_rpm_limiter import _PROXY_MaxTPMRPMLimiter
# from datetime import datetime


# @pytest.mark.asyncio
# async def test_pre_call_hook_rpm_limits():
#     """
#     Test if error raised on hitting rpm limits
#     """
#     litellm.set_verbose = True
#     _api_key = hash_token("sk-12345")
#     user_api_key_dict = UserAPIKeyAuth(api_key=_api_key, tpm_limit=9, rpm_limit=1)
#     local_cache = DualCache()
#     # redis_usage_cache = RedisCache()

#     local_cache.set_cache(
#         key=_api_key, value={"api_key": _api_key, "tpm_limit": 9, "rpm_limit": 1}
#     )

#     tpm_rpm_limiter = _PROXY_MaxTPMRPMLimiter(internal_cache=DualCache())

#     await tpm_rpm_limiter.async_pre_call_hook(
#         user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
#     )

#     kwargs = {"litellm_params": {"metadata": {"user_api_key": _api_key}}}

#     await tpm_rpm_limiter.async_log_success_event(
#         kwargs=kwargs,
#         response_obj="",
#         start_time="",
#         end_time="",
#     )

#     ## Expected cache val: {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}

#     try:
#         await tpm_rpm_limiter.async_pre_call_hook(
#             user_api_key_dict=user_api_key_dict,
#             cache=local_cache,
#             data={},
#             call_type="",
#         )

#         pytest.fail(f"Expected call to fail")
#     except Exception as e:
#         assert e.status_code == 429


# @pytest.mark.asyncio
# async def test_pre_call_hook_team_rpm_limits(
#     _redis_usage_cache: Optional[RedisCache] = None,
# ):
#     """
#     Test if error raised on hitting team rpm limits
#     """
#     litellm.set_verbose = True
#     _api_key = "sk-12345"
#     _team_id = "unique-team-id"
#     _user_api_key_dict = {
#         "api_key": _api_key,
#         "max_parallel_requests": 1,
#         "tpm_limit": 9,
#         "rpm_limit": 10,
#         "team_rpm_limit": 1,
#         "team_id": _team_id,
#     }
#     user_api_key_dict = UserAPIKeyAuth(**_user_api_key_dict)  # type: ignore
#     _api_key = hash_token(_api_key)
#     local_cache = DualCache()
#     local_cache.set_cache(key=_api_key, value=_user_api_key_dict)
#     internal_cache = DualCache(redis_cache=_redis_usage_cache)
#     tpm_rpm_limiter = _PROXY_MaxTPMRPMLimiter(internal_cache=internal_cache)
#     await tpm_rpm_limiter.async_pre_call_hook(
#         user_api_key_dict=user_api_key_dict, cache=local_cache, data={}, call_type=""
#     )

#     kwargs = {
#         "litellm_params": {
#             "metadata": {"user_api_key": _api_key, "user_api_key_team_id": _team_id}
#         }
#     }

#     await tpm_rpm_limiter.async_log_success_event(
#         kwargs=kwargs,
#         response_obj="",
#         start_time="",
#         end_time="",
#     )

#     print(f"local_cache: {local_cache}")

#     ## Expected cache val: {"current_requests": 0, "current_tpm": 0, "current_rpm": 1}

#     try:
#         await tpm_rpm_limiter.async_pre_call_hook(
#             user_api_key_dict=user_api_key_dict,
#             cache=local_cache,
#             data={},
#             call_type="",
#         )

#         pytest.fail(f"Expected call to fail")
#     except Exception as e:
#         assert e.status_code == 429  # type: ignore


# @pytest.mark.asyncio
# async def test_namespace():
#     """
#     - test if default namespace set via `proxyconfig._init_cache`
#     - respected for tpm/rpm caching
#     """
#     from litellm.proxy.proxy_server import ProxyConfig

#     redis_usage_cache: Optional[RedisCache] = None
#     cache_params = {"type": "redis", "namespace": "litellm_default"}

#     ## INIT CACHE ##
#     proxy_config = ProxyConfig()
#     setattr(litellm.proxy.proxy_server, "proxy_config", proxy_config)

#     proxy_config._init_cache(cache_params=cache_params)

#     redis_cache: Optional[RedisCache] = getattr(
#         litellm.proxy.proxy_server, "redis_usage_cache"
#     )

#     ## CHECK IF NAMESPACE SET ##
#     assert redis_cache.namespace == "litellm_default"

#     ## CHECK IF TPM/RPM RATE LIMITING WORKS ##
#     await test_pre_call_hook_team_rpm_limits(_redis_usage_cache=redis_cache)
#     current_date = datetime.now().strftime("%Y-%m-%d")
#     current_hour = datetime.now().strftime("%H")
#     current_minute = datetime.now().strftime("%M")
#     precise_minute = f"{current_date}-{current_hour}-{current_minute}"

#     cache_key = "litellm_default:usage:{}".format(precise_minute)
#     value = await redis_cache.async_get_cache(key=cache_key)
#     assert value is not None
