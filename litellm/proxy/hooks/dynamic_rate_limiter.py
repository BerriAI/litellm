# What is this?
## Allocates dynamic tpm/rpm quota for a project based on current traffic
## Tracks num active projects per minute

import asyncio
import sys
import traceback
from datetime import datetime
from typing import List, Literal, Optional, Tuple, Union

from fastapi import HTTPException

import litellm
from litellm import ModelResponse, Router
from litellm._logging import verbose_proxy_logger
from litellm.caching import DualCache
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import ModelGroupInfo
from litellm.utils import get_utc_datetime


class DynamicRateLimiterCache:
    """
    Thin wrapper on DualCache for this file.

    Track number of active projects calling a model.
    """

    def __init__(self, cache: DualCache) -> None:
        self.cache = cache
        self.ttl = 60  # 1 min ttl

    async def async_get_cache(self, model: str) -> Optional[int]:
        dt = get_utc_datetime()
        current_minute = dt.strftime("%H-%M")
        key_name = "{}:{}".format(current_minute, model)
        _response = await self.cache.async_get_cache(key=key_name)
        response: Optional[int] = None
        if _response is not None:
            response = len(_response)
        return response

    async def async_set_cache_sadd(self, model: str, value: List):
        """
        Add value to set.

        Parameters:
        - model: str, the name of the model group
        - value: str, the team id

        Returns:
        - None

        Raises:
        - Exception, if unable to connect to cache client (if redis caching enabled)
        """
        try:
            dt = get_utc_datetime()
            current_minute = dt.strftime("%H-%M")

            key_name = "{}:{}".format(current_minute, model)
            await self.cache.async_set_cache_sadd(
                key=key_name, value=value, ttl=self.ttl
            )
        except Exception as e:
            verbose_proxy_logger.error(
                "litellm.proxy.hooks.dynamic_rate_limiter.py::async_set_cache_sadd(): Exception occured - {}\n{}".format(
                    str(e), traceback.format_exc()
                )
            )
            raise e


class _PROXY_DynamicRateLimitHandler(CustomLogger):

    # Class variables or attributes
    def __init__(self, internal_usage_cache: DualCache):
        self.internal_usage_cache = DynamicRateLimiterCache(cache=internal_usage_cache)

    def update_variables(self, llm_router: Router):
        self.llm_router = llm_router

    async def check_available_tpm(
        self, model: str
    ) -> Tuple[Optional[int], Optional[int], Optional[int]]:
        """
        For a given model, get its available tpm

        Returns
        - Tuple[available_tpm, model_tpm, active_projects]
            - available_tpm: int or null - always 0 or positive.
            - remaining_model_tpm: int or null. If available tpm is int, then this will be too.
            - active_projects: int or null
        """
        active_projects = await self.internal_usage_cache.async_get_cache(model=model)
        current_model_tpm: Optional[int] = await self.llm_router.get_model_group_usage(
            model_group=model
        )
        model_group_info: Optional[ModelGroupInfo] = (
            self.llm_router.get_model_group_info(model_group=model)
        )
        total_model_tpm: Optional[int] = None
        if model_group_info is not None and model_group_info.tpm is not None:
            total_model_tpm = model_group_info.tpm

        remaining_model_tpm: Optional[int] = None
        if total_model_tpm is not None and current_model_tpm is not None:
            remaining_model_tpm = total_model_tpm - current_model_tpm
        elif total_model_tpm is not None:
            remaining_model_tpm = total_model_tpm

        available_tpm: Optional[int] = None

        if remaining_model_tpm is not None:
            if active_projects is not None:
                available_tpm = int(remaining_model_tpm / active_projects)
            else:
                available_tpm = remaining_model_tpm

        if available_tpm is not None and available_tpm < 0:
            available_tpm = 0
        return available_tpm, remaining_model_tpm, active_projects

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: Literal[
            "completion",
            "text_completion",
            "embeddings",
            "image_generation",
            "moderation",
            "audio_transcription",
        ],
    ) -> Optional[
        Union[Exception, str, dict]
    ]:  # raise exception if invalid, return a str for the user to receive - if rejected, or return a modified dictionary for passing into litellm
        """
        - For a model group
        - Check if tpm available
        - Raise RateLimitError if no tpm available
        """
        if "model" in data:
            available_tpm, model_tpm, active_projects = await self.check_available_tpm(
                model=data["model"]
            )
            if available_tpm is not None and available_tpm == 0:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Key={} over available TPM={}. Model TPM={}, Active keys={}".format(
                            user_api_key_dict.api_key,
                            available_tpm,
                            model_tpm,
                            active_projects,
                        )
                    },
                )
            elif available_tpm is not None:
                ## UPDATE CACHE WITH ACTIVE PROJECT
                asyncio.create_task(
                    self.internal_usage_cache.async_set_cache_sadd(  # this is a set
                        model=data["model"],  # type: ignore
                        value=[user_api_key_dict.token or "default_key"],
                    )
                )
        return None

    async def async_post_call_success_hook(
        self, user_api_key_dict: UserAPIKeyAuth, response
    ):
        try:
            if isinstance(response, ModelResponse):
                model_info = self.llm_router.get_model_info(
                    id=response._hidden_params["model_id"]
                )
                assert (
                    model_info is not None
                ), "Model info for model with id={} is None".format(
                    response._hidden_params["model_id"]
                )
                available_tpm, remaining_model_tpm, active_projects = (
                    await self.check_available_tpm(model=model_info["model_name"])
                )
                response._hidden_params["additional_headers"] = {
                    "x-litellm-model_group": model_info["model_name"],
                    "x-ratelimit-remaining-litellm-project-tokens": available_tpm,
                    "x-ratelimit-remaining-model-tokens": remaining_model_tpm,
                    "x-ratelimit-current-active-projects": active_projects,
                }

                return response
            return await super().async_post_call_success_hook(
                user_api_key_dict, response
            )
        except Exception as e:
            verbose_proxy_logger.error(
                "litellm.proxy.hooks.dynamic_rate_limiter.py::async_post_call_success_hook(): Exception occured - {}\n{}".format(
                    str(e), traceback.format_exc()
                )
            )
            return response
