# What is this?
## Allocates dynamic tpm/rpm quota for a project based on current traffic
## Tracks num active projects per minute

import asyncio
import os
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
            verbose_proxy_logger.exception(
                "litellm.proxy.hooks.dynamic_rate_limiter.py::async_set_cache_sadd(): Exception occured - {}".format(
                    str(e)
                )
            )
            raise e


class _PROXY_DynamicRateLimitHandler(CustomLogger):

    # Class variables or attributes
    def __init__(self, internal_usage_cache: DualCache):
        self.internal_usage_cache = DynamicRateLimiterCache(cache=internal_usage_cache)

    def update_variables(self, llm_router: Router):
        self.llm_router = llm_router

    async def check_available_usage(
        self, model: str, priority: Optional[str] = None
    ) -> Tuple[
        Optional[int], Optional[int], Optional[int], Optional[int], Optional[int]
    ]:
        """
        For a given model, get its available tpm

        Params:
        - model: str, the name of the model in the router model_list
        - priority: Optional[str], the priority for the request.

        Returns
        - Tuple[available_tpm, available_tpm, model_tpm, model_rpm, active_projects]
            - available_tpm: int or null - always 0 or positive.
            - available_tpm: int or null - always 0 or positive.
            - remaining_model_tpm: int or null. If available tpm is int, then this will be too.
            - remaining_model_rpm: int or null. If available rpm is int, then this will be too.
            - active_projects: int or null
        """
        try:
            weight: float = 1
            if (
                litellm.priority_reservation is None
                or priority not in litellm.priority_reservation
            ):
                verbose_proxy_logger.error(
                    "Priority Reservation not set. priority={}, but litellm.priority_reservation is {}.".format(
                        priority, litellm.priority_reservation
                    )
                )
            elif priority is not None and litellm.priority_reservation is not None:
                if os.getenv("LITELLM_LICENSE", None) is None:
                    verbose_proxy_logger.error(
                        "PREMIUM FEATURE: Reserving tpm/rpm by priority is a premium feature. Please add a 'LITELLM_LICENSE' to your .env to enable this.\nGet a license: https://docs.litellm.ai/docs/proxy/enterprise."
                    )
                else:
                    weight = litellm.priority_reservation[priority]

            active_projects = await self.internal_usage_cache.async_get_cache(
                model=model
            )
            current_model_tpm, current_model_rpm = (
                await self.llm_router.get_model_group_usage(model_group=model)
            )
            model_group_info: Optional[ModelGroupInfo] = (
                self.llm_router.get_model_group_info(model_group=model)
            )
            total_model_tpm: Optional[int] = None
            total_model_rpm: Optional[int] = None
            if model_group_info is not None:
                if model_group_info.tpm is not None:
                    total_model_tpm = model_group_info.tpm
                if model_group_info.rpm is not None:
                    total_model_rpm = model_group_info.rpm

            remaining_model_tpm: Optional[int] = None
            if total_model_tpm is not None and current_model_tpm is not None:
                remaining_model_tpm = total_model_tpm - current_model_tpm
            elif total_model_tpm is not None:
                remaining_model_tpm = total_model_tpm

            remaining_model_rpm: Optional[int] = None
            if total_model_rpm is not None and current_model_rpm is not None:
                remaining_model_rpm = total_model_rpm - current_model_rpm
            elif total_model_rpm is not None:
                remaining_model_rpm = total_model_rpm

            available_tpm: Optional[int] = None

            if remaining_model_tpm is not None:
                if active_projects is not None:
                    available_tpm = int(remaining_model_tpm * weight / active_projects)
                else:
                    available_tpm = int(remaining_model_tpm * weight)

            if available_tpm is not None and available_tpm < 0:
                available_tpm = 0

            available_rpm: Optional[int] = None

            if remaining_model_rpm is not None:
                if active_projects is not None:
                    available_rpm = int(remaining_model_rpm * weight / active_projects)
                else:
                    available_rpm = int(remaining_model_rpm * weight)

            if available_rpm is not None and available_rpm < 0:
                available_rpm = 0
            return (
                available_tpm,
                available_rpm,
                remaining_model_tpm,
                remaining_model_rpm,
                active_projects,
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.hooks.dynamic_rate_limiter.py::check_available_usage: Exception occurred - {}".format(
                    str(e)
                )
            )
            return None, None, None, None, None

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
            "pass_through_endpoint",
            "rerank",
        ],
    ) -> Optional[
        Union[Exception, str, dict]
    ]:  # raise exception if invalid, return a str for the user to receive - if rejected, or return a modified dictionary for passing into litellm
        """
        - For a model group
        - Check if tpm/rpm available
        - Raise RateLimitError if no tpm/rpm available
        """
        if "model" in data:
            key_priority: Optional[str] = user_api_key_dict.metadata.get(
                "priority", None
            )
            available_tpm, available_rpm, model_tpm, model_rpm, active_projects = (
                await self.check_available_usage(
                    model=data["model"], priority=key_priority
                )
            )
            ### CHECK TPM ###
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
            ### CHECK RPM ###
            elif available_rpm is not None and available_rpm == 0:
                raise HTTPException(
                    status_code=429,
                    detail={
                        "error": "Key={} over available RPM={}. Model RPM={}, Active keys={}".format(
                            user_api_key_dict.api_key,
                            available_rpm,
                            model_rpm,
                            active_projects,
                        )
                    },
                )
            elif available_rpm is not None or available_tpm is not None:
                ## UPDATE CACHE WITH ACTIVE PROJECT
                asyncio.create_task(
                    self.internal_usage_cache.async_set_cache_sadd(  # this is a set
                        model=data["model"],  # type: ignore
                        value=[user_api_key_dict.token or "default_key"],
                    )
                )
        return None

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
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
                key_priority: Optional[str] = user_api_key_dict.metadata.get(
                    "priority", None
                )
                available_tpm, available_rpm, model_tpm, model_rpm, active_projects = (
                    await self.check_available_usage(
                        model=model_info["model_name"], priority=key_priority
                    )
                )
                response._hidden_params["additional_headers"] = (
                    {  # Add additional response headers - easier debugging
                        "x-litellm-model_group": model_info["model_name"],
                        "x-ratelimit-remaining-litellm-project-tokens": available_tpm,
                        "x-ratelimit-remaining-litellm-project-requests": available_rpm,
                        "x-ratelimit-remaining-model-tokens": model_tpm,
                        "x-ratelimit-remaining-model-requests": model_rpm,
                        "x-ratelimit-current-active-projects": active_projects,
                    }
                )

                return response
            return await super().async_post_call_success_hook(
                data=data,
                user_api_key_dict=user_api_key_dict,
                response=response,
            )
        except Exception as e:
            verbose_proxy_logger.exception(
                "litellm.proxy.hooks.dynamic_rate_limiter.py::async_post_call_success_hook(): Exception occured - {}".format(
                    str(e)
                )
            )
            return response
