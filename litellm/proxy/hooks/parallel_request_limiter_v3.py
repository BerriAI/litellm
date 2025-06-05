"""
This is a rate limiter implementation based on a similar one by Envoy proxy. 

This is currently in development and not yet ready for production.
"""
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, TypedDict, Union

from fastapi import HTTPException
from pydantic import BaseModel

import litellm
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    Span = Union[_Span, Any]
    InternalUsageCache = _InternalUsageCache
else:
    Span = Any
    InternalUsageCache = Any


class RateLimitDescriptor(TypedDict):
    key: str
    value: str
    rate_limit: Optional[Dict[str, int]]


class RateLimitResponse(TypedDict):
    overall_code: str
    statuses: List[Dict[str, Any]]


class _PROXY_MaxParallelRequestsHandler_v3(CustomLogger):
    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except Exception:
            pass

    async def should_rate_limit(
        self,
        descriptors: List[RateLimitDescriptor],
        parent_otel_span: Optional[Span] = None,
    ) -> RateLimitResponse:
        """
        Check if any of the rate limit descriptors should be rate limited.
        Returns a RateLimitResponse with the overall code and status for each descriptor.
        """
        statuses = []
        overall_code = "OK"

        for descriptor in descriptors:
            key = descriptor["key"]
            value = descriptor["value"]
            rate_limit = descriptor.get("rate_limit", {}) or {}

            # Get the rate limit
            requests_limit = rate_limit.get("requests_per_unit", sys.maxsize)
            window_size = rate_limit.get("window_size", 60)  # Default 60 second window

            # Use atomic operations to check and increment in one go
            try:
                # Get current window info
                window_key = f"{key}:{value}:window"
                current_window = await self.internal_usage_cache.async_get_cache(
                    key=window_key,
                    litellm_parent_otel_span=parent_otel_span,
                )

                now = datetime.now().timestamp()

                # If no window exists or window has expired, reset it
                # if current_window is None or (now - current_window) >= window_size:
                #     print("resetting window")
                #     # Set new window start time
                #     await self.internal_usage_cache.async_set_cache(
                #         key=window_key,
                #         value=now,
                #         ttl=window_size,
                #         litellm_parent_otel_span=parent_otel_span,
                #     )
                #     # Reset counter
                #     await self.internal_usage_cache.async_set_cache(
                #         key=f"{key}:{value}:requests",
                #         value=0,
                #         ttl=window_size,
                #         litellm_parent_otel_span=parent_otel_span,
                #     )
                #     current_count = 0
                # else:
                #     # Get current count
                #     current_count = (
                #         await self.internal_usage_cache.async_get_cache(
                #             key=f"{key}:{value}:requests",
                #             litellm_parent_otel_span=parent_otel_span,
                #         )
                #         or 0
                #     )
                # Get current count
                current_count = (
                    await self.internal_usage_cache.async_get_cache(
                        key=f"{key}:{value}:requests",
                        litellm_parent_otel_span=parent_otel_span,
                    )
                    or 0
                )

                if current_count >= requests_limit:
                    status = {
                        "code": "OVER_LIMIT",
                        "current_limit": requests_limit,
                        "limit_remaining": 0,
                    }
                    overall_code = "OVER_LIMIT"
                    statuses.append(status)
                    continue

                # If we're under the limit, try to increment atomically
                new_count = await self.internal_usage_cache.async_increment_cache(
                    key=f"{key}:{value}:requests",
                    value=1,
                    ttl=window_size,
                    litellm_parent_otel_span=parent_otel_span,
                )

                # Double check after increment
                if new_count > requests_limit:
                    # We went over the limit, decrement back
                    await self.internal_usage_cache.async_increment_cache(
                        key=f"{key}:{value}:requests",
                        value=-1,
                        ttl=window_size,
                        litellm_parent_otel_span=parent_otel_span,
                    )
                    status = {
                        "code": "OVER_LIMIT",
                        "current_limit": requests_limit,
                        "limit_remaining": 0,
                    }
                    overall_code = "OVER_LIMIT"
                else:
                    status = {
                        "code": "OK",
                        "current_limit": requests_limit,
                        "limit_remaining": max(0, requests_limit - new_count),
                    }
            except Exception as e:
                verbose_proxy_logger.error(f"Error in rate limit check: {str(e)}")
                status = {
                    "code": "OVER_LIMIT",
                    "current_limit": requests_limit,
                    "limit_remaining": 0,
                }
                overall_code = "OVER_LIMIT"

            statuses.append(status)

        return RateLimitResponse(overall_code=overall_code, statuses=statuses)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        """
        Pre-call hook to check rate limits before making the API call.
        """
        self.print_verbose("Inside Rate Limit Pre-Call Hook")

        # Create rate limit descriptors
        descriptors = []

        # API Key rate limits
        if user_api_key_dict.api_key:
            descriptors.append(
                RateLimitDescriptor(
                    key="api_key",
                    value=user_api_key_dict.api_key,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.rpm_limit or sys.maxsize,
                        "window_size": 60,  # 1 minute window
                    },
                )
            )

        # # User rate limits
        # if user_api_key_dict.user_id:
        #     descriptors.append(
        #         RateLimitDescriptor(
        #             key="user",
        #             value=user_api_key_dict.user_id,
        #             rate_limit={
        #                 "requests_per_unit": user_api_key_dict.user_rpm_limit
        #                 or sys.maxsize,
        #                 "window_size": 60,
        #             },
        #         )
        #     )

        # # Team rate limits
        # if user_api_key_dict.team_id:
        #     descriptors.append(
        #         RateLimitDescriptor(
        #             key="team",
        #             value=user_api_key_dict.team_id,
        #             rate_limit={
        #                 "requests_per_unit": user_api_key_dict.team_rpm_limit
        #                 or sys.maxsize,
        #                 "window_size": 60,
        #             },
        #         )
        #     )

        # # End user rate limits
        # if user_api_key_dict.end_user_id:
        #     descriptors.append(
        #         RateLimitDescriptor(
        #             key="end_user",
        #             value=user_api_key_dict.end_user_id,
        #             rate_limit={
        #                 "requests_per_unit": getattr(
        #                     user_api_key_dict, "end_user_rpm_limit", sys.maxsize
        #                 ),
        #                 "window_size": 60,
        #             },
        #         )
        #     )

        # Check rate limits
        response = await self.should_rate_limit(
            descriptors=descriptors,
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )

        if response["overall_code"] == "OVER_LIMIT":
            # Find which descriptor hit the limit
            for i, status in enumerate(response["statuses"]):
                if status["code"] == "OVER_LIMIT":
                    descriptor = descriptors[i]
                    raise HTTPException(
                        status_code=429,
                        detail=f"Rate limit exceeded for {descriptor['key']}: {descriptor['value']}. Remaining: {status['limit_remaining']}",
                        headers={"retry-after": str(60)},  # Retry after 1 minute
                    )

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        No-op for success event since we handle increments in should_rate_limit
        """
        pass

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        No-op for failure event since we handle increments in should_rate_limit
        """
        pass

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        """
        Post-call hook to update rate limit headers in the response.
        """
        try:
            descriptors = []

            # API Key
            if user_api_key_dict.api_key:
                descriptors.append(
                    RateLimitDescriptor(
                        key="api_key",
                        value=user_api_key_dict.api_key,
                        rate_limit={
                            "requests_per_unit": user_api_key_dict.rpm_limit
                            or sys.maxsize,
                            "window_size": 60,
                        },
                    )
                )

            # # Get current limits
            # response = await self.should_rate_limit(
            #     descriptors=descriptors,
            #     parent_otel_span=user_api_key_dict.parent_otel_span,
            # )

            # # Update response headers
            # if hasattr(response, "_hidden_params"):
            #     _hidden_params = getattr(response, "_hidden_params")
            # else:
            #     _hidden_params = None

            # if _hidden_params is not None and (
            #     isinstance(_hidden_params, BaseModel)
            #     or isinstance(_hidden_params, dict)
            # ):
            #     if isinstance(_hidden_params, BaseModel):
            #         _hidden_params = _hidden_params.model_dump()

            #     _additional_headers = _hidden_params.get("additional_headers", {}) or {}

            #     # Add rate limit headers
            #     for i, status in enumerate(response["statuses"]):
            #         descriptor = descriptors[i]
            #         prefix = f"x-ratelimit-{descriptor['key']}"
            #         _additional_headers[f"{prefix}-remaining-requests"] = status[
            #             "limit_remaining"
            #         ]
            #         _additional_headers[f"{prefix}-limit-requests"] = status[
            #             "current_limit"
            #         ]

            #     setattr(
            #         response,
            #         "_hidden_params",
            #         {**_hidden_params, "additional_headers": _additional_headers},
            #     )

        except Exception as e:
            self.print_verbose(f"Error in post-call hook: {str(e)}")
