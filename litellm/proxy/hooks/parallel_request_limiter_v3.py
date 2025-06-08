"""
This is a rate limiter implementation based on a similar one by Envoy proxy. 

This is currently in development and not yet ready for production.
"""
import os
import sys
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Literal, Optional, TypedDict, Union

from fastapi import HTTPException

import litellm
from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache
    from litellm.types.caching import RedisPipelineIncrementOperation

    Span = Union[_Span, Any]
    InternalUsageCache = _InternalUsageCache
else:
    Span = Any
    InternalUsageCache = Any

BATCH_RATE_LIMITER_SCRIPT = """
local results = {}
local now = tonumber(ARGV[1])
local window_size = tonumber(ARGV[2])

-- Process each window/counter pair
for i = 1, #KEYS, 2 do
    local window_key = KEYS[i]
    local counter_key = KEYS[i + 1]
    local increment_value = tonumber(KEYS[i + 2]) or 1

    -- Check if window exists and is valid
    local window_start = redis.call('GET', window_key)
    if not window_start or (now - tonumber(window_start)) >= window_size then
        -- Reset window and counter
        redis.call('SET', window_key, tostring(now))
        redis.call('SET', counter_key, increment_value)
        redis.call('EXPIRE', window_key, window_size)
        redis.call('EXPIRE', counter_key, window_size)
        table.insert(results, tostring(now)) -- window_start
        table.insert(results, increment_value) -- counter
    else
        local counter = redis.call('INCR', counter_key)
        table.insert(results, window_start) -- window_start
        table.insert(results, counter) -- counter
    end
end

return results
"""


class RateLimitDescriptorRateLimitObject(TypedDict, total=False):
    requests_per_unit: Optional[int]
    tokens_per_unit: Optional[int]
    max_parallel_requests: Optional[int]
    window_size: Optional[int]


class RateLimitDescriptor(TypedDict):
    key: str
    value: str
    rate_limit: Optional[RateLimitDescriptorRateLimitObject]


class RateLimitResponse(TypedDict):
    overall_code: str
    statuses: List[Dict[str, Any]]


class _PROXY_MaxParallelRequestsHandler_v3(CustomLogger):
    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            self.batch_rate_limiter_script = (
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                    BATCH_RATE_LIMITER_SCRIPT
                )
            )
        else:
            self.batch_rate_limiter_script = None

        self.window_size = int(os.getenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", 60))

    def print_verbose(self, print_statement):
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa
        except Exception:
            pass

    def create_rate_limit_keys(
        self,
        key: str,
        value: str,
        rate_limit_type: Literal["requests", "tokens", "max_parallel_requests"],
    ) -> str:
        """
        Create the rate limit keys for the given key and value.
        """
        counter_key = f"{{{key}:{value}}}:{rate_limit_type}"

        return counter_key

    def is_cache_list_over_limit(
        self,
        keys_to_fetch: List[str],
        cache_values: List[Any],
        key_metadata: Dict[str, Any],
    ) -> RateLimitResponse:
        """
        Check if the cache values are over the limit.
        """
        statuses = []
        overall_code = "OK"
        for i in range(0, len(cache_values), 2):
            item_code = "OK"
            window_key = keys_to_fetch[i]
            counter_key = keys_to_fetch[i + 1]
            counter_value = cache_values[i + 1]
            requests_limit = key_metadata[window_key]["requests_limit"]
            max_parallel_requests_limit = key_metadata[window_key][
                "max_parallel_requests_limit"
            ]
            tokens_limit = key_metadata[window_key]["tokens_limit"]

            # Determine which limit to use for current_limit and limit_remaining
            if counter_key.endswith(":requests"):
                current_limit = requests_limit
            elif counter_key.endswith(":max_parallel_requests"):
                current_limit = max_parallel_requests_limit
            elif counter_key.endswith(":tokens"):
                current_limit = tokens_limit
            else:
                current_limit = None

            if (
                counter_key.endswith(":requests")
                and requests_limit is not None
                and counter_value is not None
                and int(counter_value) + 1 > requests_limit
            ):
                overall_code = "OVER_LIMIT"
                item_code = "OVER_LIMIT"
            elif (
                counter_key.endswith(":max_parallel_requests")
                and max_parallel_requests_limit is not None
                and counter_value is not None
                and int(counter_value) + 1 > max_parallel_requests_limit
            ):
                overall_code = "OVER_LIMIT"
                item_code = "OVER_LIMIT"
            elif (
                counter_key.endswith(":tokens")
                and tokens_limit is not None
                and counter_value is not None
                and int(counter_value) + 1 > tokens_limit
            ):
                overall_code = "OVER_LIMIT"
                item_code = "OVER_LIMIT"

            # Only compute limit_remaining if current_limit is not None
            if current_limit is None:
                limit_remaining = None
            elif counter_value is None:
                limit_remaining = current_limit
            else:
                limit_remaining = current_limit - counter_value

            statuses.append(
                {
                    "code": item_code,
                    "current_limit": current_limit,
                    "limit_remaining": limit_remaining,
                }
            )

        return RateLimitResponse(overall_code=overall_code, statuses=statuses)

    async def should_rate_limit(
        self,
        descriptors: List[RateLimitDescriptor],
        parent_otel_span: Optional[Span] = None,
        read_only: bool = False,
    ) -> RateLimitResponse:
        """
        Check if any of the rate limit descriptors should be rate limited.
        Returns a RateLimitResponse with the overall code and status for each descriptor.
        Uses batch operations for Redis to improve performance.
        """

        now = datetime.now().timestamp()
        now_int = int(now)  # Convert to integer for Redis Lua script

        # Collect all keys and their metadata upfront
        keys_to_fetch: List[str] = []
        key_metadata = {}  # Store metadata for each key
        for descriptor in descriptors:
            key = descriptor["key"]
            value = descriptor["value"]
            rate_limit = descriptor.get("rate_limit", {}) or {}
            requests_limit = rate_limit.get("requests_per_unit")
            tokens_limit = rate_limit.get("tokens_per_unit")
            max_parallel_requests_limit = rate_limit.get("max_parallel_requests")
            window_size = rate_limit.get("window_size") or self.window_size

            window_key = f"{{{key}:{value}}}:window"

            if requests_limit is not None:
                key = self.create_rate_limit_keys(key, value, "requests")
                keys_to_fetch.extend([window_key, key])
            elif tokens_limit is not None:
                key = self.create_rate_limit_keys(key, value, "tokens")
                keys_to_fetch.extend([window_key, key])
            elif max_parallel_requests_limit is not None:
                key = self.create_rate_limit_keys(key, value, "max_parallel_requests")
                keys_to_fetch.extend([window_key, key])
            else:
                continue

            key_metadata[window_key] = {
                "requests_limit": int(requests_limit)
                if requests_limit is not None
                else None,
                "tokens_limit": int(tokens_limit) if tokens_limit is not None else None,
                "max_parallel_requests_limit": int(max_parallel_requests_limit)
                if max_parallel_requests_limit is not None
                else None,
                "window_size": int(window_size),
            }

        ## CHECK IN-MEMORY CACHE
        cache_values = await self.internal_usage_cache.async_batch_get_cache(
            keys=keys_to_fetch,
            parent_otel_span=parent_otel_span,
            local_only=True,
        )

        if cache_values is not None:
            rate_limit_response = self.is_cache_list_over_limit(
                keys_to_fetch, cache_values, key_metadata
            )
            if rate_limit_response["overall_code"] == "OVER_LIMIT":
                return rate_limit_response

        ## IF under limit, check Redis
        if self.batch_rate_limiter_script is not None:
            cache_values = await self.batch_rate_limiter_script(
                keys=keys_to_fetch,
                args=[now_int, self.window_size],  # Use integer timestamp
            )
            # update in-memory cache with new values
            for i in range(0, len(cache_values), 2):
                window_key = keys_to_fetch[i]
                counter_key = keys_to_fetch[i + 1]
                window_value = cache_values[i]
                counter_value = cache_values[i + 1]
                await self.internal_usage_cache.async_set_cache(
                    key=counter_key,
                    value=counter_value,
                    ttl=self.window_size,
                    litellm_parent_otel_span=parent_otel_span,
                    local_only=True,
                )
                await self.internal_usage_cache.async_set_cache(
                    key=window_key,
                    value=window_value,
                    ttl=self.window_size,
                    litellm_parent_otel_span=parent_otel_span,
                    local_only=True,
                )
        else:
            raise ValueError("Batch rate limiter script is not initialized")

        rate_limit_response = self.is_cache_list_over_limit(
            keys_to_fetch, cache_values, key_metadata
        )
        return rate_limit_response

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
        from litellm.proxy.auth.auth_utils import (
            get_key_model_rpm_limit,
            get_key_model_tpm_limit,
        )

        verbose_proxy_logger.debug("Inside Rate Limit Pre-Call Hook")

        # Create rate limit descriptors
        descriptors = []

        # API Key rate limits
        if user_api_key_dict.api_key:
            descriptors.append(
                RateLimitDescriptor(
                    key="api_key",
                    value=user_api_key_dict.api_key,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.rpm_limit,
                        "tokens_per_unit": user_api_key_dict.tpm_limit,
                        "max_parallel_requests": user_api_key_dict.max_parallel_requests,
                        "window_size": self.window_size,  # 1 minute window
                    },
                )
            )

        # User rate limits
        if user_api_key_dict.user_id:
            descriptors.append(
                RateLimitDescriptor(
                    key="user",
                    value=user_api_key_dict.user_id,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.user_rpm_limit,
                        "tokens_per_unit": user_api_key_dict.user_tpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

        # Team rate limits
        if user_api_key_dict.team_id:
            descriptors.append(
                RateLimitDescriptor(
                    key="team",
                    value=user_api_key_dict.team_id,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.team_rpm_limit,
                        "tokens_per_unit": user_api_key_dict.team_tpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

        # End user rate limits
        if user_api_key_dict.end_user_id:
            descriptors.append(
                RateLimitDescriptor(
                    key="end_user",
                    value=user_api_key_dict.end_user_id,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.end_user_rpm_limit,
                        "tokens_per_unit": user_api_key_dict.end_user_tpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

        # Model rate limits
        requested_model = data.get("model", None)
        if requested_model and (
            get_key_model_tpm_limit(user_api_key_dict) is not None
            or get_key_model_rpm_limit(user_api_key_dict) is not None
        ):
            _tpm_limit_for_key_model = get_key_model_tpm_limit(user_api_key_dict) or {}
            _rpm_limit_for_key_model = get_key_model_rpm_limit(user_api_key_dict) or {}
            should_check_rate_limit = False
            if requested_model in _tpm_limit_for_key_model:
                should_check_rate_limit = True
            elif requested_model in _rpm_limit_for_key_model:
                should_check_rate_limit = True

            if should_check_rate_limit:
                model_specific_tpm_limit: Optional[int] = None
                model_specific_rpm_limit: Optional[int] = None
                if requested_model in _tpm_limit_for_key_model:
                    model_specific_tpm_limit = _tpm_limit_for_key_model[requested_model]
                if requested_model in _rpm_limit_for_key_model:
                    model_specific_rpm_limit = _rpm_limit_for_key_model[requested_model]
                descriptors.append(
                    RateLimitDescriptor(
                        key="model_per_key",
                        value=f"{user_api_key_dict.api_key}:{requested_model}",
                        rate_limit={
                            "requests_per_unit": model_specific_rpm_limit,
                            "tokens_per_unit": model_specific_tpm_limit,
                            "window_size": self.window_size,
                        },
                    )
                )

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
                        headers={
                            "retry-after": str(self.window_size)
                        },  # Retry after 1 minute
                    )

    def _create_pipeline_operations(
        self,
        key: str,
        value: str,
        rate_limit_type: Literal["requests", "tokens", "max_parallel_requests"],
        total_tokens: int,
    ) -> List["RedisPipelineIncrementOperation"]:
        """
        Create pipeline operations for TPM increments
        """
        from litellm.types.caching import RedisPipelineIncrementOperation

        pipeline_operations: List[RedisPipelineIncrementOperation] = []
        counter_key = self.create_rate_limit_keys(
            key=key,
            value=value,
            rate_limit_type="tokens",
        )
        pipeline_operations.append(
            RedisPipelineIncrementOperation(
                key=counter_key,
                increment_value=total_tokens,
                ttl=self.window_size,
            )
        )

        return pipeline_operations

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Update TPM usage on successful API calls by incrementing counters using pipeline
        """
        from litellm.litellm_core_utils.core_helpers import (
            _get_parent_otel_span_from_kwargs,
        )
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )
        from litellm.types.caching import RedisPipelineIncrementOperation
        from litellm.types.utils import ModelResponse, Usage

        litellm_parent_otel_span: Union[Span, None] = _get_parent_otel_span_from_kwargs(
            kwargs
        )
        try:
            self.print_verbose("INSIDE parallel request limiter ASYNC SUCCESS LOGGING")

            # Get metadata from kwargs
            user_api_key = kwargs["litellm_params"]["metadata"].get("user_api_key")
            user_api_key_user_id = kwargs["litellm_params"]["metadata"].get(
                "user_api_key_user_id"
            )
            user_api_key_team_id = kwargs["litellm_params"]["metadata"].get(
                "user_api_key_team_id"
            )
            user_api_key_end_user_id = kwargs.get("user") or kwargs["litellm_params"][
                "metadata"
            ].get("user_api_key_end_user_id")
            model_group = get_model_group_from_litellm_kwargs(kwargs)

            # Get total tokens from response
            total_tokens = 0
            if isinstance(response_obj, ModelResponse):
                _usage = getattr(response_obj, "usage", None)
                if _usage and isinstance(_usage, Usage):
                    total_tokens = _usage.total_tokens

            # Create pipeline operations for TPM increments
            pipeline_operations: List[RedisPipelineIncrementOperation] = []

            # API Key TPM
            if user_api_key:
                # MAX PARALLEL REQUESTS - only support for API Key, just decrement the counter
                counter_key = self.create_rate_limit_keys(
                    key="api_key",
                    value=user_api_key,
                    rate_limit_type="max_parallel_requests",
                )
                pipeline_operations.append(
                    RedisPipelineIncrementOperation(
                        key=counter_key,
                        increment_value=-1,
                        ttl=self.window_size,
                    )
                )
                pipeline_operations.extend(
                    self._create_pipeline_operations(
                        key="api_key",
                        value=user_api_key,
                        rate_limit_type="tokens",
                        total_tokens=total_tokens,
                    )
                )

            # User TPM
            if user_api_key_user_id:
                # TPM
                pipeline_operations.extend(
                    self._create_pipeline_operations(
                        key="user",
                        value=user_api_key_user_id,
                        rate_limit_type="tokens",
                        total_tokens=total_tokens,
                    )
                )

            # Team TPM
            if user_api_key_team_id:
                pipeline_operations.extend(
                    self._create_pipeline_operations(
                        key="team",
                        value=user_api_key_team_id,
                        rate_limit_type="tokens",
                        total_tokens=total_tokens,
                    )
                )

            # End User TPM
            if user_api_key_end_user_id:
                pipeline_operations.extend(
                    self._create_pipeline_operations(
                        key="end_user",
                        value=user_api_key_end_user_id,
                        rate_limit_type="tokens",
                        total_tokens=total_tokens,
                    )
                )

            # Model-specific TPM
            if model_group and user_api_key:
                pipeline_operations.extend(
                    self._create_pipeline_operations(
                        key="model_per_key",
                        value=f"{user_api_key}:{model_group}",
                        rate_limit_type="tokens",
                        total_tokens=total_tokens,
                    )
                )

            # Execute all increments in a single pipeline
            if pipeline_operations:
                await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                    increment_list=pipeline_operations,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                )

        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error in rate limit success event: {str(e)}"
            )

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
                            "window_size": self.window_size,
                        },
                    )
                )

            # Check rate limits
            # rate_limit_response = await self.should_rate_limit(
            #     descriptors=descriptors,
            #     parent_otel_span=user_api_key_dict.parent_otel_span,
            #     read_only=True,
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
            #     for i, status in enumerate(rate_limit_response["statuses"]):
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
            verbose_proxy_logger.exception(
                f"Error in rate limit post-call hook: {str(e)}"
            )
