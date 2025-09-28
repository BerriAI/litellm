"""
This is a rate limiter implementation based on a similar one by Envoy proxy.

This is currently in development and not yet ready for production.
"""

import os
from datetime import datetime
from math import floor
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    List,
    Literal,
    Optional,
    TypedDict,
    Union,
    cast,
)

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject

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
    local increment_value = 1

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

TOKEN_INCREMENT_SCRIPT = """
local results = {}

-- Process each key/increment_value/ttl triplet
for i = 1, #KEYS do
    local key = KEYS[i]
    local increment_value = tonumber(ARGV[i * 2 - 1])
    local ttl_seconds = tonumber(ARGV[i * 2])

    -- Increment the value
    local new_value = redis.call('INCRBYFLOAT', key, increment_value)

    -- Handle TTL: only set expire if ttl_seconds > 0 and key has no current TTL
    -- ttl_seconds can be 0 (no TTL) or positive (set TTL)
    if ttl_seconds and ttl_seconds > 0 then
        local current_ttl = redis.call('TTL', key)
        if current_ttl == -1 then
            redis.call('EXPIRE', key, ttl_seconds)
        end
    end

    table.insert(results, new_value)
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


class RateLimitStatus(TypedDict):
    code: str
    current_limit: int
    limit_remaining: int
    rate_limit_type: Literal["requests", "tokens", "max_parallel_requests"]
    descriptor_key: str


class RateLimitResponse(TypedDict):
    overall_code: str
    statuses: List[RateLimitStatus]


class RateLimitResponseWithDescriptors(TypedDict):
    descriptors: List[RateLimitDescriptor]
    response: RateLimitResponse


class _PROXY_MaxParallelRequestsHandler_v3(CustomLogger):
    def __init__(self, internal_usage_cache: InternalUsageCache):
        self.internal_usage_cache = internal_usage_cache
        if self.internal_usage_cache.dual_cache.redis_cache is not None:
            self.batch_rate_limiter_script = (
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                    BATCH_RATE_LIMITER_SCRIPT
                )
            )
            self.token_increment_script = (
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                    TOKEN_INCREMENT_SCRIPT
                )
            )
        else:
            self.batch_rate_limiter_script = None
            self.token_increment_script = None

        self.window_size = int(os.getenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", 60))

    async def in_memory_cache_sliding_window(
        self,
        keys: List[str],
        now_int: int,
        window_size: int,
    ) -> List[Any]:
        """
        Implement sliding window rate limiting logic using in-memory cache operations.
        This follows the same logic as the Redis Lua script but uses async cache operations.
        """
        results: List[Any] = []

        # Process each window/counter pair
        for i in range(0, len(keys), 2):
            window_key = keys[i]
            counter_key = keys[i + 1]
            increment_value = 1

            # Get the window start time
            window_start = await self.internal_usage_cache.async_get_cache(
                key=window_key,
                litellm_parent_otel_span=None,
                local_only=True,
            )

            # Check if window exists and is valid
            if window_start is None or (now_int - int(window_start)) >= window_size:
                # Reset window and counter
                await self.internal_usage_cache.async_set_cache(
                    key=window_key,
                    value=str(now_int),
                    ttl=window_size,
                    litellm_parent_otel_span=None,
                    local_only=True,
                )
                await self.internal_usage_cache.async_set_cache(
                    key=counter_key,
                    value=increment_value,
                    ttl=window_size,
                    litellm_parent_otel_span=None,
                    local_only=True,
                )
                results.append(str(now_int))  # window_start
                results.append(increment_value)  # counter
            else:
                # Increment the counter
                current_counter = await self.internal_usage_cache.async_get_cache(
                    key=counter_key,
                    litellm_parent_otel_span=None,
                    local_only=True,
                )
                new_counter_value = (
                    int(current_counter) if current_counter is not None else 0
                ) + increment_value
                await self.internal_usage_cache.async_set_cache(
                    key=counter_key,
                    value=new_counter_value,
                    ttl=window_size,
                    litellm_parent_otel_span=None,
                    local_only=True,
                )
                results.append(window_start)  # window_start
                results.append(new_counter_value)  # counter

        return results

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
        statuses: List[RateLimitStatus] = []
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
            current_limit: Optional[int] = None
            rate_limit_type: Optional[
                Literal["requests", "tokens", "max_parallel_requests"]
            ] = None
            if counter_key.endswith(":requests"):
                current_limit = requests_limit
                rate_limit_type = "requests"
            elif counter_key.endswith(":max_parallel_requests"):
                current_limit = max_parallel_requests_limit
                rate_limit_type = "max_parallel_requests"
            elif counter_key.endswith(":tokens"):
                current_limit = tokens_limit
                rate_limit_type = "tokens"

            if current_limit is None or rate_limit_type is None:
                continue

            if counter_value is not None and int(counter_value) > current_limit:
                overall_code = "OVER_LIMIT"
                item_code = "OVER_LIMIT"

            # Only compute limit_remaining if current_limit is not None
            limit_remaining = (
                current_limit - int(counter_value)
                if counter_value is not None
                else current_limit
            )

            statuses.append(
                {
                    "code": item_code,
                    "current_limit": current_limit,
                    "limit_remaining": limit_remaining,
                    "rate_limit_type": rate_limit_type,
                    "descriptor_key": key_metadata[window_key]["descriptor_key"],
                }
            )

        return RateLimitResponse(overall_code=overall_code, statuses=statuses)

    def _group_keys_by_hash_tag(self, keys: List[str]) -> Dict[str, List[str]]:
        """
        Group keys by their Redis hash tag to ensure cluster compatibility.
        Keys with the same hash tag will be processed together.
        """
        groups: Dict[str, List[str]] = {}
        for key in keys:
            # Extract hash tag from key like "{api_key:sk-123}:requests"
            if "{" in key and "}" in key:
                start = key.find("{")
                end = key.find("}", start)
                hash_tag = key[start : end + 1]
            else:
                # Fallback for keys without hash tags
                hash_tag = "no_hash_tag"

            if hash_tag not in groups:
                groups[hash_tag] = []
            groups[hash_tag].append(key)

        return groups

    async def _execute_redis_batch_rate_limiter_script(
        self,
        keys_to_fetch: List[str],
        now_int: int,
    ) -> List[Any]:
        """
        Execute Redis operations grouped by hash tag for cluster compatibility.

        Args:
            keys_to_fetch: List[str] - List of keys to fetch
            now_int: int - Current timestamp

        Returns:
            List[Any] - List of cache values
        """
        if self.batch_rate_limiter_script is None:
            return []

        key_groups = self._group_keys_by_hash_tag(keys_to_fetch)
        all_cache_values = []

        for hash_tag, group_keys in key_groups.items():
            try:
                group_cache_values = await self.batch_rate_limiter_script(
                    keys=group_keys,
                    args=[now_int, self.window_size],  # Use integer timestamp
                )
                all_cache_values.extend(group_cache_values)
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"Redis Lua script failed for hash tag {hash_tag}: {str(e)}"
                )
                # Fallback to in-memory cache for this group
                group_cache_values = await self.in_memory_cache_sliding_window(
                    keys=group_keys,
                    now_int=now_int,
                    window_size=self.window_size,
                )
                all_cache_values.extend(group_cache_values)

        return all_cache_values

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
            descriptor_key = descriptor["key"]
            descriptor_value = descriptor["value"]
            rate_limit: RateLimitDescriptorRateLimitObject = (
                descriptor.get("rate_limit") or RateLimitDescriptorRateLimitObject()
            )
            requests_limit = rate_limit.get("requests_per_unit")
            tokens_limit = rate_limit.get("tokens_per_unit")
            max_parallel_requests_limit = rate_limit.get("max_parallel_requests")
            window_size = rate_limit.get("window_size") or self.window_size

            window_key = f"{{{descriptor_key}:{descriptor_value}}}:window"

            rate_limit_set = False
            if requests_limit is not None:
                rpm_key = self.create_rate_limit_keys(
                    descriptor_key, descriptor_value, "requests"
                )
                keys_to_fetch.extend([window_key, rpm_key])
                rate_limit_set = True
            if tokens_limit is not None:
                tpm_key = self.create_rate_limit_keys(
                    descriptor_key, descriptor_value, "tokens"
                )
                keys_to_fetch.extend([window_key, tpm_key])
                rate_limit_set = True
            if max_parallel_requests_limit is not None:
                max_parallel_requests_key = self.create_rate_limit_keys(
                    descriptor_key, descriptor_value, "max_parallel_requests"
                )
                keys_to_fetch.extend([window_key, max_parallel_requests_key])
                rate_limit_set = True

            if not rate_limit_set:
                continue

            key_metadata[window_key] = {
                "requests_limit": (
                    int(requests_limit) if requests_limit is not None else None
                ),
                "tokens_limit": int(tokens_limit) if tokens_limit is not None else None,
                "max_parallel_requests_limit": (
                    int(max_parallel_requests_limit)
                    if max_parallel_requests_limit is not None
                    else None
                ),
                "window_size": int(window_size),
                "descriptor_key": descriptor_key,
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
            # Group keys by hash tag for Redis cluster compatibility
            cache_values = await self._execute_redis_batch_rate_limiter_script(
                keys_to_fetch=keys_to_fetch,
                now_int=now_int,
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
            cache_values = await self.in_memory_cache_sliding_window(
                keys=keys_to_fetch,
                now_int=now_int,
                window_size=self.window_size,
            )

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
        if user_api_key_dict.api_key and (
            user_api_key_dict.rpm_limit is not None
            or user_api_key_dict.tpm_limit is not None
            or user_api_key_dict.max_parallel_requests is not None
        ):
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
        if user_api_key_dict.user_id and (
            user_api_key_dict.user_rpm_limit is not None
            or user_api_key_dict.user_tpm_limit is not None
        ):
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
        if user_api_key_dict.team_id and (
            user_api_key_dict.team_rpm_limit is not None
            or user_api_key_dict.team_tpm_limit is not None
        ):
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

        # Team Member rate limits
        if user_api_key_dict.user_id and (
            user_api_key_dict.team_member_rpm_limit is not None
            or user_api_key_dict.team_member_tpm_limit is not None
        ):
            team_member_value = (
                f"{user_api_key_dict.team_id}:{user_api_key_dict.user_id}"
            )
            descriptors.append(
                RateLimitDescriptor(
                    key="team_member",
                    value=team_member_value,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.team_member_rpm_limit,
                        "tokens_per_unit": user_api_key_dict.team_member_tpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

        # End user rate limits
        if user_api_key_dict.end_user_id and (
            user_api_key_dict.end_user_rpm_limit is not None
            or user_api_key_dict.end_user_tpm_limit is not None
        ):
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

        # Only check rate limits if we have descriptors with actual limits
        if descriptors:
            response = await self.should_rate_limit(
                descriptors=descriptors,
                parent_otel_span=user_api_key_dict.parent_otel_span,
            )

            if response["overall_code"] == "OVER_LIMIT":
                # Find which descriptor hit the limit
                for i, status in enumerate(response["statuses"]):
                    if status["code"] == "OVER_LIMIT":
                        descriptor = descriptors[floor(i / 2)]

                        # Calculate reset time (window_start + window_size)
                        now = datetime.now().timestamp()
                        reset_time = now + self.window_size  # Conservative estimate
                        reset_time_formatted = datetime.fromtimestamp(
                            reset_time
                        ).strftime("%Y-%m-%d %H:%M:%S UTC")

                        # Handle negative remaining values more gracefully
                        remaining_display = max(0, status["limit_remaining"])

                        # Create detailed error message
                        rate_limit_type = status["rate_limit_type"]
                        current_limit = status["current_limit"]

                        detail = (
                            f"Rate limit exceeded for {descriptor['key']}: {descriptor['value']}. "
                            f"Limit type: {rate_limit_type}. "
                            f"Current limit: {current_limit}, Remaining: {remaining_display}. "
                            f"Limit resets at: {reset_time_formatted}"
                        )

                        raise HTTPException(
                            status_code=429,
                            detail=detail,
                            headers={
                                "retry-after": str(self.window_size),
                                "rate_limit_type": str(status["rate_limit_type"]),
                                "reset_at": reset_time_formatted,
                            },
                        )

            else:
                # add descriptors to request headers
                data["litellm_proxy_rate_limit_response"] = response

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

    async def _execute_token_increment_script(
        self,
        pipeline_operations: List["RedisPipelineIncrementOperation"],
    ) -> None:
        """
        Execute token increment script grouped by hash tag for cluster compatibility.
        """
        if self.token_increment_script is None:
            return

        # Group operations by hash tag for Redis cluster compatibility
        operation_keys = [op["key"] for op in pipeline_operations]
        key_groups = self._group_keys_by_hash_tag(operation_keys)

        for _hash_tag, group_keys in key_groups.items():
            # Get operations for this hash tag group
            group_operations = [
                op for op in pipeline_operations if op["key"] in group_keys
            ]

            keys = []
            args = []

            for op in group_operations:
                # Convert None TTL to 0 for Lua script
                ttl_value = op["ttl"] if op["ttl"] is not None else 0

                verbose_proxy_logger.debug(
                    f"Executing TTL-preserving increment for key={op['key']}, "
                    f"increment={op['increment_value']}, ttl={ttl_value}"
                )
                keys.append(op["key"])
                args.extend([op["increment_value"], ttl_value])

            await self.token_increment_script(
                keys=keys,
                args=args,
            )

    async def async_increment_tokens_with_ttl_preservation(
        self,
        pipeline_operations: List["RedisPipelineIncrementOperation"],
        parent_otel_span: Optional[Span] = None,
    ) -> None:
        """
        Increment token counters using Lua script to preserve existing TTL.
        This prevents TTL reset on every token increment.
        """
        if not pipeline_operations:
            return

        # Check if script is available
        if self.token_increment_script is None:
            verbose_proxy_logger.debug(
                "TTL preservation script not available, using regular pipeline"
            )
            await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                increment_list=pipeline_operations,
                litellm_parent_otel_span=parent_otel_span,
            )
            return

        try:
            await self._execute_token_increment_script(pipeline_operations)

            verbose_proxy_logger.debug(
                f"Successfully executed TTL-preserving increment for {len(pipeline_operations)} keys"
            )

        except Exception as e:
            verbose_proxy_logger.warning(
                f"TTL preservation failed, falling back to regular pipeline: {str(e)}"
            )
            # Fallback to regular pipeline on error
            await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                increment_list=pipeline_operations,
                litellm_parent_otel_span=parent_otel_span,
            )

    def get_rate_limit_type(self) -> Literal["output", "input", "total"]:
        from litellm.proxy.proxy_server import general_settings

        specified_rate_limit_type = general_settings.get(
            "token_rate_limit_type", "output"
        )
        if not specified_rate_limit_type or specified_rate_limit_type not in [
            "output",
            "input",
            "total",
        ]:
            return "total"  # default to total
        return specified_rate_limit_type

    async def async_log_success_event(self, kwargs, response_obj, start_time, end_time):
        """
        Update TPM usage on successful API calls by incrementing counters using pipeline
        """
        from litellm.litellm_core_utils.core_helpers import (
            _get_parent_otel_span_from_kwargs,
        )
        from litellm.proxy.common_utils.callback_utils import (
            get_metadata_variable_name_from_kwargs,
            get_model_group_from_litellm_kwargs,
        )
        from litellm.types.caching import RedisPipelineIncrementOperation
        from litellm.types.utils import ModelResponse, Usage

        rate_limit_type = self.get_rate_limit_type()

        litellm_parent_otel_span: Union[Span, None] = _get_parent_otel_span_from_kwargs(
            kwargs
        )
        try:
            verbose_proxy_logger.debug(
                "INSIDE parallel request limiter ASYNC SUCCESS LOGGING"
            )

            # Get metadata from kwargs
            litellm_metadata = kwargs["litellm_params"].get(
                get_metadata_variable_name_from_kwargs(kwargs), {}
            )
            if litellm_metadata is None:
                return
            user_api_key = litellm_metadata.get("user_api_key")
            user_api_key_user_id = litellm_metadata.get("user_api_key_user_id")
            user_api_key_team_id = litellm_metadata.get("user_api_key_team_id")
            user_api_key_end_user_id = kwargs.get("user") or litellm_metadata.get(
                "user_api_key_end_user_id"
            )
            model_group = get_model_group_from_litellm_kwargs(kwargs)

            # Get total tokens from response
            total_tokens = 0
            # spot fix for /responses api
            if isinstance(response_obj, ModelResponse) or isinstance(
                response_obj, BaseLiteLLMOpenAIResponseObject
            ):
                _usage = getattr(response_obj, "usage", None)
                if _usage and isinstance(_usage, Usage):
                    if rate_limit_type == "output":
                        total_tokens = _usage.completion_tokens
                    elif rate_limit_type == "input":
                        total_tokens = _usage.prompt_tokens
                    elif rate_limit_type == "total":
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
            # Team Member TPM
            if user_api_key_team_id and user_api_key_user_id:
                pipeline_operations.extend(
                    self._create_pipeline_operations(
                        key="team_member",
                        value=f"{user_api_key_team_id}:{user_api_key_user_id}",
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
                await self.async_increment_tokens_with_ttl_preservation(
                    pipeline_operations=pipeline_operations,
                    parent_otel_span=litellm_parent_otel_span,
                )

        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error in rate limit success event: {str(e)}"
            )

    async def async_log_failure_event(self, kwargs, response_obj, start_time, end_time):
        """
        Decrement max parallel requests counter for the API Key
        """
        from litellm.litellm_core_utils.core_helpers import (
            _get_parent_otel_span_from_kwargs,
        )
        from litellm.types.caching import RedisPipelineIncrementOperation

        try:
            litellm_parent_otel_span: Union[Span, None] = (
                _get_parent_otel_span_from_kwargs(kwargs)
            )
            litellm_metadata = kwargs["litellm_params"]["metadata"]
            user_api_key = (
                litellm_metadata.get("user_api_key") if litellm_metadata else None
            )
            pipeline_operations: List[RedisPipelineIncrementOperation] = []

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

            # Execute all increments in a single pipeline
            if pipeline_operations:
                await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                    increment_list=pipeline_operations,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                )
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error in rate limit failure event: {str(e)}"
            )

    async def async_post_call_success_hook(
        self, data: dict, user_api_key_dict: UserAPIKeyAuth, response
    ):
        """
        Post-call hook to update rate limit headers in the response.
        """
        try:
            from pydantic import BaseModel

            litellm_proxy_rate_limit_response = cast(
                Optional[RateLimitResponse],
                data.get("litellm_proxy_rate_limit_response", None),
            )

            if litellm_proxy_rate_limit_response is not None:
                # Update response headers
                if hasattr(response, "_hidden_params"):
                    _hidden_params = getattr(response, "_hidden_params")
                else:
                    _hidden_params = None

                if _hidden_params is not None and (
                    isinstance(_hidden_params, BaseModel)
                    or isinstance(_hidden_params, dict)
                ):
                    if isinstance(_hidden_params, BaseModel):
                        _hidden_params = _hidden_params.model_dump()

                    _additional_headers = (
                        _hidden_params.get("additional_headers", {}) or {}
                    )

                    # Add rate limit headers
                    for status in litellm_proxy_rate_limit_response["statuses"]:
                        prefix = f"x-ratelimit-{status['descriptor_key']}"
                        _additional_headers[
                            f"{prefix}-remaining-{status['rate_limit_type']}"
                        ] = status["limit_remaining"]
                        _additional_headers[
                            f"{prefix}-limit-{status['rate_limit_type']}"
                        ] = status["current_limit"]

                    setattr(
                        response,
                        "_hidden_params",
                        {**_hidden_params, "additional_headers": _additional_headers},
                    )

        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error in rate limit post-call hook: {str(e)}"
            )
