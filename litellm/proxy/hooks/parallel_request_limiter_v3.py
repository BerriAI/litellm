"""
This is a rate limiter implementation based on a similar one by Envoy proxy.

This is currently in development and not yet ready for production.
"""

import binascii
import os
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
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
from litellm.constants import DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE
from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import get_model_rate_limit_from_metadata
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

# Redis cluster slot count
REDIS_CLUSTER_SLOTS = 16384
REDIS_NODE_HASHTAG_NAME = "all_keys"


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
    def __init__(
        self,
        internal_usage_cache: InternalUsageCache,
        time_provider: Optional[Callable[[], datetime]] = None,
    ):
        self.internal_usage_cache = internal_usage_cache
        self._time_provider = time_provider or datetime.now
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
        
        # Batch rate limiter (lazy loaded)
        self._batch_rate_limiter: Optional[Any] = None

    def _get_batch_rate_limiter(self) -> Optional[Any]:
        """Get or lazy-load the batch rate limiter."""
        if self._batch_rate_limiter is None:
            try:
                from litellm.proxy.hooks.batch_rate_limiter import (
                    _PROXY_BatchRateLimiter,
                )

                self._batch_rate_limiter = _PROXY_BatchRateLimiter(
                    internal_usage_cache=self.internal_usage_cache,
                    parallel_request_limiter=self,
                )
            except Exception as e:
                verbose_proxy_logger.debug(
                    f"Could not load batch rate limiter: {str(e)}"
                )
        return self._batch_rate_limiter

    def _get_current_time(self) -> datetime:
        """Return the current time for rate limiting calculations."""
        return self._time_provider()

    def _is_redis_cluster(self) -> bool:
        """
        Check if the dual cache is using Redis cluster.

        Returns:
            bool: True if using Redis cluster, False otherwise.
        """
        from litellm.caching.redis_cluster_cache import RedisClusterCache

        return (
            self.internal_usage_cache.dual_cache.redis_cache is not None
            and isinstance(
                self.internal_usage_cache.dual_cache.redis_cache, RedisClusterCache
            )
        )

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

    def keyslot_for_redis_cluster(self, key: str) -> int:
        """
        Compute the Redis Cluster slot for a given key.

        Simple implementation of `HASH_SLOT = CRC16(key) mod 16384`

        Read more about hash slots here: https://medium.com/@linz07m/how-hash-slots-power-data-distribution-in-redis-cluster-bc5b7e74ca7d

        Args:
            key (str): The Redis key.

        Returns:
            int: The slot number (0-16383).


        """
        # Handle hash tags: use substring between { and }
        start = key.find("{")
        if start != -1:
            end = key.find("}", start + 1)
            if end != -1 and end != start + 1:
                key = key[start + 1 : end]

        # Compute CRC16 and mod 16384
        crc = binascii.crc_hqx(key.encode("utf-8"), 0)
        return crc % REDIS_CLUSTER_SLOTS

    def _group_keys_by_hash_tag(self, keys: List[str]) -> Dict[str, List[str]]:
        """
        Group keys by their Redis hash tag to ensure cluster compatibility.

        For Redis clusters, uses slot calculation to group keys that belong to the same slot.
        For regular Redis, no grouping is needed - all keys can be processed together.
        """
        groups: Dict[str, List[str]] = {}

        # Use slot calculation for Redis clusters only
        if self._is_redis_cluster():
            for key in keys:
                slot = self.keyslot_for_redis_cluster(key)
                slot_key = f"slot_{slot}"

                if slot_key not in groups:
                    groups[slot_key] = []
                groups[slot_key].append(key)
        else:
            # For regular Redis, no grouping needed - process all keys together
            groups[REDIS_NODE_HASHTAG_NAME] = keys

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

        Args:
            descriptors: List of rate limit descriptors to check
            parent_otel_span: Optional OpenTelemetry span for tracing
            read_only: If True, only check limits without incrementing counters
        """

        current_time = self._get_current_time()
        now = current_time.timestamp()
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

        ## IF under limit in-memory, check Redis
        if read_only:
            # READ-ONLY MODE: Just read current values without incrementing
            cache_values = await self.internal_usage_cache.async_batch_get_cache(
                keys=keys_to_fetch,
                parent_otel_span=parent_otel_span,
                local_only=False,  # Check Redis too
            )

            # For keys that don't exist yet, set them to 0
            if cache_values is None:
                cache_values = []
                for _ in keys_to_fetch:
                    cache_values.append(str(now_int) if _.endswith(":window") else 0)
        elif self.batch_rate_limiter_script is not None:
            # NORMAL MODE: Increment counters in Redis
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
            # NORMAL MODE: In-memory sliding window (no Redis)
            cache_values = await self.in_memory_cache_sliding_window(
                keys=keys_to_fetch,
                now_int=now_int,
                window_size=self.window_size,
            )

        rate_limit_response = self.is_cache_list_over_limit(
            keys_to_fetch, cache_values, key_metadata
        )
        return rate_limit_response

    def create_organization_rate_limit_descriptor(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: Optional[str] = None
    ) -> List[RateLimitDescriptor]:
        descriptors: List[RateLimitDescriptor] = []

        # Global org rate limits
        if user_api_key_dict.org_id is not None and (
            user_api_key_dict.organization_rpm_limit is not None
            or user_api_key_dict.organization_tpm_limit is not None
        ):
            descriptors.append(
                RateLimitDescriptor(
                    key="organization",
                    value=user_api_key_dict.org_id,
                    rate_limit={
                        "requests_per_unit": user_api_key_dict.organization_rpm_limit,
                        "tokens_per_unit": user_api_key_dict.organization_tpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

        # Model specific org rate limits
        if (
            get_model_rate_limit_from_metadata(
                user_api_key_dict, "organization_metadata", "model_rpm_limit"
            )
            is not None
            or get_model_rate_limit_from_metadata(
                user_api_key_dict, "organization_metadata", "model_tpm_limit"
            )
            is not None
        ):
            _tpm_limit_for_team_model = (
                get_model_rate_limit_from_metadata(
                    user_api_key_dict, "organization_metadata", "model_tpm_limit"
                )
                or {}
            )
            _rpm_limit_for_team_model = (
                get_model_rate_limit_from_metadata(
                    user_api_key_dict, "organization_metadata", "model_rpm_limit"
                )
                or {}
            )

            should_check_rate_limit = False
            if requested_model in _tpm_limit_for_team_model:
                should_check_rate_limit = True
            elif requested_model in _rpm_limit_for_team_model:
                should_check_rate_limit = True

            if should_check_rate_limit:
                model_specific_tpm_limit = None
                model_specific_rpm_limit = None
                if requested_model in _tpm_limit_for_team_model:
                    model_specific_tpm_limit = _tpm_limit_for_team_model[
                        requested_model
                    ]
                if requested_model in _rpm_limit_for_team_model:
                    model_specific_rpm_limit = _rpm_limit_for_team_model[
                        requested_model
                    ]
                descriptors.append(
                    RateLimitDescriptor(
                        key="model_per_organization",
                        value=f"{user_api_key_dict.org_id}:{requested_model}",
                        rate_limit={
                            "requests_per_unit": model_specific_rpm_limit,
                            "tokens_per_unit": model_specific_tpm_limit,
                            "window_size": self.window_size,
                        },
                    )
                )

        return descriptors

    def _add_model_per_key_rate_limit_descriptor(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: Optional[str],
        descriptors: List[RateLimitDescriptor],
    ) -> None:
        """
        Add model-specific rate limit descriptor for API key if applicable.

        Args:
            user_api_key_dict: User API key authentication dictionary
            requested_model: The model being requested
            descriptors: List of rate limit descriptors to append to
        """
        from litellm.proxy.auth.auth_utils import (
            get_key_model_rpm_limit,
            get_key_model_tpm_limit,
        )

        if not requested_model:
            return

        _tpm_limit_for_key_model = get_key_model_tpm_limit(user_api_key_dict)
        _rpm_limit_for_key_model = get_key_model_rpm_limit(user_api_key_dict)

        if _tpm_limit_for_key_model is None and _rpm_limit_for_key_model is None:
            return

        _tpm_limit_for_key_model = _tpm_limit_for_key_model or {}
        _rpm_limit_for_key_model = _rpm_limit_for_key_model or {}

        # Check if model has any rate limits configured
        should_check_rate_limit = (
            requested_model in _tpm_limit_for_key_model
            or requested_model in _rpm_limit_for_key_model
        )

        if not should_check_rate_limit:
            return

        # Get model-specific limits
        model_specific_tpm_limit: Optional[int] = _tpm_limit_for_key_model.get(
            requested_model
        )
        model_specific_rpm_limit: Optional[int] = _rpm_limit_for_key_model.get(
            requested_model
        )

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

    def _should_enforce_rate_limit(
        self,
        limit_type: Optional[str],
        model_has_failures: bool,
    ) -> bool:
        """
        Determine if rate limit should be enforced based on limit type and model health.

        Args:
            limit_type: Type of rate limit ("dynamic", "guaranteed_throughput", "best_effort_throughput", or None)
            model_has_failures: Whether the model has recent failures

        Returns:
            True if rate limit should be enforced, False otherwise
        """
        if limit_type == "dynamic":
            # Dynamic mode: only enforce if model has failures
            return model_has_failures
        # All other modes (including None): always enforce
        return True

    def _get_enforced_limit(
        self,
        limit_value: Optional[int],
        limit_type: Optional[str],
        model_has_failures: bool,
    ) -> Optional[int]:
        """
        Get the rate limit value to enforce based on limit type and model health.

        Args:
            limit_value: The configured limit value
            limit_type: Type of rate limit ("dynamic", "guaranteed_throughput", "best_effort_throughput", or None)
            model_has_failures: Whether the model has recent failures

        Returns:
            The limit value if it should be enforced, None otherwise
        """
        if limit_value is None:
            return None

        if self._should_enforce_rate_limit(
            limit_type=limit_type,
            model_has_failures=model_has_failures,
        ):
            return limit_value

        return None

    def _is_dynamic_rate_limiting_enabled(
        self,
        rpm_limit_type: Optional[str],
        tpm_limit_type: Optional[str],
    ) -> bool:
        """
        Check if dynamic rate limiting is enabled for either RPM or TPM.

        Args:
            rpm_limit_type: RPM rate limit type
            tpm_limit_type: TPM rate limit type

        Returns:
            True if dynamic mode is enabled for either limit type
        """
        return rpm_limit_type == "dynamic" or tpm_limit_type == "dynamic"

    def _create_rate_limit_descriptors(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        rpm_limit_type: Optional[str],
        tpm_limit_type: Optional[str],
        model_has_failures: bool,
    ) -> List[RateLimitDescriptor]:
        """
        Create all rate limit descriptors for the request.

        Returns list of descriptors for API key, user, team, team member, end user, and model-specific limits.
        """
        from litellm.proxy.auth.auth_utils import (
            get_team_model_rpm_limit,
            get_team_model_tpm_limit,
        )

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
                        "requests_per_unit": self._get_enforced_limit(
                            limit_value=user_api_key_dict.rpm_limit,
                            limit_type=rpm_limit_type,
                            model_has_failures=model_has_failures,
                        ),
                        "tokens_per_unit": self._get_enforced_limit(
                            limit_value=user_api_key_dict.tpm_limit,
                            limit_type=tpm_limit_type,
                            model_has_failures=model_has_failures,
                        ),
                        "max_parallel_requests": user_api_key_dict.max_parallel_requests,
                        "window_size": self.window_size,
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
        self._add_model_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            requested_model=requested_model,
            descriptors=descriptors,
        )

        if (
            get_team_model_rpm_limit(user_api_key_dict) is not None
            or get_team_model_tpm_limit(user_api_key_dict) is not None
        ):
            _tpm_limit_for_team_model = (
                get_team_model_tpm_limit(user_api_key_dict) or {}
            )
            _rpm_limit_for_team_model = (
                get_team_model_rpm_limit(user_api_key_dict) or {}
            )
            should_check_rate_limit = False
            if requested_model in _tpm_limit_for_team_model:
                should_check_rate_limit = True
            elif requested_model in _rpm_limit_for_team_model:
                should_check_rate_limit = True

            if should_check_rate_limit:
                model_specific_tpm_limit = None
                model_specific_rpm_limit = None
                if requested_model in _tpm_limit_for_team_model:
                    model_specific_tpm_limit = _tpm_limit_for_team_model[
                        requested_model
                    ]
                if requested_model in _rpm_limit_for_team_model:
                    model_specific_rpm_limit = _rpm_limit_for_team_model[
                        requested_model
                    ]
                descriptors.append(
                    RateLimitDescriptor(
                        key="model_per_team",
                        value=f"{user_api_key_dict.team_id}:{requested_model}",
                        rate_limit={
                            "requests_per_unit": model_specific_rpm_limit,
                            "tokens_per_unit": model_specific_tpm_limit,
                            "window_size": self.window_size,
                        },
                    )
                )

        return descriptors

    async def _check_model_has_recent_failures(
        self,
        model: str,
        parent_otel_span: Optional[Span] = None,
    ) -> bool:
        """
        Check if any deployment for this model has recent failures by using
        the router's existing failure tracking.

        Returns True if any deployment has failures in the current minute.
        """
        from litellm.proxy.proxy_server import llm_router
        from litellm.router_utils.router_callbacks.track_deployment_metrics import (
            get_deployment_failures_for_current_minute,
        )

        if llm_router is None:
            return False

        try:
            # Get all deployments for this model
            model_list = llm_router.get_model_list(model_name=model)
            if not model_list:
                return False

            # Check each deployment's failure count
            for deployment in model_list:
                deployment_id = deployment.get("model_info", {}).get("id")
                if not deployment_id:
                    continue

                # Use router's existing failure tracking
                failure_count = get_deployment_failures_for_current_minute(
                    litellm_router_instance=llm_router,
                    deployment_id=deployment_id,
                )

                if failure_count > DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE:
                    verbose_proxy_logger.debug(
                        f"[Dynamic Rate Limit] Deployment {deployment_id} has {failure_count} failures "
                        f"in current minute - enforcing rate limits for model {model}"
                    )
                    return True

            verbose_proxy_logger.debug(
                f"[Dynamic Rate Limit] No failures detected for model {model} - allowing dynamic exceeding"
            )
            return False

        except Exception as e:
            verbose_proxy_logger.debug(
                f"Error checking model failure status: {str(e)}, defaulting to enforce limits"
            )
            # Fail safe: enforce limits if we can't check
            return True
    
    def get_rate_limiter_for_call_type(self, call_type: str) -> Optional[Any]:
        """Get the rate limiter for the call type."""
        if call_type == "acreate_batch":
            batch_limiter = self._get_batch_rate_limiter()
            return batch_limiter
        return None

    def _add_team_model_rate_limit_descriptor_from_metadata(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: Optional[str],
        descriptors: List[RateLimitDescriptor],
    ) -> None:
        """Add team model rate limit descriptor from team_metadata if applicable."""
        if (
            get_model_rate_limit_from_metadata(
                user_api_key_dict, "team_metadata", "model_rpm_limit"
            )
            is not None
            or get_model_rate_limit_from_metadata(
                user_api_key_dict, "team_metadata", "model_tpm_limit"
            )
            is not None
        ):
            _tpm_limit_for_team_model = (
                get_model_rate_limit_from_metadata(
                    user_api_key_dict, "team_metadata", "model_tpm_limit"
                )
                or {}
            )
            _rpm_limit_for_team_model = (
                get_model_rate_limit_from_metadata(
                    user_api_key_dict, "team_metadata", "model_rpm_limit"
                )
                or {}
            )
            should_check_rate_limit = (
                requested_model in _tpm_limit_for_team_model
                or requested_model in _rpm_limit_for_team_model
            )

            if should_check_rate_limit and requested_model is not None:
                model_specific_tpm_limit = _tpm_limit_for_team_model.get(
                    requested_model
                )
                model_specific_rpm_limit = _rpm_limit_for_team_model.get(
                    requested_model
                )
                descriptors.append(
                    RateLimitDescriptor(
                        key="model_per_team",
                        value=f"{user_api_key_dict.team_id}:{requested_model}",
                        rate_limit={
                            "requests_per_unit": model_specific_rpm_limit,
                            "tokens_per_unit": model_specific_tpm_limit,
                            "window_size": self.window_size,
                        },
                    )
                )

    def _handle_rate_limit_error(
        self,
        response: RateLimitResponse,
        descriptors: List[RateLimitDescriptor],
    ) -> None:
        """Handle rate limit exceeded error by raising HTTPException."""
        for status in response["statuses"]:
            if status["code"] == "OVER_LIMIT":
                descriptor_key = status["descriptor_key"]
                matching_descriptor = next(
                    (desc for desc in descriptors if desc["key"] == descriptor_key),
                    None,
                )
                descriptor_value = (
                    matching_descriptor["value"]
                    if matching_descriptor is not None
                    else "unknown"
                )

                now = self._get_current_time().timestamp()
                reset_time = now + self.window_size
                reset_time_formatted = datetime.fromtimestamp(
                    reset_time
                ).strftime("%Y-%m-%d %H:%M:%S UTC")

                remaining_display = max(0, status["limit_remaining"])
                rate_limit_type = status["rate_limit_type"]
                current_limit = status["current_limit"]

                detail = (
                    f"Rate limit exceeded for {descriptor_key}: {descriptor_value}. "
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

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ):
        """
        Pre-call hook to check rate limits before making the API call.
        Supports dynamic rate limiting based on deployment health.
        """
        verbose_proxy_logger.debug("Inside Rate Limit Pre-Call Hook")

        #########################################################
        # Check if the call type has a specific rate limiter
        # eg. for Batch APIs we need to use the batch rate limiter to read the input file and count the tokens and requests
        #########################################################
        call_type_specific_rate_limiter = self.get_rate_limiter_for_call_type(call_type=call_type)
        if call_type_specific_rate_limiter:
            return await call_type_specific_rate_limiter.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type=call_type,
            )

        # Get rate limit types from metadata
        metadata = user_api_key_dict.metadata or {}
        rpm_limit_type = metadata.get("rpm_limit_type")
        tpm_limit_type = metadata.get("tpm_limit_type")

        # For dynamic mode, check if the model has recent failures
        model_has_failures = False
        requested_model = data.get("model", None)

        if (
            self._is_dynamic_rate_limiting_enabled(
                rpm_limit_type=rpm_limit_type,
                tpm_limit_type=tpm_limit_type,
            )
            and requested_model
        ):
            model_has_failures = await self._check_model_has_recent_failures(
                model=requested_model,
                parent_otel_span=user_api_key_dict.parent_otel_span,
            )

        # Create rate limit descriptors
        descriptors = self._create_rate_limit_descriptors(
            user_api_key_dict=user_api_key_dict,
            data=data,
            rpm_limit_type=rpm_limit_type,
            tpm_limit_type=tpm_limit_type,
            model_has_failures=model_has_failures,
        )

        # Add team model rate limits from team_metadata
        self._add_team_model_rate_limit_descriptor_from_metadata(
            user_api_key_dict=user_api_key_dict,
            requested_model=requested_model,
            descriptors=descriptors,
        )

        # Org Level Rate Limits
        descriptors.extend(
            self.create_organization_rate_limit_descriptor(
                user_api_key_dict, requested_model
            )
        )
        # Only check rate limits if we have descriptors with actual limits
        if descriptors:
            response = await self.should_rate_limit(
                descriptors=descriptors,
                parent_otel_span=user_api_key_dict.parent_otel_span,
            )

            if response["overall_code"] == "OVER_LIMIT":
                self._handle_rate_limit_error(
                    response=response,
                    descriptors=descriptors,
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
            user_api_key_organization_id = litellm_metadata.get(
                "user_api_key_organization_id"
            )
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
            if model_group and user_api_key_team_id:
                pipeline_operations.extend(
                    self._create_pipeline_operations(
                        key="model_per_team",
                        value=f"{user_api_key_team_id}:{model_group}",
                        rate_limit_type="tokens",
                        total_tokens=total_tokens,
                    )
                )

            if model_group and user_api_key_organization_id:
                pipeline_operations.extend(
                    self._create_pipeline_operations(
                        key="model_per_organization",
                        value=f"{user_api_key_organization_id}:{model_group}",
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
            litellm_parent_otel_span: Union[
                Span, None
            ] = _get_parent_otel_span_from_kwargs(kwargs)
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
