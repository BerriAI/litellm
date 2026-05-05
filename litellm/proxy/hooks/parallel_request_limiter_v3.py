"""
This is a rate limiter implementation based on a similar one by Envoy proxy.

This is currently in development and not yet ready for production.
"""

import asyncio
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
    Set,
    Tuple,
    TypedDict,
    Union,
    cast,
)

from fastapi import HTTPException

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.constants import DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import get_model_rate_limit_from_metadata
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject
from litellm.types.utils import ModelResponse, Usage

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
        -- This happens when window_key exists but counter_key doesn't (e.g., tokens key
        -- created after requests key when both share the same window_key)
        local current_ttl = redis.call('TTL', counter_key)
        if current_ttl == -1 then
            redis.call('EXPIRE', counter_key, window_size)
        end
        table.insert(results, window_start) -- window_start
        table.insert(results, counter) -- counter
    end
end

return results
"""

CHECK_AND_INCREMENT_BY_N_SCRIPT = """
-- Atomic check-and-increment-by-N across one or more descriptors.
-- All-or-nothing: if any descriptor would exceed its limit, no counter is
-- modified.
--
-- Uses Redis server time (`redis.call('TIME')`) instead of a client-supplied
-- timestamp so that window resets are deterministic across replicas with
-- skewed wall-clocks. This prevents a clock-skew-induced reopening of the
-- TOCTOU window across multi-replica deployments.
--
-- KEYS layout: pairs of (window_key, counter_key), one pair per descriptor.
-- ARGV layout: per-descriptor 4-tuple, starting at ARGV[1]:
--     ARGV[(i-1)*4 + 1] = limit
--     ARGV[(i-1)*4 + 2] = increment
--     ARGV[(i-1)*4 + 3] = ttl_seconds (counter TTL when window resets)
--     ARGV[(i-1)*4 + 4] = window_size_seconds (sliding-window length)
--
-- Return on success: { 0, new_counter_1, new_counter_2, ... }
-- Return on over-limit: { 1, descriptor_index, current_counter, limit }
local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1])
local descriptor_count = #KEYS / 2

-- Pass 1: read state, validate. Abort without writing if any over limit.
local descriptor_state = {}
for i = 1, descriptor_count do
    local window_key = KEYS[(i - 1) * 2 + 1]
    local counter_key = KEYS[(i - 1) * 2 + 2]
    local arg_base = (i - 1) * 4 + 1
    local limit = tonumber(ARGV[arg_base])
    local increment = tonumber(ARGV[arg_base + 1])
    local window_size = tonumber(ARGV[arg_base + 3])

    local window_start = redis.call('GET', window_key)
    local window_expired = (not window_start) or
        ((now - tonumber(window_start)) >= window_size)

    local current_counter
    if window_expired then
        current_counter = 0
    else
        current_counter = tonumber(redis.call('GET', counter_key) or 0)
    end

    if current_counter + increment > limit then
        return { 1, i, current_counter, limit }
    end

    descriptor_state[i] = { window_expired, current_counter }
end

-- Pass 2: all checks passed. Apply increments.
local results = { 0 }
for i = 1, descriptor_count do
    local window_key = KEYS[(i - 1) * 2 + 1]
    local counter_key = KEYS[(i - 1) * 2 + 2]
    local arg_base = (i - 1) * 4 + 1
    local increment = tonumber(ARGV[arg_base + 1])
    local ttl = tonumber(ARGV[arg_base + 2])
    local window_size = tonumber(ARGV[arg_base + 3])

    local window_expired = descriptor_state[i][1]

    if window_expired then
        redis.call('SET', window_key, tostring(now))
        redis.call('SET', counter_key, increment)
        redis.call('EXPIRE', window_key, window_size)
        if ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(results, increment)
    else
        local new_counter = redis.call('INCRBY', counter_key, increment)
        local current_ttl = redis.call('TTL', counter_key)
        if current_ttl == -1 and ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(results, new_counter)
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

# TPM token reservation tuning constants.
# When max_tokens is not specified in the request we still need to reserve
# *some* output budget; these define that fallback estimate.
DEFAULT_MAX_TOKENS_ESTIMATE = 4096
DEFAULT_CHARS_PER_TOKEN = 4
# Stash for the reserved-token count on the request data dict so success/
# failure callbacks can reconcile against the upfront reservation.
TPM_RESERVED_TOKENS_KEY = "_litellm_tpm_reserved_tokens"
# Stash for the model identifier the reservation was charged against.
# Reconciliation must target the same key that was incremented at reservation
TPM_RESERVED_MODEL_KEY = "_litellm_tpm_reserved_model"
# Stash for the (scope_key, scope_value) pairs whose :tokens counter the
# upfront reservation incremented. Reconciliation applies the delta to these
# scopes only; scopes without a configured TPM limit were never charged at
# pre-call and must receive the full actual usage instead of the delta —
# otherwise their counters drift negative whenever actual < reserved.
TPM_RESERVED_SCOPES_KEY = "_litellm_tpm_reserved_scopes"
# Idempotency marker for the reservation refund path. Set when any failure
# callback releases the reservation so the next callback in the same flow
# (e.g. async_log_failure_event firing after async_post_call_failure_hook)
# does not double-refund.
TPM_RESERVATION_RELEASED_KEY = "_litellm_tpm_reservation_released"


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
            self.check_and_increment_by_n_script = (
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                    CHECK_AND_INCREMENT_BY_N_SCRIPT
                )
            )
        else:
            self.batch_rate_limiter_script = None
            self.token_increment_script = None
            self.check_and_increment_by_n_script = None

        self.window_size = int(os.getenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", 60))

        # Batch rate limiter (lazy loaded)
        self._batch_rate_limiter: Optional[Any] = None

        # Serializes multi-phase check+increment sequences (batch + dynamic
        # limiters) within this process to close the TOCTOU window between
        # read-only check and counter increment. Multi-replica deployments
        # additionally rely on Redis Lua atomicity for cross-process safety.
        #
        # Coarse granularity: this single lock serializes ALL atomic check+
        # increment operations across batch and dynamic limiters on this
        # instance. A slow batch input-file fetch (which happens upstream of
        # the lock) does not block here, but Redis Lua latency does. If
        # contention shows up under load (visible as p99 latency spikes
        # correlated with batch traffic), shard to a per-descriptor-key lock
        # via a `weakref.WeakValueDictionary[str, asyncio.Lock]`. Punted as a
        # follow-up because Lua dominates wall-time and the lock is held for
        # one round-trip.
        self._check_and_increment_lock = asyncio.Lock()

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

    def _estimate_tokens_for_request(
        self,
        data: dict,
        model: Optional[str] = None,
    ) -> int:
        """
        Estimate total tokens this request will consume so we can reserve them
        upfront (input + output budget):
        estimated = input_tokens + max_tokens.

        Supports chat (messages), completions (prompt), and embeddings (input).
        """
        messages = data.get("messages")
        prompt = data.get("prompt")
        input_text = data.get("input")  # embeddings

        match (messages, prompt, input_text):
            case (messages, _, _) if messages:
                total_chars = len(get_str_from_messages(messages))
            case (_, str() as p, _):
                total_chars = len(p)
            case (_, list() as p, _):
                total_chars = sum(len(str(item)) for item in p)
            case (_, _, str() as t):
                total_chars = len(t)
            case (_, _, list() as t):
                total_chars = sum(len(str(item)) for item in t)
            case _:
                total_chars = 0

        estimated_input_tokens = (
            max(1, total_chars // DEFAULT_CHARS_PER_TOKEN) if total_chars > 0 else 0
        )

        explicit_max_tokens = data.get("max_tokens") or data.get(
            "max_completion_tokens"
        )

        match (explicit_max_tokens, input_text):
            case (mt, _) if mt is not None:
                max_tokens_estimate = int(mt)
            case (_, embeddings_input) if embeddings_input:
                # Embeddings have no output tokens
                max_tokens_estimate = 0
            case _ if total_chars == 0:
                # Fully contentless request (no messages, prompt, or input).
                # Don't apply the conservative output-budget floor here — it
                # would over-reserve and could push small TPM limits into a
                # false 429. The caller floors at 1 so backpressure still
                # applies once the counter is at limit.
                max_tokens_estimate = 0
            case _:
                # No max_tokens specified — reserve at least the input size with a
                # conservative floor so a stream of small concurrent requests can't
                # collectively bypass the limit.
                max_tokens_estimate = max(
                    estimated_input_tokens,
                    DEFAULT_MAX_TOKENS_ESTIMATE // 4,
                )

        total_estimated = estimated_input_tokens + max_tokens_estimate

        verbose_proxy_logger.debug(
            f"TPM reservation estimate: input={estimated_input_tokens}, "
            f"max_tokens={max_tokens_estimate} (explicit={explicit_max_tokens is not None}), "
            f"total={total_estimated}"
        )

        return total_estimated

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
        skip_tpm_check: bool = False,
    ) -> RateLimitResponse:
        """
        Check if any of the rate limit descriptors should be rate limited.
        Returns a RateLimitResponse with the overall code and status for each descriptor.
        Uses batch operations for Redis to improve performance.

        Args:
            descriptors: List of rate limit descriptors to check
            parent_otel_span: Optional OpenTelemetry span for tracing
            read_only: If True, only check limits without incrementing counters
            skip_tpm_check: If True, ignore each descriptor's ``tokens_per_unit``
                — the :tokens counter is neither read nor incremented by this
                pass. Callers that handle TPM via the atomic
                ``reserve_tpm_tokens`` reservation path should set this to
                avoid the +1-per-key Lua / in-memory increment double-charging
                the tokens counter.
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
            tokens_limit = None if skip_tpm_check else rate_limit.get("tokens_per_unit")
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

    async def atomic_check_and_increment_by_n(
        self,
        descriptors: List[RateLimitDescriptor],
        increments: List[Dict[Literal["requests", "tokens"], int]],
        parent_otel_span: Optional[Span] = None,
    ) -> RateLimitResponse:
        """
        Atomic check-and-increment-by-N across one or more descriptors.

        All-or-nothing: if any descriptor would exceed its limit, no counter is
        modified and the response carries `overall_code = "OVER_LIMIT"` with
        the offending descriptor's status. Closes the TOCTOU window between
        read and increment in both single-process and multi-process (Redis)
        deployments.

        Cluster-safety: each descriptor's keys all share a `{key:value}` hash
        tag, so the Redis Lua path issues one Lua call per descriptor — every
        call's keys co-locate on a single Redis Cluster slot, avoiding
        CROSSSLOT errors. Cross-descriptor atomicity is preserved via
        refund-on-rollback: if descriptor i is OVER_LIMIT, descriptors 0..i-1
        get a direct INCRBY refund (refunds need no atomicity guarantee).

        Args:
            descriptors: rate-limit descriptors to check
            increments: per-descriptor increment amounts, indexed parallel to
                `descriptors`. Each entry is `{"requests": int, "tokens": int}`
                — values default to 0 when a descriptor has no matching limit.

        Returns:
            RateLimitResponse with one status per (descriptor, rate_limit_type)
            counter, mirroring `should_rate_limit`'s shape.
        """
        if len(descriptors) != len(increments):
            raise ValueError(
                "atomic_check_and_increment_by_n: descriptors and increments "
                "must have the same length"
            )

        # Build per-descriptor (keys, args, meta) groups. All keys within a
        # group share the descriptor's {key:value} hash tag, so a single Lua
        # call per group never triggers CROSSSLOT on Redis Cluster.
        descriptor_groups: List[Tuple[List[str], List[Any], List[Dict[str, Any]]]] = []
        for descriptor, increment_amounts in zip(descriptors, increments):
            keys, args, meta = self._build_descriptor_atomic_payload(
                descriptor=descriptor,
                increment_amounts=increment_amounts,
            )
            if keys:
                descriptor_groups.append((keys, args, meta))

        if not descriptor_groups:
            return RateLimitResponse(overall_code="OK", statuses=[])

        # Multi-process atomicity via Redis Lua, per descriptor for slot
        # co-location. Single-process atomicity falls back to the
        # asyncio.Lock + in-memory sliding window below — there are no
        # cluster slot concerns locally, so we keep the batched 2-phase
        # critical section for true cross-descriptor atomicity.
        if self.check_and_increment_by_n_script is not None:
            return await self._atomic_lua_per_descriptor(
                descriptor_groups=descriptor_groups,
                parent_otel_span=parent_otel_span,
            )

        flat_meta: List[Dict[str, Any]] = [
            m for _keys, _args, group_meta in descriptor_groups for m in group_meta
        ]
        async with self._check_and_increment_lock:
            return await self._atomic_check_and_increment_in_memory(
                per_counter_meta=flat_meta,
                parent_otel_span=parent_otel_span,
            )

    def _build_descriptor_atomic_payload(
        self,
        descriptor: RateLimitDescriptor,
        increment_amounts: Dict[Literal["requests", "tokens"], int],
    ) -> Tuple[List[str], List[Any], List[Dict[str, Any]]]:
        """
        Build (KEYS, ARGV, per-counter meta) for a single descriptor's Lua
        call. All keys returned share the descriptor's {key:value} hash tag.
        """
        descriptor_key = descriptor["key"]
        descriptor_value = descriptor["value"]
        rate_limit: RateLimitDescriptorRateLimitObject = (
            descriptor.get("rate_limit") or RateLimitDescriptorRateLimitObject()
        )
        window_size = rate_limit.get("window_size") or self.window_size
        window_key = f"{{{descriptor_key}:{descriptor_value}}}:window"

        keys: List[str] = []
        args: List[Any] = []
        meta: List[Dict[str, Any]] = []

        for rate_limit_type in ("requests", "tokens"):
            rlt: Literal["requests", "tokens"] = cast(
                Literal["requests", "tokens"], rate_limit_type
            )
            if rlt == "requests":
                limit_value = rate_limit.get("requests_per_unit")
                inc_amount = int(increment_amounts.get("requests", 0) or 0)
            else:
                limit_value = rate_limit.get("tokens_per_unit")
                inc_amount = int(increment_amounts.get("tokens", 0) or 0)
            if limit_value is None or inc_amount <= 0:
                continue
            counter_key = self.create_rate_limit_keys(
                descriptor_key, descriptor_value, rlt
            )
            # Counter-key TTL and window_size are conceptually distinct
            # ("how long the counter Redis key lives" vs "how long the
            # sliding window is"). Kept as separate values so a future
            # custom-TTL descriptor doesn't reintroduce a silent expiry bug.
            ttl_seconds = int(window_size)
            window_size_seconds = int(window_size)
            keys.extend([window_key, counter_key])
            # 4-tuple matches the Lua ARGV layout:
            #   [limit, increment, ttl_seconds, window_size_seconds].
            args.extend(
                [int(limit_value), inc_amount, ttl_seconds, window_size_seconds]
            )
            meta.append(
                {
                    "descriptor_key": descriptor_key,
                    "current_limit": int(limit_value),
                    "rate_limit_type": rlt,
                    "window_key": window_key,
                    "counter_key": counter_key,
                    "increment": inc_amount,
                    "ttl": ttl_seconds,
                    "window_size": window_size_seconds,
                }
            )
        return keys, args, meta

    async def _atomic_lua_per_descriptor(
        self,
        descriptor_groups: List[Tuple[List[str], List[Any], List[Dict[str, Any]]]],
        parent_otel_span: Optional[Span] = None,
    ) -> RateLimitResponse:
        """
        Run Lua check-and-increment one descriptor at a time so each call's
        keys co-locate on a single Redis Cluster slot. On OVER_LIMIT for
        descriptor i, refund descriptors 0..i-1's increments. On Lua failure
        mid-loop, refund applied increments and fall back to in-memory.
        """
        applied: List[List[Dict[str, Any]]] = []
        statuses: List[RateLimitStatus] = []

        for _idx, (keys, args, meta) in enumerate(descriptor_groups):
            try:
                raw = await self.check_and_increment_by_n_script(
                    keys=keys,
                    args=args,
                )
            except Exception as e:
                # Lua failure (timeout, OOM, network partition) leaves Redis
                # state ambiguous. Refund any prior groups so Redis returns
                # to its pre-call state, then fall back to in-memory for the
                # whole call (counters there are independent of Redis).
                verbose_proxy_logger.error(
                    f"atomic_check_and_increment_by_n: Redis Lua execution "
                    f"failed ({type(e).__name__}: {e}). Refunding "
                    f"{len(applied)} prior descriptors and falling back to "
                    f"in-memory enforcement — counters will diverge from "
                    f"Redis until window expires (window_size="
                    f"{self.window_size}s)."
                )
                await self._refund_applied_descriptor_groups(applied)
                flat_meta: List[Dict[str, Any]] = [
                    m for _k, _a, group_meta in descriptor_groups for m in group_meta
                ]
                async with self._check_and_increment_lock:
                    return await self._atomic_check_and_increment_in_memory(
                        per_counter_meta=flat_meta,
                        parent_otel_span=parent_otel_span,
                    )

            response = self._build_atomic_response(raw, meta)
            if response["overall_code"] == "OVER_LIMIT":
                await self._refund_applied_descriptor_groups(applied)
                return response
            applied.append(meta)
            statuses.extend(response["statuses"])

        return RateLimitResponse(overall_code="OK", statuses=statuses)

    async def _refund_applied_descriptor_groups(
        self,
        applied: List[List[Dict[str, Any]]],
    ) -> None:
        """
        Decrement counters for descriptor groups already applied via Lua.
        Best-effort: refund failures are logged but not raised — the original
        OVER_LIMIT / fallback decision is what matters to the caller.
        """
        if not applied:
            return
        redis_cache = self.internal_usage_cache.dual_cache.redis_cache
        if redis_cache is None:
            return
        for group_meta in applied:
            for entry in group_meta:
                try:
                    await redis_cache.async_increment(
                        key=entry["counter_key"],
                        value=-entry["increment"],
                    )
                except Exception as e:
                    verbose_proxy_logger.warning(
                        f"Failed to refund {entry['counter_key']} on "
                        f"cross-descriptor rollback: {e}"
                    )

    def _build_atomic_response(
        self,
        raw: List[Any],
        per_counter_meta: List[Dict[str, Any]],
    ) -> RateLimitResponse:
        """Convert Lua script return value to RateLimitResponse.

        Indexing invariant: `per_counter_meta` and `KEYS` are parallel-indexed
        at the COUNTER level, not the descriptor level. A descriptor with both
        RPM and TPM limits emits two `(window_key, counter_key)` pairs and
        two meta entries — one per counter. The Lua script's loop variable
        `i` therefore enumerates counters, and the over-limit return tuple
        `{1, i, ...}` carries a counter index that maps directly to
        `per_counter_meta[i - 1]`. Keep these arrays parallel at the counter
        level when modifying this code.
        """
        if not raw:
            return RateLimitResponse(overall_code="OK", statuses=[])

        status_code = int(raw[0])
        if status_code == 1:
            # Over limit: { 1, counter_index (1-based), current_counter, limit }
            descriptor_index = int(raw[1]) - 1
            current_counter = int(raw[2])
            limit = int(raw[3])
            meta = per_counter_meta[descriptor_index]
            return RateLimitResponse(
                overall_code="OVER_LIMIT",
                statuses=[
                    RateLimitStatus(
                        code="OVER_LIMIT",
                        current_limit=limit,
                        limit_remaining=max(0, limit - current_counter),
                        rate_limit_type=meta["rate_limit_type"],
                        descriptor_key=meta["descriptor_key"],
                    )
                ],
            )

        statuses: List[RateLimitStatus] = []
        for meta, new_counter in zip(per_counter_meta, raw[1:]):
            statuses.append(
                RateLimitStatus(
                    code="OK",
                    current_limit=meta["current_limit"],
                    limit_remaining=max(0, meta["current_limit"] - int(new_counter)),
                    rate_limit_type=meta["rate_limit_type"],
                    descriptor_key=meta["descriptor_key"],
                )
            )
        return RateLimitResponse(overall_code="OK", statuses=statuses)

    async def _atomic_check_and_increment_in_memory(
        self,
        per_counter_meta: List[Dict[str, Any]],
        parent_otel_span: Optional[Span] = None,
    ) -> RateLimitResponse:
        """In-memory all-or-nothing check-and-increment. Caller holds lock.

        Reads/writes the LOCAL DualCache (`local_only=True`) — note this is
        a different store from Redis. When this fallback fires after a Lua
        failure, in-memory counters will diverge from Redis until each key's
        window expires (TTL bounds divergence).
        """
        # Use a single 'now' for the duration of this critical section so all
        # descriptors evaluate window expiry consistently.
        now_int = int(self._get_current_time().timestamp())

        # Pass 1: read state, validate.
        descriptor_state: List[Dict[str, Any]] = []
        for meta in per_counter_meta:
            window_size = meta["window_size"]
            window_start = await self.internal_usage_cache.async_get_cache(
                key=meta["window_key"],
                litellm_parent_otel_span=parent_otel_span,
                local_only=True,
            )
            window_expired = (
                window_start is None or (now_int - int(window_start)) >= window_size
            )
            current_counter = (
                0
                if window_expired
                else int(
                    await self.internal_usage_cache.async_get_cache(
                        key=meta["counter_key"],
                        litellm_parent_otel_span=parent_otel_span,
                        local_only=True,
                    )
                    or 0
                )
            )
            if current_counter + meta["increment"] > meta["current_limit"]:
                return RateLimitResponse(
                    overall_code="OVER_LIMIT",
                    statuses=[
                        RateLimitStatus(
                            code="OVER_LIMIT",
                            current_limit=meta["current_limit"],
                            limit_remaining=max(
                                0, meta["current_limit"] - current_counter
                            ),
                            rate_limit_type=meta["rate_limit_type"],
                            descriptor_key=meta["descriptor_key"],
                        )
                    ],
                )
            descriptor_state.append(
                {"window_expired": window_expired, "current": current_counter}
            )

        # Pass 2: apply increments.
        statuses: List[RateLimitStatus] = []
        for meta, state in zip(per_counter_meta, descriptor_state):
            new_counter = (
                meta["increment"]
                if state["window_expired"]
                else state["current"] + meta["increment"]
            )
            if state["window_expired"]:
                await self.internal_usage_cache.async_set_cache(
                    key=meta["window_key"],
                    value=str(now_int),
                    ttl=meta["window_size"],
                    litellm_parent_otel_span=parent_otel_span,
                    local_only=True,
                )
            await self.internal_usage_cache.async_set_cache(
                key=meta["counter_key"],
                value=new_counter,
                ttl=meta["ttl"],
                litellm_parent_otel_span=parent_otel_span,
                local_only=True,
            )
            statuses.append(
                RateLimitStatus(
                    code="OK",
                    current_limit=meta["current_limit"],
                    limit_remaining=max(0, meta["current_limit"] - new_counter),
                    rate_limit_type=meta["rate_limit_type"],
                    descriptor_key=meta["descriptor_key"],
                )
            )
        return RateLimitResponse(overall_code="OK", statuses=statuses)

    async def reserve_tpm_tokens(
        self,
        descriptors: List[RateLimitDescriptor],
        estimated_tokens: int,
        parent_otel_span: Optional[Span] = None,
    ) -> RateLimitResponse:
        """
        Reserve ``estimated_tokens`` against every TPM-bearing descriptor
        BEFORE the upstream call, so concurrent requests cannot all observe
        "under limit" before any of them increments the counter.

        Thin wrapper around ``atomic_check_and_increment_by_n``: builds a
        TPM-only descriptor/increment list and delegates the all-or-nothing
        atomicity (Lua on Redis, asyncio-locked DualCache otherwise) to the
        shared primitive.
        """
        tpm_descriptors: List[RateLimitDescriptor] = [
            d
            for d in descriptors
            if (d.get("rate_limit") or {}).get("tokens_per_unit") is not None
        ]
        if not tpm_descriptors:
            return RateLimitResponse(overall_code="OK", statuses=[])

        increments: List[Dict[Literal["requests", "tokens"], int]] = [
            {"tokens": estimated_tokens} for _ in tpm_descriptors
        ]
        return await self.atomic_check_and_increment_by_n(
            descriptors=tpm_descriptors,
            increments=increments,
            parent_otel_span=parent_otel_span,
        )

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

        _tpm_limit_for_key_model = get_key_model_tpm_limit(
            user_api_key_dict, model_name=requested_model
        )
        _rpm_limit_for_key_model = get_key_model_rpm_limit(
            user_api_key_dict, model_name=requested_model
        )

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

    def _get_agent_from_registry(self, agent_id: str) -> Optional[Any]:
        """Look up an agent from the in-memory registry by ID."""
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        return global_agent_registry.get_agent_by_id(agent_id=agent_id)

    def _get_resolved_agent_id(
        self, user_api_key_dict: UserAPIKeyAuth, data: dict
    ) -> Optional[str]:
        """
        Resolve the agent_id from either the API key or request metadata.
        Key-level agent_id takes precedence over metadata/header-supplied agent_id.
        """
        key_agent_id = getattr(user_api_key_dict, "agent_id", None)
        if key_agent_id:
            return key_agent_id
        metadata = data.get("metadata") or {}
        return metadata.get("agent_id")

    def _get_session_id_from_data(self, data: dict) -> Optional[str]:
        """Extract session_id from request metadata or litellm_session_id."""
        session_id = data.get("litellm_session_id")
        if session_id:
            return str(session_id)
        metadata = data.get("metadata") or {}
        session_id = metadata.get("session_id")
        if session_id:
            return str(session_id)
        litellm_metadata = data.get("litellm_metadata") or {}
        session_id = litellm_metadata.get("session_id")
        if session_id:
            return str(session_id)
        return None

    def _create_agent_rate_limit_descriptors(
        self,
        agent_id: str,
        data: dict,
    ) -> List[RateLimitDescriptor]:
        """
        Create rate limit descriptors for agent-level and session-level limits.

        Agent-level: caps total RPM/TPM across all sessions for a given agent.
        Session-level: caps RPM/TPM within a single session (identified by session_id).
        """
        descriptors: List[RateLimitDescriptor] = []

        agent = self._get_agent_from_registry(agent_id)
        if agent is None:
            return descriptors

        agent_rpm = getattr(agent, "rpm_limit", None)
        agent_tpm = getattr(agent, "tpm_limit", None)
        if agent_rpm is not None or agent_tpm is not None:
            descriptors.append(
                RateLimitDescriptor(
                    key="agent",
                    value=agent_id,
                    rate_limit={
                        "requests_per_unit": agent_rpm,
                        "tokens_per_unit": agent_tpm,
                        "window_size": self.window_size,
                    },
                )
            )

        session_rpm = getattr(agent, "session_rpm_limit", None)
        session_tpm = getattr(agent, "session_tpm_limit", None)
        if session_rpm is not None or session_tpm is not None:
            session_id = self._get_session_id_from_data(data)
            if session_id is not None:
                descriptors.append(
                    RateLimitDescriptor(
                        key="agent_session",
                        value=f"{agent_id}:{session_id}",
                        rate_limit={
                            "requests_per_unit": session_rpm,
                            "tokens_per_unit": session_tpm,
                            "window_size": self.window_size,
                        },
                    )
                )

        return descriptors

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

        Returns list of descriptors for API key, user, team, team member, end user,
        model-specific, agent, and agent-session limits.
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

        # Agent-level and session-level rate limits
        resolved_agent_id = self._get_resolved_agent_id(user_api_key_dict, data)

        if resolved_agent_id:
            descriptors.extend(
                self._create_agent_rate_limit_descriptors(
                    agent_id=resolved_agent_id,
                    data=data,
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

    def _add_project_model_rate_limit_descriptor_from_metadata(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: Optional[str],
        descriptors: List[RateLimitDescriptor],
    ) -> None:
        """Add project model rate limit descriptor from project_metadata if applicable."""
        if (
            get_model_rate_limit_from_metadata(
                user_api_key_dict, "project_metadata", "model_rpm_limit"
            )
            is not None
            or get_model_rate_limit_from_metadata(
                user_api_key_dict, "project_metadata", "model_tpm_limit"
            )
            is not None
        ):
            _tpm_limit_for_project_model = (
                get_model_rate_limit_from_metadata(
                    user_api_key_dict, "project_metadata", "model_tpm_limit"
                )
                or {}
            )
            _rpm_limit_for_project_model = (
                get_model_rate_limit_from_metadata(
                    user_api_key_dict, "project_metadata", "model_rpm_limit"
                )
                or {}
            )
            should_check_rate_limit = (
                requested_model in _tpm_limit_for_project_model
                or requested_model in _rpm_limit_for_project_model
            )

            if should_check_rate_limit and requested_model is not None:
                model_specific_tpm_limit = _tpm_limit_for_project_model.get(
                    requested_model
                )
                model_specific_rpm_limit = _rpm_limit_for_project_model.get(
                    requested_model
                )
                descriptors.append(
                    RateLimitDescriptor(
                        key="model_per_project",
                        value=f"{user_api_key_dict.project_id}:{requested_model}",
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
                reset_time_formatted = datetime.fromtimestamp(reset_time).strftime(
                    "%Y-%m-%d %H:%M:%S UTC"
                )

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
        call_type_specific_rate_limiter = self.get_rate_limiter_for_call_type(
            call_type=call_type
        )
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

        # Project Level Rate Limits
        self._add_project_model_rate_limit_descriptor_from_metadata(
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
            # First pass: RPM and max_parallel_requests sliding-window check.
            # `skip_tpm_check=True` tells should_rate_limit to ignore each
            # descriptor's tokens_per_unit so its +1-per-key Lua / in-memory
            # increment never touches the :tokens counters — those are owned
            # exclusively by the atomic reserve_tpm_tokens path below. Without
            # this, every concurrent in-flight request would pre-inflate the
            # :tokens counter by 1, shrinking the effective TPM budget by N
            # and causing false-positive 429s under bursts.
            response = await self.should_rate_limit(
                descriptors=descriptors,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                skip_tpm_check=True,
            )

            if response["overall_code"] == "OVER_LIMIT":
                self._handle_rate_limit_error(
                    response=response,
                    descriptors=descriptors,
                )
            else:
                # add descriptors to request headers
                data["litellm_proxy_rate_limit_response"] = response

            # ----------------------------------------------------------------
            # TPM token reservation
            # Atomically reserve estimated tokens upfront so concurrent
            # requests cannot all observe "under limit" before any of them
            # has incremented the counter. atomic_check_and_increment_by_n
            # uses Redis Lua when available and falls back to an asyncio-locked
            # in-memory check otherwise — single-worker protection still holds
            # even without Redis.
            # ----------------------------------------------------------------
            has_tpm_limits = any(
                (d.get("rate_limit") or {}).get("tokens_per_unit") is not None
                for d in descriptors
            )

            if has_tpm_limits:
                # Floor at 1 token so contentless requests (/responses,
                # tool-call continuations, empty messages) still flow
                # through the atomic counter and get backpressure when at
                # limit. Without this floor, N concurrent contentless
                # requests would all pass pre-call with no enforcement.
                # Post-call reconciliation refunds the over-reservation
                # delta when actual usage comes in below the floor.
                estimated_tokens = max(
                    self._estimate_tokens_for_request(
                        data=data,
                        model=requested_model,
                    ),
                    1,
                )

                tpm_response = await self.reserve_tpm_tokens(
                    descriptors=descriptors,
                    estimated_tokens=estimated_tokens,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )

                if tpm_response["overall_code"] == "OVER_LIMIT":
                    self._handle_rate_limit_error(
                        response=tpm_response,
                        descriptors=descriptors,
                    )
                else:
                    data["_litellm_rate_limit_descriptors"] = descriptors
                    # Capture the exact (key, value) scopes the reservation
                    # incremented so post-call reconciliation only applies
                    # the (actual - reserved) delta to those — unreserved
                    # scopes get charged the full actual usage instead.
                    reserved_scopes: List[Tuple[str, str]] = [
                        (d["key"], d["value"])
                        for d in descriptors
                        if (d.get("rate_limit") or {}).get("tokens_per_unit")
                        is not None
                    ]
                    self._stash_reservation_in_data(
                        data=data,
                        estimated_tokens=estimated_tokens,
                        reserved_model=requested_model,
                        reserved_scopes=reserved_scopes,
                    )

                    # Merge TPM statuses into the stored rate-limit response
                    # so x-ratelimit-{key}-remaining-tokens / -limit-tokens
                    # headers reach the client. Without this, the RPM-only
                    # response from should_rate_limit (skip_tpm_check=True)
                    # silently drops all token headers.
                    stored_response = data.get("litellm_proxy_rate_limit_response")
                    if isinstance(stored_response, dict):
                        stored_response.setdefault("statuses", []).extend(
                            tpm_response["statuses"]
                        )
                    elif tpm_response["statuses"]:
                        data["litellm_proxy_rate_limit_response"] = tpm_response

                    verbose_proxy_logger.debug(
                        f"TPM tokens reserved: {estimated_tokens} for model {requested_model}"
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

    def _get_total_tokens_from_usage(
        self, usage: Optional[Any], rate_limit_type: Literal["output", "input", "total"]
    ) -> int:
        """
        Get total tokens from response usage for rate limiting.

        For 'input' and 'total' rate limit types, cached tokens are excluded
        because providers like AWS Bedrock don't count cached tokens toward
        rate limits. This aligns LiteLLM's TPM calculation with provider behavior.
        """
        total_tokens = 0
        cached_tokens = 0

        if usage:
            if isinstance(usage, Usage):
                if rate_limit_type == "output":
                    total_tokens = usage.completion_tokens or 0
                elif rate_limit_type == "input":
                    total_tokens = usage.prompt_tokens or 0
                elif rate_limit_type == "total":
                    total_tokens = usage.total_tokens or 0

                # Get cached tokens to exclude from input/total
                if rate_limit_type in ("input", "total"):
                    if (
                        hasattr(usage, "prompt_tokens_details")
                        and usage.prompt_tokens_details is not None
                    ):
                        cached_tokens = (
                            getattr(usage.prompt_tokens_details, "cached_tokens", 0)
                            or 0
                        )

            elif isinstance(usage, dict):
                # Responses API usage comes as a dict
                if rate_limit_type == "output":
                    total_tokens = usage.get("completion_tokens", 0) or 0
                elif rate_limit_type == "input":
                    total_tokens = usage.get("prompt_tokens", 0) or 0
                elif rate_limit_type == "total":
                    total_tokens = usage.get("total_tokens", 0) or 0

                # Get cached tokens from dict
                if rate_limit_type in ("input", "total"):
                    prompt_details = usage.get("prompt_tokens_details") or {}
                    if isinstance(prompt_details, dict):
                        cached_tokens = prompt_details.get("cached_tokens", 0) or 0

        # Subtract cached tokens for input/total (providers don't count them)
        if cached_tokens > 0:
            total_tokens = max(0, total_tokens - cached_tokens)

        return total_tokens

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
            "token_rate_limit_type", "total"
        )
        if specified_rate_limit_type not in [
            "output",
            "input",
            "total",
        ]:
            return "total"  # default to total
        return specified_rate_limit_type

    @staticmethod
    def _stash_reservation_in_data(
        data: Dict[str, Any],
        estimated_tokens: int,
        reserved_model: Optional[str],
        reserved_scopes: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """
        Persist the reservation amount, model, and reserved scopes into every
        channel a callback might read from: top-level kwargs (via ``**data``),
        request metadata, and litellm_metadata. Keeps reservation and
        reconciliation in sync.

        ``reserved_scopes`` is serialized as a list of [key, value] pairs so
        it round-trips through JSON-based metadata transports.
        """
        scopes_payload: Optional[List[List[str]]] = (
            [[k, v] for k, v in reserved_scopes] if reserved_scopes else None
        )

        data[TPM_RESERVED_TOKENS_KEY] = estimated_tokens
        if reserved_model:
            data[TPM_RESERVED_MODEL_KEY] = reserved_model
        if scopes_payload is not None:
            data[TPM_RESERVED_SCOPES_KEY] = scopes_payload

        for channel in ("metadata", "litellm_metadata"):
            existing = data.get(channel)
            if isinstance(existing, dict):
                existing[TPM_RESERVED_TOKENS_KEY] = estimated_tokens
                if reserved_model:
                    existing[TPM_RESERVED_MODEL_KEY] = reserved_model
                if scopes_payload is not None:
                    existing[TPM_RESERVED_SCOPES_KEY] = scopes_payload
            elif channel == "metadata":
                # Only auto-create ``metadata`` (preserves prior behavior);
                # ``litellm_metadata`` is set by the router and shouldn't be
                # conjured here.
                stash: Dict[str, Any] = {TPM_RESERVED_TOKENS_KEY: estimated_tokens}
                if reserved_model:
                    stash[TPM_RESERVED_MODEL_KEY] = reserved_model
                if scopes_payload is not None:
                    stash[TPM_RESERVED_SCOPES_KEY] = scopes_payload
                data[channel] = stash

    @staticmethod
    def _lookup_stashed_value(
        kwargs: Any,
        standard_logging_metadata: Optional[Dict[str, Any]],
        key: str,
    ) -> Any:
        """
        Resolve a stashed value from any of the channels the request data can
        flow through to a callback.

        Checks (in priority order):
          1. kwargs (top-level data fields propagate via **data)
          2. kwargs["litellm_params"]["metadata"] (request metadata channel)
          3. standard_logging_metadata (covers tests that mock the SLO directly)
        """
        candidate = kwargs.get(key) if isinstance(kwargs, dict) else None
        if candidate is None:
            litellm_params = (
                kwargs.get("litellm_params") if isinstance(kwargs, dict) else None
            )
            if isinstance(litellm_params, dict):
                lp_metadata = litellm_params.get("metadata")
                if isinstance(lp_metadata, dict):
                    candidate = lp_metadata.get(key)
        if candidate is None and isinstance(standard_logging_metadata, dict):
            candidate = standard_logging_metadata.get(key)
        return candidate

    @classmethod
    def _get_reserved_tokens_from_kwargs(
        cls,
        kwargs: Any,
        standard_logging_metadata: Optional[Dict[str, Any]] = None,
    ) -> int:
        candidate = cls._lookup_stashed_value(
            kwargs, standard_logging_metadata, TPM_RESERVED_TOKENS_KEY
        )
        try:
            return int(candidate or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _get_reserved_model_from_kwargs(
        cls,
        kwargs: Any,
        standard_logging_metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        """
        Resolve the model the upfront reservation was charged against. Used to
        target reconciliation at the same key that was incremented, regardless
        of whether the router later set a different ``model_group`` in
        ``litellm_params.metadata``.
        """
        candidate = cls._lookup_stashed_value(
            kwargs, standard_logging_metadata, TPM_RESERVED_MODEL_KEY
        )
        return candidate if isinstance(candidate, str) and candidate else None

    @classmethod
    def _get_reserved_scopes_from_kwargs(
        cls,
        kwargs: Any,
        standard_logging_metadata: Optional[Dict[str, Any]] = None,
    ) -> Set[Tuple[str, str]]:
        """
        Resolve the (scope_key, scope_value) pairs the upfront reservation
        actually charged. Reconciliation distinguishes these from
        unreserved scopes — applying the delta to reserved scopes (which
        already carry +reserved on the counter) and the full actual to
        unreserved ones (which were never charged).
        """
        candidate = cls._lookup_stashed_value(
            kwargs, standard_logging_metadata, TPM_RESERVED_SCOPES_KEY
        )
        if not isinstance(candidate, list):
            return set()
        scopes: Set[Tuple[str, str]] = set()
        for entry in candidate:
            if (
                isinstance(entry, (list, tuple))
                and len(entry) == 2
                and isinstance(entry[0], str)
                and isinstance(entry[1], str)
            ):
                scopes.add((entry[0], entry[1]))
        return scopes

    @classmethod
    def _is_reservation_released(
        cls,
        kwargs: Any,
        standard_logging_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """True if a prior callback already refunded this request's reservation."""
        return bool(
            cls._lookup_stashed_value(
                kwargs, standard_logging_metadata, TPM_RESERVATION_RELEASED_KEY
            )
        )

    @staticmethod
    def _mark_reservation_released(data: Any) -> None:
        """
        Stamp the released flag into every metadata channel a sibling
        callback might read from. async_post_call_failure_hook receives the
        request data dict; async_log_failure_event reads kwargs +
        standard_logging_object.metadata. Same dict identity across
        ``request_data["metadata"]`` and ``kwargs["litellm_params"]["metadata"]``
        means writes here propagate to the other hook.
        """
        if not isinstance(data, dict):
            return
        data[TPM_RESERVATION_RELEASED_KEY] = True
        for channel in ("metadata", "litellm_metadata"):
            existing = data.get(channel)
            if isinstance(existing, dict):
                existing[TPM_RESERVATION_RELEASED_KEY] = True
        litellm_params = data.get("litellm_params")
        if isinstance(litellm_params, dict):
            lp_metadata = litellm_params.get("metadata")
            if isinstance(lp_metadata, dict):
                lp_metadata[TPM_RESERVATION_RELEASED_KEY] = True
        slo = data.get("standard_logging_object")
        if isinstance(slo, dict):
            slo_meta = slo.get("metadata")
            if isinstance(slo_meta, dict):
                slo_meta[TPM_RESERVATION_RELEASED_KEY] = True

    def _collect_tpm_scope_targets(
        self,
        standard_logging_metadata: Dict[str, Any],
        kwargs: Any,
        model_group: Optional[str],
    ) -> List[Tuple[str, str]]:
        """
        Enumerate every (scope_key, scope_value) pair that *might* carry a
        TPM counter for this request — independent of whether each scope had
        a configured TPM limit at pre-call. Reservation awareness happens at
        the emitter; this helper just lists the candidate scopes so callers
        can split reserved-vs-unreserved.
        """
        user_api_key = standard_logging_metadata.get("user_api_key_hash")
        user_api_key_user_id = standard_logging_metadata.get("user_api_key_user_id")
        user_api_key_team_id = standard_logging_metadata.get("user_api_key_team_id")
        user_api_key_organization_id = standard_logging_metadata.get(
            "user_api_key_org_id"
        )
        user_api_key_project_id = standard_logging_metadata.get(
            "user_api_key_project_id"
        )
        user_api_key_end_user_id = (
            kwargs.get("user") if isinstance(kwargs, dict) else None
        ) or standard_logging_metadata.get("user_api_key_end_user_id")
        agent_id = standard_logging_metadata.get("agent_id")
        session_id = standard_logging_metadata.get(
            "session_id"
        ) or standard_logging_metadata.get("trace_id")

        targets: List[Tuple[str, str]] = []
        if user_api_key:
            targets.append(("api_key", user_api_key))
        if user_api_key_user_id:
            targets.append(("user", user_api_key_user_id))
        if user_api_key_team_id:
            targets.append(("team", user_api_key_team_id))
        if user_api_key_team_id and user_api_key_user_id:
            targets.append(
                ("team_member", f"{user_api_key_team_id}:{user_api_key_user_id}")
            )
        if user_api_key_end_user_id:
            targets.append(("end_user", user_api_key_end_user_id))
        if user_api_key_organization_id:
            targets.append(("organization", user_api_key_organization_id))
        if model_group:
            if user_api_key:
                targets.append(("model_per_key", f"{user_api_key}:{model_group}"))
            if user_api_key_team_id:
                targets.append(
                    ("model_per_team", f"{user_api_key_team_id}:{model_group}")
                )
            if user_api_key_organization_id:
                targets.append(
                    (
                        "model_per_organization",
                        f"{user_api_key_organization_id}:{model_group}",
                    )
                )
            if user_api_key_project_id:
                targets.append(
                    (
                        "model_per_project",
                        f"{user_api_key_project_id}:{model_group}",
                    )
                )
        if agent_id:
            targets.append(("agent", agent_id))
            if session_id:
                targets.append(("agent_session", f"{agent_id}:{session_id}"))
        return targets

    def _build_reservation_aware_tpm_ops(
        self,
        targets: List[Tuple[str, str]],
        reserved_scopes: Set[Tuple[str, str]],
        actual_tokens: int,
        reserved_tokens: int,
    ) -> List[RedisPipelineIncrementOperation]:
        """
        Emit per-scope TPM increment ops with reservation awareness.

        - Reserved scope (counter already at +reserved from pre-call):
          reconcile to actual via ``actual - reserved``.
        - Unreserved scope (counter never touched at pre-call):
          charge the full ``actual``.

        Same primitive serves success reconciliation, over-reservation
        release, and failure refund — pass ``actual_tokens=0`` for the pure
        refund case (reserved scopes get -reserved, unreserved get 0/skip).
        """
        ops: List[RedisPipelineIncrementOperation] = []
        for scope_key, scope_value in targets:
            if (scope_key, scope_value) in reserved_scopes:
                increment = actual_tokens - reserved_tokens
            else:
                increment = actual_tokens
            if increment == 0:
                continue
            ops.append(
                RedisPipelineIncrementOperation(
                    key=self.create_rate_limit_keys(scope_key, scope_value, "tokens"),
                    increment_value=increment,
                    ttl=self.window_size,
                )
            )
        return ops

    def _build_success_event_pipeline_operations(
        self,
        kwargs: Any,
        response_obj: Any,
        rate_limit_type: Literal["output", "input", "total"],
    ) -> List[RedisPipelineIncrementOperation]:
        """Build Redis pipeline increment ops for TPM / parallel-request counters."""
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )

        # Get metadata from standard_logging_object - this correctly handles both
        # 'metadata' and 'litellm_metadata' fields from litellm_params
        standard_logging_object = kwargs.get("standard_logging_object") or {}
        standard_logging_metadata = standard_logging_object.get("metadata") or {}

        user_api_key = standard_logging_metadata.get("user_api_key_hash")
        model_group = get_model_group_from_litellm_kwargs(kwargs)

        # Get total tokens from response
        total_tokens = 0
        # spot fix for /responses api
        if isinstance(response_obj, ModelResponse) or isinstance(
            response_obj, BaseLiteLLMOpenAIResponseObject
        ):
            _usage = getattr(response_obj, "usage", None)
            total_tokens = self._get_total_tokens_from_usage(
                usage=_usage, rate_limit_type=rate_limit_type
            )

        reserved_tokens = self._get_reserved_tokens_from_kwargs(
            kwargs=kwargs,
            standard_logging_metadata=standard_logging_metadata,
        )
        reserved_model = self._get_reserved_model_from_kwargs(
            kwargs=kwargs,
            standard_logging_metadata=standard_logging_metadata,
        )
        reserved_scopes = self._get_reserved_scopes_from_kwargs(
            kwargs=kwargs,
            standard_logging_metadata=standard_logging_metadata,
        )
        # Reconciliation must target the same model-scoped counter that the
        # pre-call reservation incremented. If a reservation was made,
        # ``reserved_model`` is authoritative; otherwise fall back to the
        # router's ``model_group`` (covers the no-reservation charge path).
        reconcile_model = reserved_model or model_group

        pipeline_operations: List[RedisPipelineIncrementOperation] = []

        # max_parallel_requests is its own counter (api-key only) — always decrement.
        if user_api_key:
            pipeline_operations.append(
                RedisPipelineIncrementOperation(
                    key=self.create_rate_limit_keys(
                        key="api_key",
                        value=user_api_key,
                        rate_limit_type="max_parallel_requests",
                    ),
                    increment_value=-1,
                    ttl=self.window_size,
                )
            )

        # ----------------------------------------------------------------
        # TPM reconciliation
        # Per-scope behavior:
        #   reserved scope    -> apply (actual - reserved) delta to settle
        #                        the counter at +actual.
        #   unreserved scope  -> charge the full actual usage (the
        #                        reservation never incremented this scope).
        # When no reservation was made, reserved_tokens=0 and reserved_scopes
        # is empty, so every scope falls through the unreserved branch and
        # gets the full actual charge — matching pre-PR behavior.
        # ----------------------------------------------------------------
        targets = self._collect_tpm_scope_targets(
            standard_logging_metadata=standard_logging_metadata,
            kwargs=kwargs,
            model_group=reconcile_model,
        )
        if reserved_tokens > 0 and total_tokens < reserved_tokens:
            verbose_proxy_logger.debug(
                f"Releasing unused TPM budget on success: "
                f"reserved={reserved_tokens}, actual={total_tokens}, "
                f"release={reserved_tokens - total_tokens}"
            )
        pipeline_operations.extend(
            self._build_reservation_aware_tpm_ops(
                targets=targets,
                reserved_scopes=reserved_scopes,
                actual_tokens=total_tokens,
                reserved_tokens=reserved_tokens,
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

        rate_limit_type = self.get_rate_limit_type()

        litellm_parent_otel_span: Union[Span, None] = _get_parent_otel_span_from_kwargs(
            kwargs
        )
        try:
            verbose_proxy_logger.debug(
                "INSIDE parallel request limiter ASYNC SUCCESS LOGGING"
            )

            pipeline_operations = self._build_success_event_pipeline_operations(
                kwargs=kwargs,
                response_obj=response_obj,
                rate_limit_type=rate_limit_type,
            )

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
        On failure: decrement max_parallel_requests and refund the upfront
        TPM reservation only against the scopes the reservation actually
        charged. Unreserved scopes were never incremented at pre-call, so
        refunding them would drive their counter negative.
        """
        from litellm.litellm_core_utils.core_helpers import (
            _get_parent_otel_span_from_kwargs,
        )

        try:
            litellm_parent_otel_span: Union[Span, None] = (
                _get_parent_otel_span_from_kwargs(kwargs)
            )
            standard_logging_object = kwargs.get("standard_logging_object") or {}
            standard_logging_metadata = standard_logging_object.get("metadata") or {}
            user_api_key = standard_logging_metadata.get("user_api_key_hash")

            pipeline_operations: List[RedisPipelineIncrementOperation] = []

            if user_api_key:
                pipeline_operations.append(
                    RedisPipelineIncrementOperation(
                        key=self.create_rate_limit_keys(
                            key="api_key",
                            value=user_api_key,
                            rate_limit_type="max_parallel_requests",
                        ),
                        increment_value=-1,
                        ttl=self.window_size,
                    )
                )

            # Skip the reservation refund if async_post_call_failure_hook
            # already released it (proxy-level rejection that also bubbles up
            # here as an LLM-error callback). max_parallel_requests is its
            # own counter and is always decremented per call.
            already_released = self._is_reservation_released(
                kwargs=kwargs,
                standard_logging_metadata=standard_logging_metadata,
            )
            reserved_tokens = (
                0
                if already_released
                else self._get_reserved_tokens_from_kwargs(
                    kwargs=kwargs,
                    standard_logging_metadata=standard_logging_metadata,
                )
            )
            if reserved_tokens > 0:
                verbose_proxy_logger.debug(
                    f"Releasing reserved TPM tokens on failure: {reserved_tokens}"
                )
                # Refund only against the scopes the reservation actually
                # charged. _build_reservation_aware_tpm_ops with
                # actual_tokens=0 emits -reserved on reserved scopes and 0
                # on unreserved (skipped), so unreserved scopes can't drift
                # negative. Targets are derived purely from the reserved
                # set so we don't even need to re-collect them from
                # metadata.
                reserved_scopes = self._get_reserved_scopes_from_kwargs(
                    kwargs=kwargs,
                    standard_logging_metadata=standard_logging_metadata,
                )
                pipeline_operations.extend(
                    self._build_reservation_aware_tpm_ops(
                        targets=list(reserved_scopes),
                        reserved_scopes=reserved_scopes,
                        actual_tokens=0,
                        reserved_tokens=reserved_tokens,
                    )
                )

            if pipeline_operations:
                await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                    increment_list=pipeline_operations,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                )
            if reserved_tokens > 0:
                self._mark_reservation_released(kwargs)
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

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: Optional[str] = None,
    ) -> None:
        """
        Release any TPM reservation when the request is rejected after the
        pre-call hook reserved tokens but before the LLM call ran (e.g. a
        downstream guardrail/auth hook raised). Without this, those
        reservations are stranded — async_log_failure_event is a litellm
        completion-level callback and never fires for proxy-side rejections.

        Idempotent via TPM_RESERVATION_RELEASED_KEY: if both this hook and
        async_log_failure_event end up running in the same flow, only the
        first refund applies.
        """
        try:
            if self._is_reservation_released(kwargs=request_data):
                return
            reserved_tokens = self._get_reserved_tokens_from_kwargs(kwargs=request_data)
            if reserved_tokens <= 0:
                return

            # Refund directly against the descriptors we reserved against —
            # the pre-call hook stashes them on the request data before
            # success/failure callbacks run.
            stashed = request_data.get("_litellm_rate_limit_descriptors")
            descriptors: List[RateLimitDescriptor] = (
                stashed if isinstance(stashed, list) else []
            )
            ops: List[RedisPipelineIncrementOperation] = []
            for descriptor in descriptors:
                rate_limit = descriptor.get("rate_limit") or {}
                if rate_limit.get("tokens_per_unit") is None:
                    continue
                ops.append(
                    RedisPipelineIncrementOperation(
                        key=self.create_rate_limit_keys(
                            descriptor["key"],
                            descriptor["value"],
                            "tokens",
                        ),
                        increment_value=-reserved_tokens,
                        ttl=self.window_size,
                    )
                )
            if ops:
                verbose_proxy_logger.debug(
                    f"Releasing reserved TPM tokens on proxy-level "
                    f"rejection: {reserved_tokens}"
                )
                await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                    increment_list=ops,
                    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                )
            self._mark_reservation_released(request_data)
        except Exception as e:
            verbose_proxy_logger.exception(
                f"Error releasing TPM reservation on post-call failure: {e}"
            )
        return None
