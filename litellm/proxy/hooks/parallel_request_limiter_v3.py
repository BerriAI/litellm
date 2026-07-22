"""
This is a rate limiter implementation based on a similar one by Envoy proxy.

This is currently in development and not yet ready for production.
"""

import asyncio
import binascii
import os
import uuid
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

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.constants import DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    get_key_tag_rpm_limit,
    get_model_rate_limit_from_metadata,
)
from litellm.proxy.auth.budget_throttle import throttled_limit
from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
    map_v3_rate_limit_type,
)
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject
from litellm.types.utils import (
    CallTypes,
    EmbeddingResponse,
    ModelResponse,
    TextCompletionResponse,
    Usage,
)

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

PARALLEL_ACQUIRE_SCRIPT = """
-- Atomic check-and-acquire for the max_parallel_requests concurrency gauge.
-- Each gauge key is a sorted set of per-request slot ids scored by acquire
-- time (Redis server clock). In-flight requests are counted by ZCARD after
-- pruning slots older than the slot TTL, so unlike the windowed RPM/TPM
-- counters the gauge is never reset while requests are in flight, a
-- rejected request never occupies a slot, and a slot leaked by a crashed
-- worker self-heals after the slot TTL even under continuous traffic.
--
-- KEYS: one gauge zset key per descriptor.
-- ARGV: per-key triples (limit, slot_ttl_seconds, slot_id).
-- Success: { 0, in_flight_1, ... }. Over-limit: { 1, key_index, in_flight, limit }.
local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1])
for i = 1, #KEYS do
    local limit = tonumber(ARGV[(i - 1) * 3 + 1])
    local slot_ttl = tonumber(ARGV[(i - 1) * 3 + 2])
    redis.call('ZREMRANGEBYSCORE', KEYS[i], '-inf', now - slot_ttl)
    local in_flight = redis.call('ZCARD', KEYS[i])
    if in_flight + 1 > limit then
        return { 1, i, in_flight, limit }
    end
end
local results = { 0 }
for i = 1, #KEYS do
    local slot_ttl = tonumber(ARGV[(i - 1) * 3 + 2])
    local slot_id = ARGV[(i - 1) * 3 + 3]
    redis.call('ZADD', KEYS[i], now, slot_id)
    redis.call('EXPIRE', KEYS[i], slot_ttl)
    table.insert(results, redis.call('ZCARD', KEYS[i]))
end
return results
"""

PARALLEL_RELEASE_SCRIPT = """
-- Release one slot per gauge key by removing this request's slot id.
-- ZREM of an absent member (or key) is a no-op, so a release without a
-- matching acquire (proxy-side rejection, double-fired callback, slot
-- already expired) can never free a slot owned by another request.
-- KEYS: gauge zset keys. ARGV: per-key slot_id.
-- Returns the remaining in-flight count per key.
local results = {}
for i = 1, #KEYS do
    redis.call('ZREM', KEYS[i], ARGV[i])
    table.insert(results, redis.call('ZCARD', KEYS[i]))
end
return results
"""

PARALLEL_COUNT_SCRIPT = """
-- Read the current in-flight count per gauge key (prunes expired slots
-- first so leaked slots do not inflate the reading).
-- KEYS: gauge zset keys. ARGV: per-key slot_ttl_seconds.
local time_reply = redis.call('TIME')
local now = tonumber(time_reply[1])
local results = {}
for i = 1, #KEYS do
    redis.call('ZREMRANGEBYSCORE', KEYS[i], '-inf', now - tonumber(ARGV[i]))
    table.insert(results, redis.call('ZCARD', KEYS[i]))
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
# Fraction of the available output budget reserved as the upfront floor when
# the request omits max_tokens. Applied to both DEFAULT_MAX_TOKENS_ESTIMATE
# (baseline floor) and to the smallest configured TPM limit (capped floor for
# small per-tenant TPM caps).
_TPM_FLOOR_FRACTION = 4
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
# Project-scoped ITPM/OTPM reservation stash, mirroring the TPM_RESERVED_*
# keys above but tracked separately since ITPM and OTPM are reserved from
# different estimates (input tokens vs. max_tokens) and reconciled against
# different actual-usage fields (prompt_tokens vs. completion_tokens).
ITPM_RESERVED_TOKENS_KEY = "_litellm_itpm_reserved_tokens"
ITPM_RESERVED_SCOPES_KEY = "_litellm_itpm_reserved_scopes"
OTPM_RESERVED_TOKENS_KEY = "_litellm_otpm_reserved_tokens"
OTPM_RESERVED_SCOPES_KEY = "_litellm_otpm_reserved_scopes"
# Descriptor "key" values for project-scoped ITPM/OTPM. Distinct from
# "model_per_project" (the combined-TPM descriptor) so both can be enforced
# on the same project+model simultaneously without colliding on cache keys.
PROJECT_ITPM_DESCRIPTOR_KEY = "model_per_project_itpm"
PROJECT_OTPM_DESCRIPTOR_KEY = "model_per_project_otpm"
RATE_LIMIT_DESCRIPTORS_KEY = "_litellm_rate_limit_descriptors"
# Pre-call RateLimitResponse stashed here so streaming success logging can
# mirror ``x-ratelimit-*`` headers into the SLP. Streaming exits
# common_request_processing before ``async_post_call_success_hook`` runs.
RATE_LIMIT_RESPONSE_KEY = "_litellm_proxy_rate_limit_response"
# Holds the acquisition the pre-call hook made for this request: the slot id
# plus the gauge counter keys it was registered under. The success/failure
# callbacks release only this exact acquisition: those callbacks also fire
# for requests rejected at pre-call (which never acquired a slot), and an
# id-less release would free a slot still owned by another in-flight request
# — every rejection would then raise effective concurrency above the
# configured limit.
MAX_PARALLEL_SLOT_ACQUIRED_KEY = "_litellm_max_parallel_slot_acquired"
# How long an acquired slot counts toward the in-flight total before it is
# considered leaked (worker crashed without any release callback firing) and
# pruned. Also the longest request duration the gauge can track: a request
# running longer than this stops occupying its slot.
PARALLEL_REQUEST_SLOT_TTL_SECONDS = 3600
# Stash keys live ONLY in metadata channels — never at the top level of the
# request body. Top-level keys are forwarded as body params to upstream
# providers, which reject unknown fields with 400/429 errors.
_LITELLM_STASH_KEYS: Tuple[str, ...] = (
    TPM_RESERVED_TOKENS_KEY,
    TPM_RESERVED_MODEL_KEY,
    TPM_RESERVED_SCOPES_KEY,
    TPM_RESERVATION_RELEASED_KEY,
    ITPM_RESERVED_TOKENS_KEY,
    ITPM_RESERVED_SCOPES_KEY,
    OTPM_RESERVED_TOKENS_KEY,
    OTPM_RESERVED_SCOPES_KEY,
    RATE_LIMIT_DESCRIPTORS_KEY,
    RATE_LIMIT_RESPONSE_KEY,
    MAX_PARALLEL_SLOT_ACQUIRED_KEY,
)


class RateLimitDescriptorRateLimitObject(TypedDict, total=False):
    requests_per_unit: Optional[int]
    tokens_per_unit: Optional[int]
    max_parallel_requests: Optional[int]
    window_size: Optional[int]


class RateLimitDescriptor(TypedDict):
    key: str
    value: str
    rate_limit: Optional[RateLimitDescriptorRateLimitObject]


class ParallelRequestGauge(TypedDict):
    counter_key: str
    limit: int
    descriptor_key: str


class ParallelSlotAcquisition(TypedDict):
    slot_id: str
    counter_keys: list[str]


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
            self.batch_rate_limiter_script = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                BATCH_RATE_LIMITER_SCRIPT
            )
            self.token_increment_script = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                TOKEN_INCREMENT_SCRIPT
            )
            self.check_and_increment_by_n_script = (
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(CHECK_AND_INCREMENT_BY_N_SCRIPT)
            )
            self.parallel_acquire_script = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                PARALLEL_ACQUIRE_SCRIPT
            )
            self.parallel_release_script = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                PARALLEL_RELEASE_SCRIPT
            )
            self.parallel_count_script = self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                PARALLEL_COUNT_SCRIPT
            )
        else:
            self.batch_rate_limiter_script = None
            self.token_increment_script = None
            self.check_and_increment_by_n_script = None
            self.parallel_acquire_script = None
            self.parallel_release_script = None
            self.parallel_count_script = None

        self.window_size = int(os.getenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", 60))

        # project_id:model pairs already warned about model_itpm_limit/model_otpm_limit
        # configured alongside model_tpm_limit, so the warning is logged once per pair
        # rather than per request.
        self._project_io_token_conflict_warned: set[str] = set()

        # When disabled, TPM is enforced post-call from actual usage (pre-v1.82
        # behavior) instead of reserving an estimated budget upfront, shedding
        # the extra per-request Redis Lua round-trip and the global-lock
        # in-memory fallback that the reservation path incurs.
        self.tpm_reservation_enabled = os.getenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "true").lower() == "true"

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
                verbose_proxy_logger.debug(f"Could not load batch rate limiter: {str(e)}")
        return self._batch_rate_limiter

    def _get_current_time(self) -> datetime:
        """Return the current time for rate limiting calculations."""
        return self._time_provider()

    @staticmethod
    def _no_max_tokens_output_floor(
        min_configured_tpm_limit: Optional[int],
    ) -> int:
        """Output-budget floor used when the request omits max_tokens.

        Capped at a fraction of the smallest configured TPM limit so a small
        per-tenant cap can't be tripped by the floor alone. Returns the
        baseline floor when no limit is provided.
        """
        baseline = DEFAULT_MAX_TOKENS_ESTIMATE // _TPM_FLOOR_FRACTION
        if min_configured_tpm_limit is None:
            return baseline
        return min(baseline, max(1, min_configured_tpm_limit // _TPM_FLOOR_FRACTION))

    def _estimate_tokens_for_request(
        self,
        data: dict,
        model: Optional[str] = None,
        min_configured_tpm_limit: Optional[int] = None,
    ) -> int:
        """
        Estimate total tokens this request will consume so we can reserve them
        upfront (input + output budget):
        estimated = input_tokens + max_tokens.

        Supports chat (messages), completions (prompt), and embeddings (input).

        ``min_configured_tpm_limit`` is the smallest ``tokens_per_unit`` among
        the TPM-bearing descriptors this request will be charged against. When
        provided, the no-``max_tokens`` output-budget floor is capped at a
        fraction of that limit so small TPM caps remain usable. Omit to
        preserve the unconstrained floor.
        """
        estimated_input_tokens, max_tokens_estimate = self._estimate_input_and_output_tokens(
            data=data,
            min_configured_tpm_limit=min_configured_tpm_limit,
        )
        total_estimated = estimated_input_tokens + max_tokens_estimate

        verbose_proxy_logger.debug(
            f"TPM reservation estimate: input={estimated_input_tokens}, "
            f"max_tokens={max_tokens_estimate}, total={total_estimated}"
        )

        return total_estimated

    def _estimate_input_and_output_tokens(
        self,
        data: dict,
        min_configured_tpm_limit: int | None = None,
    ) -> tuple[int, int]:
        """
        Estimate input tokens and output (max_tokens) budget separately, so
        callers needing independent ITPM/OTPM reservations (rather than one
        combined TPM reservation) can use each half on its own.

        ``min_configured_tpm_limit`` is the smallest ``tokens_per_unit`` among
        the TPM-bearing descriptors this request will be charged against. When
        provided, the no-``max_tokens`` output-budget floor is capped at a
        fraction of that limit so small TPM caps remain usable. Omit to
        preserve the unconstrained floor.
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

        estimated_input_tokens = max(1, total_chars // DEFAULT_CHARS_PER_TOKEN) if total_chars > 0 else 0

        explicit_max_tokens = data.get("max_tokens") or data.get("max_completion_tokens")

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
                # collectively bypass the limit. Cap the floor by a fraction of
                # the smallest TPM limit this request will be charged against,
                # so a small per-tenant TPM cap can't be tripped by the floor
                # alone.
                output_floor = self._no_max_tokens_output_floor(min_configured_tpm_limit)
                max_tokens_estimate = max(estimated_input_tokens, output_floor)

        return estimated_input_tokens, max_tokens_estimate

    def _is_redis_cluster(self) -> bool:
        """
        Check if the dual cache is using Redis cluster.

        Returns:
            bool: True if using Redis cluster, False otherwise.
        """
        from litellm.caching.redis_cluster_cache import RedisClusterCache

        return self.internal_usage_cache.dual_cache.redis_cache is not None and isinstance(
            self.internal_usage_cache.dual_cache.redis_cache, RedisClusterCache
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
                new_counter_value = (int(current_counter) if current_counter is not None else 0) + increment_value
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
            tokens_limit = key_metadata[window_key]["tokens_limit"]

            # Determine which limit to use for current_limit and limit_remaining
            current_limit: Optional[int] = None
            rate_limit_type: Optional[Literal["requests", "tokens", "max_parallel_requests"]] = None
            if counter_key.endswith(":requests"):
                current_limit = requests_limit
                rate_limit_type = "requests"
            elif counter_key.endswith(":tokens"):
                current_limit = tokens_limit
                rate_limit_type = "tokens"

            if current_limit is None or rate_limit_type is None:
                continue

            if counter_value is not None and int(counter_value) > current_limit:
                overall_code = "OVER_LIMIT"
                item_code = "OVER_LIMIT"

            # Only compute limit_remaining if current_limit is not None
            limit_remaining = current_limit - int(counter_value) if counter_value is not None else current_limit

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
                verbose_proxy_logger.warning(f"Redis Lua script failed for hash tag {hash_tag}: {str(e)}")
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
        parallel_slot_id: str | None = None,
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

        ``max_parallel_requests`` descriptors are enforced by the dedicated
        concurrency-gauge path (``_check_parallel_request_gauges``), never by
        the windowed counters. The gauge phase must stay AFTER the windowed
        check so a windowed rejection never strands an acquired slot; the
        reverse order would leak one gauge slot per RPM/TPM rejection.
        ``parallel_slot_id`` names the slot an admission registers; callers
        that enforce (not read_only) should pass the id they will later
        release with — when omitted, a generated slot id is used and the slot
        can only be reclaimed by TTL expiry.
        """

        current_time = self._get_current_time()
        now = current_time.timestamp()
        now_int = int(now)  # Convert to integer for Redis Lua script

        keys_to_fetch, key_metadata, gauges = self._collect_windowed_keys_and_gauges(
            descriptors=descriptors,
            skip_tpm_check=skip_tpm_check,
        )

        windowed_response = RateLimitResponse(overall_code="OK", statuses=[])
        if keys_to_fetch:
            ## CHECK IN-MEMORY CACHE
            cache_values = await self.internal_usage_cache.async_batch_get_cache(
                keys=keys_to_fetch,
                parent_otel_span=parent_otel_span,
                local_only=True,
            )

            if cache_values is not None:
                rate_limit_response = self.is_cache_list_over_limit(keys_to_fetch, cache_values, key_metadata)
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

            windowed_response = self.is_cache_list_over_limit(keys_to_fetch, cache_values, key_metadata)
            if windowed_response["overall_code"] == "OVER_LIMIT":
                return windowed_response

        if not gauges:
            return windowed_response

        gauge_response = await self._check_parallel_request_gauges(
            gauges=gauges,
            slot_id=parallel_slot_id or uuid.uuid4().hex,
            parent_otel_span=parent_otel_span,
            read_only=read_only,
        )
        return RateLimitResponse(
            overall_code=gauge_response["overall_code"],
            statuses=[*windowed_response["statuses"], *gauge_response["statuses"]],
        )

    def _collect_windowed_keys_and_gauges(
        self,
        descriptors: list[RateLimitDescriptor],
        skip_tpm_check: bool,
    ) -> tuple[list[str], dict[str, dict[str, Any]], list[ParallelRequestGauge]]:
        """
        Split descriptors into the windowed (window_key, counter_key) fetch
        list with its per-window metadata, and the concurrency gauges for
        descriptors carrying a max_parallel_requests limit.
        """
        keys_to_fetch: List[str] = []
        key_metadata: dict[str, dict[str, Any]] = {}
        gauges: list[ParallelRequestGauge] = []
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

            if max_parallel_requests_limit is not None:
                gauges.append(
                    ParallelRequestGauge(
                        counter_key=self.create_rate_limit_keys(
                            descriptor_key, descriptor_value, "max_parallel_requests"
                        ),
                        limit=int(max_parallel_requests_limit),
                        descriptor_key=descriptor_key,
                    )
                )

            rate_limit_set = False
            if requests_limit is not None:
                rpm_key = self.create_rate_limit_keys(descriptor_key, descriptor_value, "requests")
                keys_to_fetch.extend([window_key, rpm_key])
                rate_limit_set = True
            if tokens_limit is not None:
                tpm_key = self.create_rate_limit_keys(descriptor_key, descriptor_value, "tokens")
                keys_to_fetch.extend([window_key, tpm_key])
                rate_limit_set = True

            if not rate_limit_set:
                continue

            key_metadata[window_key] = {
                "requests_limit": (int(requests_limit) if requests_limit is not None else None),
                "tokens_limit": int(tokens_limit) if tokens_limit is not None else None,
                "window_size": int(window_size),
                "descriptor_key": descriptor_key,
            }
        return keys_to_fetch, key_metadata, gauges

    def _gauge_status(self, gauge: ParallelRequestGauge, in_flight: int, code: str) -> RateLimitStatus:
        return RateLimitStatus(
            code=code,
            current_limit=gauge["limit"],
            limit_remaining=max(0, gauge["limit"] - in_flight),
            rate_limit_type="max_parallel_requests",
            descriptor_key=gauge["descriptor_key"],
        )

    def _gauge_in_flight_from_cache_value(self, raw_value: Any) -> int:
        """
        In-flight count from a cached gauge value: a dict of slot_id ->
        acquire timestamp when the in-memory registry is authoritative, or
        the mirrored integer count from the last Redis script result.
        """
        if raw_value is None:
            return 0
        if isinstance(raw_value, dict):
            cutoff = self._get_current_time().timestamp() - PARALLEL_REQUEST_SLOT_TTL_SECONDS
            return sum(1 for ts in raw_value.values() if isinstance(ts, (int, float)) and ts >= cutoff)
        return max(0, int(raw_value))

    async def _check_parallel_request_gauges(
        self,
        gauges: list[ParallelRequestGauge],
        slot_id: str,
        parent_otel_span: Span | None = None,
        read_only: bool = False,
    ) -> RateLimitResponse:
        """
        Enforce max_parallel_requests as a concurrency gauge over a per-slot
        registry: each admitted request registers ``slot_id`` with its
        acquire time, and admission requires in_flight + 1 <= limit over the
        unexpired slots. Unlike the windowed RPM/TPM counters, the gauge is
        never reset while requests are in flight, a rejected request never
        occupies a slot, and a slot leaked by a crashed worker is pruned
        after PARALLEL_REQUEST_SLOT_TTL_SECONDS even under continuous
        traffic. Releases remove exactly this request's slot id, so a
        double-fired or unmatched release can never free another request's
        slot.
        """
        gauge_keys = [gauge["counter_key"] for gauge in gauges]

        if read_only:
            if self.parallel_count_script is not None:
                try:
                    raw_counts = await self.parallel_count_script(
                        keys=gauge_keys,
                        args=[PARALLEL_REQUEST_SLOT_TTL_SECONDS for _ in gauges],
                    )
                    counts = [max(0, int(value)) for value in raw_counts]
                except Exception as e:  # noqa: BLE001 - any Redis/Lua failure degrades to the local mirror, never a 500
                    verbose_proxy_logger.warning(f"parallel_count_script failed, using local mirror: {str(e)}")
                    counts = await self._read_local_gauge_counts(gauge_keys, parent_otel_span)
            else:
                counts = await self._read_local_gauge_counts(gauge_keys, parent_otel_span)
            statuses = []
            overall_code = "OK"
            for gauge, in_flight in zip(gauges, counts):
                code = "OVER_LIMIT" if in_flight >= gauge["limit"] else "OK"
                if code == "OVER_LIMIT":
                    overall_code = "OVER_LIMIT"
                statuses.append(self._gauge_status(gauge, in_flight, code))
            return RateLimitResponse(overall_code=overall_code, statuses=statuses)

        local_counts = await self._read_local_gauge_counts(gauge_keys, parent_otel_span)
        for gauge, in_flight in zip(gauges, local_counts):
            if in_flight >= gauge["limit"]:
                return RateLimitResponse(
                    overall_code="OVER_LIMIT",
                    statuses=[self._gauge_status(gauge, in_flight, "OVER_LIMIT")],
                )

        if self.parallel_acquire_script is not None:
            try:
                raw = await self.parallel_acquire_script(
                    keys=gauge_keys,
                    args=[
                        arg for gauge in gauges for arg in (gauge["limit"], PARALLEL_REQUEST_SLOT_TTL_SECONDS, slot_id)
                    ],
                )
            except Exception as e:  # noqa: BLE001 - any Redis/Lua failure degrades to in-memory enforcement, never a 500
                verbose_proxy_logger.warning(
                    f"parallel_acquire_script failed, falling back to in-memory gauge: {str(e)}"
                )
                async with self._check_and_increment_lock:
                    return await self._acquire_parallel_slots_in_memory(gauges, slot_id, parent_otel_span)
            if int(raw[0]) == 1:
                gauge = gauges[int(raw[1]) - 1]
                return RateLimitResponse(
                    overall_code="OVER_LIMIT",
                    statuses=[self._gauge_status(gauge, int(raw[2]), "OVER_LIMIT")],
                )
            statuses = []
            for gauge, in_flight in zip(gauges, raw[1:]):
                await self.internal_usage_cache.async_set_cache(
                    key=gauge["counter_key"],
                    value=int(in_flight),
                    ttl=PARALLEL_REQUEST_SLOT_TTL_SECONDS,
                    litellm_parent_otel_span=parent_otel_span,
                    local_only=True,
                )
                statuses.append(self._gauge_status(gauge, int(in_flight), "OK"))
            return RateLimitResponse(overall_code="OK", statuses=statuses)

        async with self._check_and_increment_lock:
            return await self._acquire_parallel_slots_in_memory(gauges, slot_id, parent_otel_span)

    async def _read_local_gauge_counts(
        self,
        gauge_keys: list[str],
        parent_otel_span: Span | None = None,
    ) -> list[int]:
        values = await self.internal_usage_cache.async_batch_get_cache(
            keys=gauge_keys,
            parent_otel_span=parent_otel_span,
            local_only=True,
        )
        if values is None:
            return [0 for _ in gauge_keys]
        return [self._gauge_in_flight_from_cache_value(value) for value in values]

    async def _acquire_parallel_slots_in_memory(
        self,
        gauges: list[ParallelRequestGauge],
        slot_id: str,
        parent_otel_span: Span | None = None,
    ) -> RateLimitResponse:
        """
        All-or-nothing in-memory slot-registry acquire. Caller holds the lock.

        A cached dict is the authoritative in-memory registry. A cached
        integer is the count mirrored from the last successful Redis script
        call: when Redis fails over to this path, that mirror still counts
        the slots in flight on the Redis side, so it is carried forward as
        an integer counter (not discarded as an empty registry, which would
        briefly double the admitted concurrency during a Redis outage).
        """
        now = self._get_current_time().timestamp()
        cutoff = now - PARALLEL_REQUEST_SLOT_TTL_SECONDS
        states: list[tuple[dict[str, float] | None, int]] = []
        for gauge in gauges:
            raw_value = await self.internal_usage_cache.async_get_cache(
                key=gauge["counter_key"],
                litellm_parent_otel_span=parent_otel_span,
                local_only=True,
            )
            if isinstance(raw_value, dict):
                registry: dict[str, float] | None = {
                    key: float(ts) for key, ts in raw_value.items() if isinstance(ts, (int, float)) and ts >= cutoff
                }
                in_flight = len(registry or {})
            elif raw_value is None:
                registry = {}
                in_flight = 0
            else:
                registry = None
                in_flight = max(0, int(raw_value))
            if in_flight + 1 > gauge["limit"]:
                return RateLimitResponse(
                    overall_code="OVER_LIMIT",
                    statuses=[self._gauge_status(gauge, in_flight, "OVER_LIMIT")],
                )
            states.append((registry, in_flight))

        statuses = []
        for gauge, (registry, in_flight) in zip(gauges, states):
            new_value: Union[dict[str, float], int] = (
                {**registry, slot_id: now} if registry is not None else in_flight + 1
            )
            await self.internal_usage_cache.async_set_cache(
                key=gauge["counter_key"],
                value=new_value,
                ttl=PARALLEL_REQUEST_SLOT_TTL_SECONDS,
                litellm_parent_otel_span=parent_otel_span,
                local_only=True,
            )
            statuses.append(self._gauge_status(gauge, in_flight + 1, "OK"))
        return RateLimitResponse(overall_code="OK", statuses=statuses)

    async def _release_parallel_request_slots(
        self,
        acquisition: ParallelSlotAcquisition,
        parent_otel_span: Span | None = None,
    ) -> None:
        """
        Release the max_parallel_requests slots acquired at pre-call by
        removing this request's slot id from every gauge it was registered
        under. Removing an absent slot id is a no-op, so a release without a
        matching acquire or a double-fired release can never free another
        request's slot. The in-memory fallback decrements integer mirror
        values (floored at 0) because the mirror carries no per-slot ids.
        """
        counter_keys = acquisition["counter_keys"]
        slot_id = acquisition["slot_id"]
        if not counter_keys or not slot_id:
            return
        if self.parallel_release_script is not None:
            try:
                raw = await self.parallel_release_script(
                    keys=counter_keys,
                    args=[slot_id for _ in counter_keys],
                )
                for counter_key, remaining in zip(counter_keys, raw):
                    await self.internal_usage_cache.async_set_cache(
                        key=counter_key,
                        value=max(0, int(remaining)),
                        ttl=PARALLEL_REQUEST_SLOT_TTL_SECONDS,
                        litellm_parent_otel_span=parent_otel_span,
                        local_only=True,
                    )
                return
            except Exception as e:  # noqa: BLE001 - any Redis/Lua failure degrades to the in-memory release, never a 500
                verbose_proxy_logger.warning(
                    f"parallel_release_script failed, falling back to in-memory release: {str(e)}"
                )

        async with self._check_and_increment_lock:
            for counter_key in counter_keys:
                raw_value = await self.internal_usage_cache.async_get_cache(
                    key=counter_key,
                    litellm_parent_otel_span=parent_otel_span,
                    local_only=True,
                )
                if isinstance(raw_value, dict):
                    if slot_id not in raw_value:
                        continue
                    new_value: Union[dict[str, float], int] = {
                        key: ts for key, ts in raw_value.items() if key != slot_id
                    }
                elif raw_value is None:
                    continue
                else:
                    new_value = max(0, int(raw_value) - 1)
                await self.internal_usage_cache.async_set_cache(
                    key=counter_key,
                    value=new_value,
                    ttl=PARALLEL_REQUEST_SLOT_TTL_SECONDS,
                    litellm_parent_otel_span=parent_otel_span,
                    local_only=True,
                )

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
            raise ValueError("atomic_check_and_increment_by_n: descriptors and increments must have the same length")

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

        flat_meta: List[Dict[str, Any]] = [m for _keys, _args, group_meta in descriptor_groups for m in group_meta]
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
            rlt: Literal["requests", "tokens"] = cast(Literal["requests", "tokens"], rate_limit_type)
            if rlt == "requests":
                limit_value = rate_limit.get("requests_per_unit")
                inc_amount = int(increment_amounts.get("requests", 0) or 0)
            else:
                limit_value = rate_limit.get("tokens_per_unit")
                inc_amount = int(increment_amounts.get("tokens", 0) or 0)
            if limit_value is None or inc_amount <= 0:
                continue
            counter_key = self.create_rate_limit_keys(descriptor_key, descriptor_value, rlt)
            # Counter-key TTL and window_size are conceptually distinct
            # ("how long the counter Redis key lives" vs "how long the
            # sliding window is"). Kept as separate values so a future
            # custom-TTL descriptor doesn't reintroduce a silent expiry bug.
            ttl_seconds = int(window_size)
            window_size_seconds = int(window_size)
            keys.extend([window_key, counter_key])
            # 4-tuple matches the Lua ARGV layout:
            #   [limit, increment, ttl_seconds, window_size_seconds].
            args.extend([int(limit_value), inc_amount, ttl_seconds, window_size_seconds])
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
                raw = await self.check_and_increment_by_n_script(  # pyright: ignore[reportOptionalCall]  # sole caller guards it is not None
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
                flat_meta: List[Dict[str, Any]] = [m for _k, _a, group_meta in descriptor_groups for m in group_meta]
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
                        f"Failed to refund {entry['counter_key']} on cross-descriptor rollback: {e}"
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
            window_expired = window_start is None or (now_int - int(window_start)) >= window_size
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
                            limit_remaining=max(0, meta["current_limit"] - current_counter),
                            rate_limit_type=meta["rate_limit_type"],
                            descriptor_key=meta["descriptor_key"],
                        )
                    ],
                )
            descriptor_state.append({"window_expired": window_expired, "current": current_counter})

        # Pass 2: apply increments.
        statuses: List[RateLimitStatus] = []
        for meta, state in zip(per_counter_meta, descriptor_state):
            new_counter = meta["increment"] if state["window_expired"] else state["current"] + meta["increment"]
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

        Excludes project ITPM/OTPM descriptors -- those are reserved
        separately (different estimate per bucket) via ``reserve_io_tokens``.
        """
        tpm_descriptors: List[RateLimitDescriptor] = [
            d
            for d in descriptors
            if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
            and (d.get("rate_limit") or {}).get("tokens_per_unit") is not None
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

    async def _refund_reserved_tokens(
        self,
        scopes: list[tuple[str, str]],
        amount: int,
        parent_otel_span: Span | None = None,
    ) -> None:
        """
        Directly decrement previously-reserved token counters for ``scopes``
        by ``amount``. Used to roll back a reservation that already
        succeeded once a *different* bucket in the same request turns out to
        be over its limit (e.g. ITPM reserved fine, OTPM then hits its
        limit -- the ITPM reservation must not be left inflated).
        """
        if amount <= 0 or not scopes:
            return
        pipeline_operations = [
            RedisPipelineIncrementOperation(
                key=self.create_rate_limit_keys(scope_key, scope_value, "tokens"),
                increment_value=-amount,
                ttl=self.window_size,
            )
            for scope_key, scope_value in scopes
        ]
        await self.async_increment_tokens_with_ttl_preservation(
            pipeline_operations=pipeline_operations,
            parent_otel_span=parent_otel_span,
        )

    async def reserve_io_tokens(
        self,
        descriptors: list[RateLimitDescriptor],
        estimated_input_tokens: int,
        estimated_output_tokens: int,
        parent_otel_span: Span | None = None,
    ) -> tuple[RateLimitResponse, int, int]:
        """
        Reserve ``estimated_input_tokens`` against project ITPM descriptors
        and ``estimated_output_tokens`` against project OTPM descriptors.

        ITPM and OTPM are reserved from different-sized estimates, so unlike
        same-size TPM descriptors they can't share a single
        ``atomic_check_and_increment_by_n`` call -- each bucket gets its own
        all-or-nothing atomic call. If the OTPM reservation is over limit
        after ITPM already succeeded, the ITPM reservation this call made is
        rolled back before returning, so a partial reservation never leaks.

        Returns ``(response, itpm_reserved, otpm_reserved)`` -- the latter two
        are the amounts actually reserved (0 if that bucket wasn't
        configured, or if the reservation failed), for the caller to stash
        for post-call reconciliation.
        """
        itpm_descriptors = [d for d in descriptors if d["key"] == PROJECT_ITPM_DESCRIPTOR_KEY]
        otpm_descriptors = [d for d in descriptors if d["key"] == PROJECT_OTPM_DESCRIPTOR_KEY]

        if not itpm_descriptors and not otpm_descriptors:
            return RateLimitResponse(overall_code="OK", statuses=[]), 0, 0

        statuses: list[RateLimitStatus] = []
        itpm_reserved = 0

        if itpm_descriptors:
            itpm_response = await self.atomic_check_and_increment_by_n(
                descriptors=itpm_descriptors,
                increments=[{"tokens": estimated_input_tokens} for _ in itpm_descriptors],
                parent_otel_span=parent_otel_span,
            )
            if itpm_response["overall_code"] == "OVER_LIMIT":
                return itpm_response, 0, 0
            itpm_reserved = estimated_input_tokens
            statuses.extend(itpm_response["statuses"])

        if otpm_descriptors:
            otpm_response = await self.atomic_check_and_increment_by_n(
                descriptors=otpm_descriptors,
                increments=[{"tokens": estimated_output_tokens} for _ in otpm_descriptors],
                parent_otel_span=parent_otel_span,
            )
            if otpm_response["overall_code"] == "OVER_LIMIT":
                if itpm_reserved > 0:
                    await self._refund_reserved_tokens(
                        scopes=[(d["key"], d["value"]) for d in itpm_descriptors],
                        amount=itpm_reserved,
                        parent_otel_span=parent_otel_span,
                    )
                return otpm_response, 0, 0
            statuses.extend(otpm_response["statuses"])
            return (
                RateLimitResponse(overall_code="OK", statuses=statuses),
                itpm_reserved,
                estimated_output_tokens,
            )

        return RateLimitResponse(overall_code="OK", statuses=statuses), itpm_reserved, 0

    def create_organization_rate_limit_descriptor(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: Optional[str] = None
    ) -> List[RateLimitDescriptor]:
        descriptors: List[RateLimitDescriptor] = []

        # Global org rate limits
        if user_api_key_dict.org_id is not None and (
            user_api_key_dict.organization_rpm_limit is not None or user_api_key_dict.organization_tpm_limit is not None
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
            get_model_rate_limit_from_metadata(user_api_key_dict, "organization_metadata", "model_rpm_limit")
            is not None
            or get_model_rate_limit_from_metadata(user_api_key_dict, "organization_metadata", "model_tpm_limit")
            is not None
        ):
            _tpm_limit_for_team_model = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "organization_metadata", "model_tpm_limit") or {}
            )
            _rpm_limit_for_team_model = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "organization_metadata", "model_rpm_limit") or {}
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
                    model_specific_tpm_limit = _tpm_limit_for_team_model[requested_model]
                if requested_model in _rpm_limit_for_team_model:
                    model_specific_rpm_limit = _rpm_limit_for_team_model[requested_model]
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

        _tpm_limit_for_key_model = get_key_model_tpm_limit(user_api_key_dict, model_name=requested_model)
        _rpm_limit_for_key_model = get_key_model_rpm_limit(user_api_key_dict, model_name=requested_model)

        if _tpm_limit_for_key_model is None and _rpm_limit_for_key_model is None:
            return

        _tpm_limit_for_key_model = _tpm_limit_for_key_model or {}
        _rpm_limit_for_key_model = _rpm_limit_for_key_model or {}

        # Check if model has any rate limits configured
        should_check_rate_limit = (
            requested_model in _tpm_limit_for_key_model or requested_model in _rpm_limit_for_key_model
        )

        if not should_check_rate_limit:
            return

        # Get model-specific limits
        model_specific_tpm_limit: Optional[int] = _tpm_limit_for_key_model.get(requested_model)
        model_specific_rpm_limit: Optional[int] = _rpm_limit_for_key_model.get(requested_model)

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

    def _add_tag_per_key_rate_limit_descriptor(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        data: dict,
        descriptors: list[RateLimitDescriptor],
    ) -> None:
        """
        Add per-request-tag rpm limit descriptors for the API key.

        Each tag carried on the request that has a configured limit gets its own
        ``{api_key}:{tag}`` counter, so a burst on one tag/group never consumes
        another's budget. Tags without a configured limit fall through to the
        key-level descriptor.
        """
        if not user_api_key_dict.api_key:
            return

        tag_rpm_limit = get_key_tag_rpm_limit(user_api_key_dict) or {}
        if not tag_rpm_limit:
            return

        for tag in dict.fromkeys(get_tags_from_request_body(data)):
            rpm_limit = tag_rpm_limit.get(tag)
            if rpm_limit is None:
                continue
            descriptors.append(
                RateLimitDescriptor(
                    key="tag_per_key",
                    value=f"{user_api_key_dict.api_key}:{tag}",
                    rate_limit={
                        "requests_per_unit": rpm_limit,
                        "tokens_per_unit": None,
                        "window_size": self.window_size,
                    },
                )
            )

    def _add_mcp_per_key_rate_limit_descriptor(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        mcp_server_name: Optional[str],
        descriptors: List[RateLimitDescriptor],
    ) -> None:
        """
        Add a per-MCP-server rpm descriptor for the API key, if a limit is
        configured for the server being called.

        MCP tool calls have no token usage, so only requests_per_unit is set;
        tokens_per_unit stays None so the TPM reservation path is never engaged.
        """
        from litellm.proxy.auth.auth_utils import get_key_mcp_rpm_limit

        if not mcp_server_name or not user_api_key_dict.api_key:
            return

        mcp_rpm_limit = get_key_mcp_rpm_limit(user_api_key_dict)
        if not mcp_rpm_limit:
            return

        server_rpm_limit = mcp_rpm_limit.get(mcp_server_name)
        if server_rpm_limit is None:
            return

        descriptors.append(
            RateLimitDescriptor(
                key="mcp_per_key",
                value=f"{user_api_key_dict.api_key}:{mcp_server_name}",
                rate_limit={
                    "requests_per_unit": server_rpm_limit,
                    "tokens_per_unit": None,
                    "window_size": self.window_size,
                },
            )
        )

    def _add_mcp_per_team_rate_limit_descriptor(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        mcp_server_name: Optional[str],
        descriptors: List[RateLimitDescriptor],
    ) -> None:
        """
        Add a per-MCP-server rpm descriptor for the team, if a limit is
        configured for the server being called.
        """
        from litellm.proxy.auth.auth_utils import get_team_mcp_rpm_limit

        if not mcp_server_name or not user_api_key_dict.team_id:
            return

        mcp_rpm_limit = get_team_mcp_rpm_limit(user_api_key_dict)
        if not mcp_rpm_limit:
            return

        server_rpm_limit = mcp_rpm_limit.get(mcp_server_name)
        if server_rpm_limit is None:
            return

        descriptors.append(
            RateLimitDescriptor(
                key="mcp_per_team",
                value=f"{user_api_key_dict.team_id}:{mcp_server_name}",
                rate_limit={
                    "requests_per_unit": server_rpm_limit,
                    "tokens_per_unit": None,
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

    def _get_resolved_agent_id(self, user_api_key_dict: UserAPIKeyAuth, data: dict) -> Optional[str]:
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
        call_type: Optional[str] = None,
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
            throttle_pct = user_api_key_dict.budget_throttle_pct
            descriptors.append(
                RateLimitDescriptor(
                    key="api_key",
                    value=user_api_key_dict.api_key,
                    rate_limit={
                        "requests_per_unit": self._get_enforced_limit(
                            limit_value=throttled_limit(user_api_key_dict.rpm_limit, throttle_pct),
                            limit_type=rpm_limit_type,
                            model_has_failures=model_has_failures,
                        ),
                        "tokens_per_unit": self._get_enforced_limit(
                            limit_value=throttled_limit(user_api_key_dict.tpm_limit, throttle_pct),
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
            user_api_key_dict.user_rpm_limit is not None or user_api_key_dict.user_tpm_limit is not None
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
            user_api_key_dict.team_rpm_limit is not None or user_api_key_dict.team_tpm_limit is not None
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
            user_api_key_dict.team_member_rpm_limit is not None or user_api_key_dict.team_member_tpm_limit is not None
        ):
            team_member_value = f"{user_api_key_dict.team_id}:{user_api_key_dict.user_id}"
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
            user_api_key_dict.end_user_rpm_limit is not None or user_api_key_dict.end_user_tpm_limit is not None
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

        # Per-request-tag rate limits scoped to this key
        self._add_tag_per_key_rate_limit_descriptor(
            user_api_key_dict=user_api_key_dict,
            data=data,
            descriptors=descriptors,
        )

        # REST MCP calls pass the raw body through this hook before server
        # resolution; only the later synthetic hook payload may carry this key.
        if call_type == CallTypes.call_mcp_tool.value and "server_id" not in data:
            mcp_server_name = data.get("mcp_server_name", None)
            self._add_mcp_per_key_rate_limit_descriptor(
                user_api_key_dict=user_api_key_dict,
                mcp_server_name=mcp_server_name,
                descriptors=descriptors,
            )
            self._add_mcp_per_team_rate_limit_descriptor(
                user_api_key_dict=user_api_key_dict,
                mcp_server_name=mcp_server_name,
                descriptors=descriptors,
            )

        if (
            get_team_model_rpm_limit(user_api_key_dict) is not None
            or get_team_model_tpm_limit(user_api_key_dict) is not None
        ):
            _tpm_limit_for_team_model = get_team_model_tpm_limit(user_api_key_dict) or {}
            _rpm_limit_for_team_model = get_team_model_rpm_limit(user_api_key_dict) or {}
            should_check_rate_limit = False
            if requested_model in _tpm_limit_for_team_model:
                should_check_rate_limit = True
            elif requested_model in _rpm_limit_for_team_model:
                should_check_rate_limit = True

            if should_check_rate_limit:
                model_specific_tpm_limit = None
                model_specific_rpm_limit = None
                if requested_model in _tpm_limit_for_team_model:
                    model_specific_tpm_limit = _tpm_limit_for_team_model[requested_model]
                if requested_model in _rpm_limit_for_team_model:
                    model_specific_rpm_limit = _rpm_limit_for_team_model[requested_model]
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
            verbose_proxy_logger.debug(f"Error checking model failure status: {str(e)}, defaulting to enforce limits")
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
            get_model_rate_limit_from_metadata(user_api_key_dict, "team_metadata", "model_rpm_limit") is not None
            or get_model_rate_limit_from_metadata(user_api_key_dict, "team_metadata", "model_tpm_limit") is not None
        ):
            _tpm_limit_for_team_model = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "team_metadata", "model_tpm_limit") or {}
            )
            _rpm_limit_for_team_model = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "team_metadata", "model_rpm_limit") or {}
            )
            should_check_rate_limit = (
                requested_model in _tpm_limit_for_team_model or requested_model in _rpm_limit_for_team_model
            )

            if should_check_rate_limit and requested_model is not None:
                model_specific_tpm_limit = _tpm_limit_for_team_model.get(requested_model)
                model_specific_rpm_limit = _rpm_limit_for_team_model.get(requested_model)
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
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_rpm_limit") is not None
            or get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_tpm_limit") is not None
        ):
            _tpm_limit_for_project_model = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_tpm_limit") or {}
            )
            _rpm_limit_for_project_model = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_rpm_limit") or {}
            )
            should_check_rate_limit = (
                requested_model in _tpm_limit_for_project_model or requested_model in _rpm_limit_for_project_model
            )

            if should_check_rate_limit and requested_model is not None:
                model_specific_tpm_limit = _tpm_limit_for_project_model.get(requested_model)
                model_specific_rpm_limit = _rpm_limit_for_project_model.get(requested_model)
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

    def _warn_project_io_token_and_tpm_coexist_once(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str,
    ) -> None:
        """Log once per project:model pair when model_itpm_limit/model_otpm_limit
        is configured alongside model_tpm_limit. Both are enforced (mirrors the
        deployment-level itpm/otpm + tpm/rpm coexistence behavior in
        ``model_rate_limit_check.py``); this is purely a heads-up warning.
        """
        tpm_limit_for_project_model = (
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_tpm_limit") or {}
        )
        if requested_model not in tpm_limit_for_project_model:
            return
        warn_key = f"{user_api_key_dict.project_id}:{requested_model}"
        if warn_key in self._project_io_token_conflict_warned:
            return
        self._project_io_token_conflict_warned.add(warn_key)
        verbose_proxy_logger.warning(
            f"Project '{user_api_key_dict.project_id}' configures model_itpm_limit/model_otpm_limit "
            f"alongside model_tpm_limit for model '{requested_model}'; both limit types are enforced"
        )

    def _add_project_io_token_rate_limit_descriptors_from_metadata(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str | None,
        descriptors: list[RateLimitDescriptor],
    ) -> None:
        """Add project-scoped ITPM/OTPM descriptors from project_metadata.

        Enforced independently of, and alongside, the combined ``model_per_project``
        TPM descriptor above -- these give Bedrock Mantle-style separate input/output
        token quotas at the project level.
        """
        if requested_model is None or user_api_key_dict.project_id is None:
            return

        itpm_limit_for_project_model = (
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_itpm_limit") or {}
        )
        otpm_limit_for_project_model = (
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_otpm_limit") or {}
        )

        model_itpm_limit = itpm_limit_for_project_model.get(requested_model)
        model_otpm_limit = otpm_limit_for_project_model.get(requested_model)

        if model_itpm_limit is None and model_otpm_limit is None:
            return

        self._warn_project_io_token_and_tpm_coexist_once(user_api_key_dict, requested_model)

        descriptor_value = f"{user_api_key_dict.project_id}:{requested_model}"
        if model_itpm_limit is not None:
            descriptors.append(
                RateLimitDescriptor(
                    key=PROJECT_ITPM_DESCRIPTOR_KEY,
                    value=descriptor_value,
                    rate_limit={
                        "requests_per_unit": None,
                        "tokens_per_unit": model_itpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )
        if model_otpm_limit is not None:
            descriptors.append(
                RateLimitDescriptor(
                    key=PROJECT_OTPM_DESCRIPTOR_KEY,
                    value=descriptor_value,
                    rate_limit={
                        "requests_per_unit": None,
                        "tokens_per_unit": model_otpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

    def _handle_rate_limit_error(
        self,
        response: RateLimitResponse,
        descriptors: List[RateLimitDescriptor],
        requested_model: Optional[str] = None,
    ) -> None:
        """Handle rate limit exceeded by raising :class:`ProxyRateLimitError` (a 429)."""
        for status in response["statuses"]:
            if status["code"] == "OVER_LIMIT":
                descriptor_key = status["descriptor_key"]
                matching_descriptor = next(
                    (desc for desc in descriptors if desc["key"] == descriptor_key),
                    None,
                )
                descriptor_value = matching_descriptor["value"] if matching_descriptor is not None else "unknown"

                now = self._get_current_time().timestamp()
                reset_time = now + self.window_size
                reset_time_formatted = datetime.fromtimestamp(reset_time).strftime("%Y-%m-%d %H:%M:%S UTC")

                remaining_display = max(0, status["limit_remaining"])
                rate_limit_type = status["rate_limit_type"]
                current_limit = status["current_limit"]

                detail = (
                    f"Rate limit exceeded for {descriptor_key}: {descriptor_value}. "
                    f"Limit type: {rate_limit_type}. "
                    f"Current limit: {current_limit}, Remaining: {remaining_display}. "
                    f"Limit resets at: {reset_time_formatted}"
                )

                resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(requested_model)
                raise ProxyRateLimitError(
                    detail=detail,
                    headers={
                        "retry-after": str(self.window_size),
                        "rate_limit_type": str(status["rate_limit_type"]),
                        "reset_at": reset_time_formatted,
                    },
                    rate_limit_type=map_v3_rate_limit_type(status["rate_limit_type"]),
                    model=resolved_model,
                    llm_provider=llm_provider,
                )

    async def _reserve_project_io_tokens_or_raise(
        self,
        descriptors: list[RateLimitDescriptor],
        data: dict,
        requested_model: str | None,
        user_api_key_dict: UserAPIKeyAuth,
        tpm_reservation_scopes: list[tuple[str, str]],
        tpm_reservation_amount: int,
    ) -> None:
        """
        Reserve project-scoped ITPM/OTPM tokens (Bedrock Mantle-style
        separate input/output token buckets), independently of -- and, when
        both are configured, in addition to -- the combined-TPM reservation
        the caller already made. Raises (via ``_handle_rate_limit_error``) on
        an over-limit reservation, first rolling back the combined-TPM
        reservation named by ``tpm_reservation_scopes``/``tpm_reservation_amount``
        if one was made, so a partial reservation never leaks.
        """
        io_token_descriptors = [
            d for d in descriptors if d["key"] in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
        ]
        if not io_token_descriptors:
            return

        configured_otpm_limits = [
            int(v)
            for d in io_token_descriptors
            if d["key"] == PROJECT_OTPM_DESCRIPTOR_KEY
            for v in [(d.get("rate_limit") or {}).get("tokens_per_unit")]
            if v is not None
        ]
        min_configured_otpm_limit = min(configured_otpm_limits) if configured_otpm_limits else None

        estimated_input_tokens, estimated_output_tokens = self._estimate_input_and_output_tokens(
            data=data,
            min_configured_tpm_limit=min_configured_otpm_limit,
        )
        estimated_input_tokens = max(estimated_input_tokens, 1)
        estimated_output_tokens = max(estimated_output_tokens, 1)

        # Hard-cap generation length so an unbounded response can't overshoot
        # the OTPM budget before post-call reconciliation runs, mirroring the
        # combined-TPM floor cap in the caller. Only tightens
        # data["max_tokens"]; never loosens a cap already set.
        capped_output_floor = self._no_max_tokens_output_floor(min_configured_otpm_limit)
        baseline_floor = DEFAULT_MAX_TOKENS_ESTIMATE // _TPM_FLOOR_FRACTION
        has_explicit_max_tokens = data.get("max_tokens") is not None or data.get("max_completion_tokens") is not None
        is_embedding = data.get("input") is not None
        if capped_output_floor < baseline_floor and not has_explicit_max_tokens and not is_embedding:
            existing_cap = data.get("max_tokens")
            if existing_cap is None or capped_output_floor < existing_cap:
                data["max_tokens"] = capped_output_floor

        io_response, itpm_reserved, otpm_reserved = await self.reserve_io_tokens(
            descriptors=io_token_descriptors,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )

        if io_response["overall_code"] == "OVER_LIMIT":
            # A combined-TPM reservation may have already succeeded above for
            # this same request; refund it too, or its counter stays inflated
            # until the window's TTL expires.
            await self._refund_reserved_tokens(
                scopes=tpm_reservation_scopes,
                amount=tpm_reservation_amount,
                parent_otel_span=user_api_key_dict.parent_otel_span,
            )
            acquisition = self._get_parallel_slot_acquisition(kwargs=data)
            if acquisition is not None:
                await self._release_parallel_request_slots(
                    acquisition=acquisition,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )
                self._clear_parallel_slot_marker(data)
            self._handle_rate_limit_error(
                response=io_response,
                descriptors=descriptors,
                requested_model=requested_model,
            )
            return

        if itpm_reserved > 0:
            itpm_scopes = [
                (d["key"], d["value"]) for d in io_token_descriptors if d["key"] == PROJECT_ITPM_DESCRIPTOR_KEY
            ]
            self._stash_value_in_metadata_channels(data=data, key=ITPM_RESERVED_TOKENS_KEY, value=itpm_reserved)
            self._stash_value_in_metadata_channels(
                data=data,
                key=ITPM_RESERVED_SCOPES_KEY,
                value=[[k, v] for k, v in itpm_scopes],
            )
        if otpm_reserved > 0:
            otpm_scopes = [
                (d["key"], d["value"]) for d in io_token_descriptors if d["key"] == PROJECT_OTPM_DESCRIPTOR_KEY
            ]
            self._stash_value_in_metadata_channels(data=data, key=OTPM_RESERVED_TOKENS_KEY, value=otpm_reserved)
            self._stash_value_in_metadata_channels(
                data=data,
                key=OTPM_RESERVED_SCOPES_KEY,
                value=[[k, v] for k, v in otpm_scopes],
            )

        stored_response = data.get("litellm_proxy_rate_limit_response")
        if isinstance(stored_response, dict):
            stored_response.setdefault("statuses", []).extend(io_response["statuses"])
        elif io_response["statuses"]:
            data["litellm_proxy_rate_limit_response"] = io_response
            self._stash_value_in_metadata_channels(
                data=data,
                key=RATE_LIMIT_RESPONSE_KEY,
                value=io_response,
            )

        verbose_proxy_logger.debug(
            f"ITPM/OTPM tokens reserved: itpm={itpm_reserved}, otpm={otpm_reserved} for model {requested_model}"
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

        # Reject caller-supplied stash values before any read/write. Otherwise
        # a client can inject ``_litellm_rate_limit_descriptors`` /
        # ``_litellm_tpm_reserved_tokens`` in body ``metadata`` and have
        # ``async_post_call_failure_hook`` refund TPM counters against scopes
        # they name (e.g. another tenant's api_key).
        self._strip_stash_keys_from_all_channels(data)

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
            call_type=call_type,
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
        self._add_project_io_token_rate_limit_descriptors_from_metadata(
            user_api_key_dict=user_api_key_dict,
            requested_model=requested_model,
            descriptors=descriptors,
        )

        # Org Level Rate Limits
        descriptors.extend(self.create_organization_rate_limit_descriptor(user_api_key_dict, requested_model))

        # Only check rate limits if we have descriptors with actual limits
        if descriptors:
            # First pass: RPM and max_parallel_requests sliding-window check.
            # When reservation is enabled, `skip_tpm_check=True` tells
            # should_rate_limit to ignore each descriptor's tokens_per_unit so
            # its +1-per-key Lua / in-memory increment never touches the
            # :tokens counters — those are owned exclusively by the atomic
            # reserve_tpm_tokens path below. Without this, every concurrent
            # in-flight request would pre-inflate the :tokens counter by 1,
            # shrinking the effective TPM budget by N and causing
            # false-positive 429s under bursts. When reservation is disabled,
            # this pass enforces TPM directly from the post-call counters.
            parallel_counter_keys = [
                self.create_rate_limit_keys(d["key"], d["value"], "max_parallel_requests")
                for d in descriptors
                if (d.get("rate_limit") or {}).get("max_parallel_requests") is not None
            ]
            parallel_slot_id = uuid.uuid4().hex if parallel_counter_keys else None

            response = await self.should_rate_limit(
                descriptors=descriptors,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                skip_tpm_check=self.tpm_reservation_enabled,
                parallel_slot_id=parallel_slot_id,
            )

            if response["overall_code"] == "OVER_LIMIT":
                self._handle_rate_limit_error(
                    response=response,
                    descriptors=descriptors,
                    requested_model=requested_model,
                )
            else:
                # add descriptors to request headers
                data["litellm_proxy_rate_limit_response"] = response
                # Mirror into metadata so streaming success logging can find
                # it via ``kwargs["litellm_params"]["metadata"]``.
                self._stash_value_in_metadata_channels(
                    data=data,
                    key=RATE_LIMIT_RESPONSE_KEY,
                    value=response,
                )
                if parallel_slot_id is not None:
                    self._stash_value_in_metadata_channels(
                        data=data,
                        key=MAX_PARALLEL_SLOT_ACQUIRED_KEY,
                        value={
                            "slot_id": parallel_slot_id,
                            "counter_keys": parallel_counter_keys,
                        },
                    )

            # ----------------------------------------------------------------
            # TPM token reservation
            # Atomically reserve estimated tokens upfront so concurrent
            # requests cannot all observe "under limit" before any of them
            # has incremented the counter. atomic_check_and_increment_by_n
            # uses Redis Lua when available and falls back to an asyncio-locked
            # in-memory check otherwise — single-worker protection still holds
            # even without Redis.
            # ----------------------------------------------------------------
            configured_tpm_limits = [
                int(v)
                for d in descriptors
                if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
                for v in [(d.get("rate_limit") or {}).get("tokens_per_unit")]
                if v is not None
            ]
            has_tpm_limits = bool(configured_tpm_limits)

            # Populated on a successful combined-TPM reservation below, so the
            # project ITPM/OTPM block further down can roll it back if a
            # different bucket in the same request subsequently hits its
            # limit. Stays empty/0 whenever no combined-TPM reservation was
            # made (or it was over limit, in which case execution never
            # reaches the ITPM/OTPM block -- `_handle_rate_limit_error` raises).
            tpm_reservation_scopes: list[tuple[str, str]] = []
            tpm_reservation_amount = 0

            if has_tpm_limits and self.tpm_reservation_enabled:
                min_configured_tpm_limit = min(configured_tpm_limits)

                # When the configured TPM cap is small enough to constrain the
                # no-max_tokens floor, also hard-cap the model output via
                # data["max_tokens"] so concurrent unbounded generations can't
                # spend past the limit before post-call reconciliation runs.
                # Skip when the request already sets max_tokens or has no
                # generation budget at all (embeddings).
                capped_floor = self._no_max_tokens_output_floor(min_configured_tpm_limit)
                baseline_floor = DEFAULT_MAX_TOKENS_ESTIMATE // _TPM_FLOOR_FRACTION
                has_explicit_max_tokens = (
                    data.get("max_tokens") is not None or data.get("max_completion_tokens") is not None
                )
                is_embedding = data.get("input") is not None
                if capped_floor < baseline_floor and not has_explicit_max_tokens and not is_embedding:
                    data["max_tokens"] = capped_floor

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
                        min_configured_tpm_limit=min_configured_tpm_limit,
                    ),
                    1,
                )

                tpm_response = await self.reserve_tpm_tokens(
                    descriptors=descriptors,
                    estimated_tokens=estimated_tokens,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )

                if tpm_response["overall_code"] == "OVER_LIMIT":
                    acquisition = self._get_parallel_slot_acquisition(kwargs=data)
                    if acquisition is not None:
                        await self._release_parallel_request_slots(
                            acquisition=acquisition,
                            parent_otel_span=user_api_key_dict.parent_otel_span,
                        )
                        self._clear_parallel_slot_marker(data)
                    self._handle_rate_limit_error(
                        response=tpm_response,
                        descriptors=descriptors,
                        requested_model=requested_model,
                    )
                else:
                    self._stash_value_in_metadata_channels(
                        data=data,
                        key=RATE_LIMIT_DESCRIPTORS_KEY,
                        value=descriptors,
                    )
                    # Capture the exact (key, value) scopes the reservation
                    # incremented so post-call reconciliation only applies
                    # the (actual - reserved) delta to those — unreserved
                    # scopes get charged the full actual usage instead.
                    reserved_scopes: List[Tuple[str, str]] = [
                        (d["key"], d["value"])
                        for d in descriptors
                        if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
                        and (d.get("rate_limit") or {}).get("tokens_per_unit") is not None
                    ]
                    tpm_reservation_scopes = reserved_scopes
                    tpm_reservation_amount = estimated_tokens
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
                        stored_response.setdefault("statuses", []).extend(tpm_response["statuses"])
                    elif tpm_response["statuses"]:
                        data["litellm_proxy_rate_limit_response"] = tpm_response
                        # Keep the metadata stash in sync when this is the
                        # first snapshot written.
                        self._stash_value_in_metadata_channels(
                            data=data,
                            key=RATE_LIMIT_RESPONSE_KEY,
                            value=tpm_response,
                        )

                    verbose_proxy_logger.debug(f"TPM tokens reserved: {estimated_tokens} for model {requested_model}")

            await self._reserve_project_io_tokens_or_raise(
                descriptors=descriptors,
                data=data,
                requested_model=requested_model,
                user_api_key_dict=user_api_key_dict,
                tpm_reservation_scopes=tpm_reservation_scopes,
                tpm_reservation_amount=tpm_reservation_amount,
            )

        # Defense-in-depth: scrub any stash key that escaped onto data
        # top-level (stale cache hit, router pass, test fixture) before the
        # body is forwarded to the provider.
        self._strip_stash_keys_from_top_level(data)

    @staticmethod
    def _strip_stash_keys_from_top_level(data: Any) -> None:
        if not isinstance(data, dict):
            return
        for stash_key in _LITELLM_STASH_KEYS:
            data.pop(stash_key, None)

    @classmethod
    def _strip_stash_keys_from_all_channels(cls, data: Any) -> None:
        if not isinstance(data, dict):
            return
        cls._strip_stash_keys_from_top_level(data)
        for channel in ("metadata", "litellm_metadata"):
            channel_dict = data.get(channel)
            if isinstance(channel_dict, dict):
                for stash_key in _LITELLM_STASH_KEYS:
                    channel_dict.pop(stash_key, None)

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
                    if hasattr(usage, "prompt_tokens_details") and usage.prompt_tokens_details is not None:
                        cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0

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
            group_operations = [op for op in pipeline_operations if op["key"] in group_keys]

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
            verbose_proxy_logger.debug("TTL preservation script not available, using regular pipeline")
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
            verbose_proxy_logger.warning(f"TTL preservation failed, falling back to regular pipeline: {str(e)}")
            # Fallback to regular pipeline on error
            await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                increment_list=pipeline_operations,
                litellm_parent_otel_span=parent_otel_span,
            )

    def get_rate_limit_type(self) -> Literal["output", "input", "total"]:
        from litellm.proxy.proxy_server import general_settings

        specified_rate_limit_type = general_settings.get("token_rate_limit_type", "total")
        if specified_rate_limit_type not in [
            "output",
            "input",
            "total",
        ]:
            return "total"  # default to total
        return specified_rate_limit_type

    @staticmethod
    def _merge_ratelimit_statuses_into_additional_headers(
        additional_headers: Dict[str, Any],
        statuses: List[RateLimitStatus],
    ) -> Dict[str, Any]:
        """
        Return ``additional_headers`` extended with
        ``x-ratelimit-{descriptor_key}-{remaining|limit}-{rate_limit_type}``
        entries. Non-mutating so callers pick their own target dict.
        """
        merged: Dict[str, Any] = dict(additional_headers)
        for status in statuses:
            prefix = f"x-ratelimit-{status['descriptor_key']}"
            merged[f"{prefix}-remaining-{status['rate_limit_type']}"] = status["limit_remaining"]
            merged[f"{prefix}-limit-{status['rate_limit_type']}"] = status["current_limit"]
        return merged

    @staticmethod
    def _stash_value_in_metadata_channels(
        data: Dict[str, Any],
        key: str,
        value: Any,
    ) -> None:
        for channel in ("metadata", "litellm_metadata"):
            existing = data.get(channel)
            if isinstance(existing, dict):
                existing[key] = value
            elif channel == "metadata":
                # ``litellm_metadata`` is owned by the router; don't conjure
                # it here.
                data[channel] = {key: value}

    @classmethod
    def _stash_reservation_in_data(
        cls,
        data: Dict[str, Any],
        estimated_tokens: int,
        reserved_model: Optional[str],
        reserved_scopes: Optional[List[Tuple[str, str]]] = None,
    ) -> None:
        """
        ``reserved_scopes`` is serialized as a list of [key, value] pairs so
        it round-trips through JSON-based metadata transports.
        """
        scopes_payload: Optional[List[List[str]]] = [[k, v] for k, v in reserved_scopes] if reserved_scopes else None

        cls._stash_value_in_metadata_channels(data=data, key=TPM_RESERVED_TOKENS_KEY, value=estimated_tokens)
        if reserved_model:
            cls._stash_value_in_metadata_channels(data=data, key=TPM_RESERVED_MODEL_KEY, value=reserved_model)
        if scopes_payload is not None:
            cls._stash_value_in_metadata_channels(data=data, key=TPM_RESERVED_SCOPES_KEY, value=scopes_payload)

    @staticmethod
    def _lookup_stashed_value(
        kwargs: Any,
        standard_logging_metadata: Optional[Dict[str, Any]],
        key: str,
    ) -> Any:
        """
        Resolve a stashed value from any metadata channel the request data
        can flow through to a callback. Top-level ``kwargs`` is not checked
        because stash keys must never live there.
        """
        candidate: Any = None
        if isinstance(kwargs, dict):
            for channel in ("metadata", "litellm_metadata"):
                channel_dict = kwargs.get(channel)
                if isinstance(channel_dict, dict) and key in channel_dict:
                    candidate = channel_dict.get(key)
                    if candidate is not None:
                        return candidate
            litellm_params = kwargs.get("litellm_params")
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
        candidate = cls._lookup_stashed_value(kwargs, standard_logging_metadata, TPM_RESERVED_TOKENS_KEY)
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
        candidate = cls._lookup_stashed_value(kwargs, standard_logging_metadata, TPM_RESERVED_MODEL_KEY)
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
        candidate = cls._lookup_stashed_value(kwargs, standard_logging_metadata, TPM_RESERVED_SCOPES_KEY)
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
    def _get_reserved_itpm_tokens_from_kwargs(
        cls,
        kwargs: Any,
        standard_logging_metadata: dict[str, Any] | None = None,
    ) -> int:
        candidate = cls._lookup_stashed_value(kwargs, standard_logging_metadata, ITPM_RESERVED_TOKENS_KEY)
        try:
            return int(candidate or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _get_reserved_otpm_tokens_from_kwargs(
        cls,
        kwargs: Any,
        standard_logging_metadata: dict[str, Any] | None = None,
    ) -> int:
        candidate = cls._lookup_stashed_value(kwargs, standard_logging_metadata, OTPM_RESERVED_TOKENS_KEY)
        try:
            return int(candidate or 0)
        except (TypeError, ValueError):
            return 0

    @classmethod
    def _get_reserved_itpm_scopes_from_kwargs(
        cls,
        kwargs: Any,
        standard_logging_metadata: dict[str, Any] | None = None,
    ) -> set[tuple[str, str]]:
        return cls._narrow_reserved_scopes(
            cls._lookup_stashed_value(kwargs, standard_logging_metadata, ITPM_RESERVED_SCOPES_KEY)
        )

    @classmethod
    def _get_reserved_otpm_scopes_from_kwargs(
        cls,
        kwargs: Any,
        standard_logging_metadata: dict[str, Any] | None = None,
    ) -> set[tuple[str, str]]:
        return cls._narrow_reserved_scopes(
            cls._lookup_stashed_value(kwargs, standard_logging_metadata, OTPM_RESERVED_SCOPES_KEY)
        )

    @staticmethod
    def _narrow_reserved_scopes(candidate: Any) -> set[tuple[str, str]]:
        if not isinstance(candidate, list):
            return set()
        scopes: set[tuple[str, str]] = set()
        for entry in candidate:
            if (
                isinstance(entry, (list, tuple))
                and len(entry) == 2
                and isinstance(entry[0], str)
                and isinstance(entry[1], str)
            ):
                scopes.add((entry[0], entry[1]))
        return scopes

    def _resolve_io_token_reconcile_usage(
        self,
        response_obj: Any,
    ) -> tuple[int, int, bool]:
        """
        Resolve ``(billable_input_tokens, completion_tokens, usage_resolved)``
        for ITPM/OTPM reconciliation. Cache-read tokens are excluded from
        billable input -- Bedrock Mantle doesn't count them toward ITPM --
        but they're untouched everywhere else (cost/usage logging still sees
        the full prompt token count).
        """
        usage: Any | None = None
        if isinstance(
            response_obj,
            (ModelResponse, EmbeddingResponse, TextCompletionResponse, BaseLiteLLMOpenAIResponseObject),
        ):
            usage = getattr(response_obj, "usage", None)

        if isinstance(usage, Usage):
            prompt_tokens = usage.prompt_tokens or 0
            completion_tokens = usage.completion_tokens or 0
            cached_tokens = 0
            if usage.prompt_tokens_details is not None:
                cached_tokens = getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
        elif isinstance(usage, dict):
            prompt_tokens = usage.get("prompt_tokens", 0) or 0
            completion_tokens = usage.get("completion_tokens", 0) or 0
            prompt_details = usage.get("prompt_tokens_details") or {}
            cached_tokens = (prompt_details.get("cached_tokens", 0) or 0) if isinstance(prompt_details, dict) else 0
        else:
            return 0, 0, False

        return max(0, prompt_tokens - cached_tokens), completion_tokens, True

    def _build_io_token_reservation_ops(
        self,
        kwargs: Any,
        response_obj: Any,
    ) -> list["RedisPipelineIncrementOperation"]:
        """
        Reconcile project ITPM/OTPM reservations to actual usage on success:
        ITPM to billable input tokens, OTPM to actual completion tokens.
        Reuses ``_build_reservation_aware_tpm_ops``'s delta pattern -- ITPM/OTPM
        are stored in the same ":tokens" cache bucket as combined TPM, just
        under distinct scope keys, so the reservation-aware increment math is
        identical; only the usage fields being reconciled against differ.
        """
        standard_logging_object = kwargs.get("standard_logging_object") or {}
        standard_logging_metadata = standard_logging_object.get("metadata") or {}

        itpm_reserved = self._get_reserved_itpm_tokens_from_kwargs(kwargs, standard_logging_metadata)
        otpm_reserved = self._get_reserved_otpm_tokens_from_kwargs(kwargs, standard_logging_metadata)
        if itpm_reserved <= 0 and otpm_reserved <= 0:
            return []

        billable_input, completion_tokens, usage_resolved = self._resolve_io_token_reconcile_usage(response_obj)
        if not usage_resolved:
            # Usage missing (e.g. a response the SDK couldn't parse) -- keep
            # the reservation rather than guess; it self-expires at the
            # window TTL instead of drifting the counter on a bad guess.
            return []

        ops: list[RedisPipelineIncrementOperation] = []
        if itpm_reserved > 0:
            itpm_scopes = self._get_reserved_itpm_scopes_from_kwargs(kwargs, standard_logging_metadata)
            ops.extend(
                self._build_reservation_aware_tpm_ops(
                    targets=list(itpm_scopes),
                    reserved_scopes=itpm_scopes,
                    actual_tokens=billable_input,
                    reserved_tokens=itpm_reserved,
                )
            )
        if otpm_reserved > 0:
            otpm_scopes = self._get_reserved_otpm_scopes_from_kwargs(kwargs, standard_logging_metadata)
            ops.extend(
                self._build_reservation_aware_tpm_ops(
                    targets=list(otpm_scopes),
                    reserved_scopes=otpm_scopes,
                    actual_tokens=completion_tokens,
                    reserved_tokens=otpm_reserved,
                )
            )
        return ops

    @classmethod
    def _is_reservation_released(
        cls,
        kwargs: Any,
        standard_logging_metadata: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """True if a prior callback already refunded this request's reservation."""
        return bool(cls._lookup_stashed_value(kwargs, standard_logging_metadata, TPM_RESERVATION_RELEASED_KEY))

    @classmethod
    def _get_parallel_slot_acquisition(
        cls,
        kwargs: Any,
        standard_logging_metadata: dict[str, Any] | None = None,
    ) -> ParallelSlotAcquisition | None:
        """The slot acquisition this request's pre-call hook made, if any."""
        candidate = cls._lookup_stashed_value(kwargs, standard_logging_metadata, MAX_PARALLEL_SLOT_ACQUIRED_KEY)
        if not isinstance(candidate, dict):
            return None
        slot_id = candidate.get("slot_id")
        counter_keys = candidate.get("counter_keys")
        if not isinstance(slot_id, str) or not slot_id:
            return None
        if not isinstance(counter_keys, list) or not counter_keys:
            return None
        if not all(isinstance(key, str) and key for key in counter_keys):
            return None
        return ParallelSlotAcquisition(slot_id=slot_id, counter_keys=counter_keys)

    @staticmethod
    def _clear_parallel_slot_marker(data: Any) -> None:
        """
        Remove the acquired-slot marker from every metadata channel a sibling
        callback might read, so one release per acquire is an invariant even
        when multiple callbacks fire for the same request.
        """
        if not isinstance(data, dict):
            return
        for channel in ("metadata", "litellm_metadata"):
            channel_dict = data.get(channel)
            if isinstance(channel_dict, dict):
                channel_dict.pop(MAX_PARALLEL_SLOT_ACQUIRED_KEY, None)
        litellm_params = data.get("litellm_params")
        if isinstance(litellm_params, dict):
            lp_metadata = litellm_params.get("metadata")
            if isinstance(lp_metadata, dict):
                lp_metadata.pop(MAX_PARALLEL_SLOT_ACQUIRED_KEY, None)
        slo = data.get("standard_logging_object")
        if isinstance(slo, dict):
            slo_meta = slo.get("metadata")
            if isinstance(slo_meta, dict):
                slo_meta.pop(MAX_PARALLEL_SLOT_ACQUIRED_KEY, None)

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
        user_api_key_organization_id = standard_logging_metadata.get("user_api_key_org_id")
        user_api_key_project_id = standard_logging_metadata.get("user_api_key_project_id")
        user_api_key_end_user_id = (
            kwargs.get("user") if isinstance(kwargs, dict) else None
        ) or standard_logging_metadata.get("user_api_key_end_user_id")
        agent_id = standard_logging_metadata.get("agent_id")
        session_id = standard_logging_metadata.get("session_id") or standard_logging_metadata.get("trace_id")

        targets: List[Tuple[str, str]] = []
        if user_api_key:
            targets.append(("api_key", user_api_key))
        if user_api_key_user_id:
            targets.append(("user", user_api_key_user_id))
        if user_api_key_team_id:
            targets.append(("team", user_api_key_team_id))
        if user_api_key_team_id and user_api_key_user_id:
            targets.append(("team_member", f"{user_api_key_team_id}:{user_api_key_user_id}"))
        if user_api_key_end_user_id:
            targets.append(("end_user", user_api_key_end_user_id))
        if user_api_key_organization_id:
            targets.append(("organization", user_api_key_organization_id))
        if model_group:
            if user_api_key:
                targets.append(("model_per_key", f"{user_api_key}:{model_group}"))
            if user_api_key_team_id:
                targets.append(("model_per_team", f"{user_api_key_team_id}:{model_group}"))
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

        model_group = get_model_group_from_litellm_kwargs(kwargs)

        # Get total tokens from response
        total_tokens = 0
        if isinstance(
            response_obj,
            (
                ModelResponse,
                EmbeddingResponse,
                TextCompletionResponse,
                BaseLiteLLMOpenAIResponseObject,
            ),
        ):
            _usage = getattr(response_obj, "usage", None)
            total_tokens = self._get_total_tokens_from_usage(usage=_usage, rate_limit_type=rate_limit_type)

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

        litellm_parent_otel_span: Union[Span, None] = _get_parent_otel_span_from_kwargs(kwargs)
        try:
            verbose_proxy_logger.debug("INSIDE parallel request limiter ASYNC SUCCESS LOGGING")

            standard_logging_object = kwargs.get("standard_logging_object") or {}
            standard_logging_metadata = standard_logging_object.get("metadata") or {}
            acquisition = self._get_parallel_slot_acquisition(
                kwargs=kwargs,
                standard_logging_metadata=standard_logging_metadata,
            )
            if acquisition is not None:
                await self._release_parallel_request_slots(
                    acquisition=acquisition,
                    parent_otel_span=litellm_parent_otel_span,
                )
                self._clear_parallel_slot_marker(kwargs)

            pipeline_operations = self._build_success_event_pipeline_operations(
                kwargs=kwargs,
                response_obj=response_obj,
                rate_limit_type=rate_limit_type,
            )
            pipeline_operations.extend(self._build_io_token_reservation_ops(kwargs=kwargs, response_obj=response_obj))

            if pipeline_operations:
                await self.async_increment_tokens_with_ttl_preservation(
                    pipeline_operations=pipeline_operations,
                    parent_otel_span=litellm_parent_otel_span,
                )

        except Exception as e:
            verbose_proxy_logger.exception(f"Error in rate limit success event: {str(e)}")

    async def async_logging_hook(
        self,
        kwargs: dict,
        result: Any,
        call_type: str,
    ) -> Tuple[dict, Any]:
        """
        Mirror the pre-call rate-limit snapshot into the SLP so streaming
        success callbacks see the same ``x-ratelimit-*`` headers the
        non-streaming path writes via ``async_post_call_success_hook``.
        Runs in the earlier of the two callback loops inside
        ``async_success_handler`` so downstream callbacks see the values
        regardless of registration order. Idempotent for non-streaming.
        """
        self._mirror_ratelimit_response_into_logging_payload(
            kwargs=kwargs,
            response_obj=result,
        )
        return kwargs, result

    def _mirror_ratelimit_response_into_logging_payload(
        self,
        kwargs: Any,
        response_obj: Any,
    ) -> None:
        """
        Copy the stashed ``RateLimitResponse`` into the SLP's
        ``hidden_params.additional_headers`` and the response object's
        ``_hidden_params.additional_headers`` (when the latter is a dict).
        """
        if not isinstance(kwargs, dict):
            return

        standard_logging_object = kwargs.get("standard_logging_object")
        standard_logging_metadata: Optional[Dict[str, Any]] = None
        if isinstance(standard_logging_object, dict):
            slp_metadata = standard_logging_object.get("metadata")
            if isinstance(slp_metadata, dict):
                standard_logging_metadata = slp_metadata

        statuses = self._narrow_ratelimit_statuses(
            self._lookup_stashed_value(
                kwargs=kwargs,
                standard_logging_metadata=standard_logging_metadata,
                key=RATE_LIMIT_RESPONSE_KEY,
            )
        )
        if not statuses:
            return

        if isinstance(standard_logging_object, dict):
            hidden_params = standard_logging_object.get("hidden_params")
            if not isinstance(hidden_params, dict):
                hidden_params = {}
            existing = hidden_params.get("additional_headers")
            hidden_params["additional_headers"] = self._merge_ratelimit_statuses_into_additional_headers(
                additional_headers=existing if isinstance(existing, dict) else {},
                statuses=statuses,
            )
            standard_logging_object["hidden_params"] = hidden_params

        response_hidden = getattr(response_obj, "_hidden_params", None)
        if isinstance(response_hidden, dict):
            existing = response_hidden.get("additional_headers")
            response_hidden["additional_headers"] = self._merge_ratelimit_statuses_into_additional_headers(
                additional_headers=existing if isinstance(existing, dict) else {},
                statuses=statuses,
            )

    @staticmethod
    def _narrow_ratelimit_statuses(stashed: Any) -> List[RateLimitStatus]:
        """
        Narrow a stashed ``RateLimitResponse``-shaped dict to a typed
        ``statuses`` list. Entries missing any header-write field are dropped;
        an empty list means "nothing to mirror".
        """
        if not isinstance(stashed, dict):
            return []
        raw_statuses = stashed.get("statuses")
        if not isinstance(raw_statuses, list):
            return []
        narrowed: List[RateLimitStatus] = []
        for entry in raw_statuses:
            if not isinstance(entry, dict):
                continue
            descriptor_key = entry.get("descriptor_key")
            rate_limit_type = entry.get("rate_limit_type")
            current_limit = entry.get("current_limit")
            limit_remaining = entry.get("limit_remaining")
            if (
                isinstance(descriptor_key, str)
                and rate_limit_type in ("requests", "tokens", "max_parallel_requests")
                and isinstance(current_limit, int)
                and isinstance(limit_remaining, int)
            ):
                narrowed.append(
                    RateLimitStatus(
                        code=entry.get("code", "OK") if isinstance(entry.get("code"), str) else "OK",
                        current_limit=current_limit,
                        limit_remaining=limit_remaining,
                        rate_limit_type=rate_limit_type,
                        descriptor_key=descriptor_key,
                    )
                )
        return narrowed

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
            litellm_parent_otel_span: Union[Span, None] = _get_parent_otel_span_from_kwargs(kwargs)
            standard_logging_object = kwargs.get("standard_logging_object") or {}
            standard_logging_metadata = standard_logging_object.get("metadata") or {}

            pipeline_operations: List[RedisPipelineIncrementOperation] = []

            acquisition = self._get_parallel_slot_acquisition(
                kwargs=kwargs,
                standard_logging_metadata=standard_logging_metadata,
            )
            if acquisition is not None:
                await self._release_parallel_request_slots(
                    acquisition=acquisition,
                    parent_otel_span=litellm_parent_otel_span,
                )
                self._clear_parallel_slot_marker(kwargs)

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
                verbose_proxy_logger.debug(f"Releasing reserved TPM tokens on failure: {reserved_tokens}")
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

            # Refund project ITPM/OTPM reservations the same way -- full
            # refund, since a failed call has no billable usage to reconcile
            # against.
            itpm_reserved = (
                0 if already_released else self._get_reserved_itpm_tokens_from_kwargs(kwargs, standard_logging_metadata)
            )
            if itpm_reserved > 0:
                itpm_scopes = self._get_reserved_itpm_scopes_from_kwargs(kwargs, standard_logging_metadata)
                pipeline_operations.extend(
                    self._build_reservation_aware_tpm_ops(
                        targets=list(itpm_scopes),
                        reserved_scopes=itpm_scopes,
                        actual_tokens=0,
                        reserved_tokens=itpm_reserved,
                    )
                )

            otpm_reserved = (
                0 if already_released else self._get_reserved_otpm_tokens_from_kwargs(kwargs, standard_logging_metadata)
            )
            if otpm_reserved > 0:
                otpm_scopes = self._get_reserved_otpm_scopes_from_kwargs(kwargs, standard_logging_metadata)
                pipeline_operations.extend(
                    self._build_reservation_aware_tpm_ops(
                        targets=list(otpm_scopes),
                        reserved_scopes=otpm_scopes,
                        actual_tokens=0,
                        reserved_tokens=otpm_reserved,
                    )
                )

            if pipeline_operations:
                await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                    increment_list=pipeline_operations,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                )
            if reserved_tokens > 0 or itpm_reserved > 0 or otpm_reserved > 0:
                self._mark_reservation_released(kwargs)
        except Exception as e:
            verbose_proxy_logger.exception(f"Error in rate limit failure event: {str(e)}")

    async def async_release_max_parallel_requests_on_disconnect(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        request_data: dict | None = None,
    ) -> None:
        """
        Release the api-key ``max_parallel_requests`` slot that
        ``async_pre_call_hook`` acquired, for a request that ended without
        either logging callback firing.

        The slot is normally released by ``async_log_success_event`` (natural
        stream completion) or ``async_log_failure_event`` (LLM error). When a
        client cancels a stream mid-flight, the cancellation surfaces as
        ``asyncio.CancelledError`` / ``GeneratorExit`` and neither callback
        runs, so without this the slot leaks per cancelled stream until its
        TTL prunes it. ``request_data`` carries the stashed acquisition;
        its presence (not the key object's current max_parallel_requests
        configuration, which can change mid-request) decides whether there
        is anything to release.
        """
        acquisition = self._get_parallel_slot_acquisition(kwargs=request_data)
        if acquisition is None:
            return

        await self._release_parallel_request_slots(
            acquisition=acquisition,
            parent_otel_span=None,
        )
        self._clear_parallel_slot_marker(request_data)

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response):
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
                    isinstance(_hidden_params, BaseModel) or isinstance(_hidden_params, dict)
                ):
                    if isinstance(_hidden_params, BaseModel):
                        _hidden_params = _hidden_params.model_dump()

                    _additional_headers = self._merge_ratelimit_statuses_into_additional_headers(
                        additional_headers=_hidden_params.get("additional_headers", {}) or {},
                        statuses=litellm_proxy_rate_limit_response["statuses"],
                    )

                    setattr(
                        response,
                        "_hidden_params",
                        {**_hidden_params, "additional_headers": _additional_headers},
                    )

        except Exception as e:
            verbose_proxy_logger.exception(f"Error in rate limit post-call hook: {str(e)}")

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: Optional[str] = None,
    ) -> None:
        """
        Release the parallel-request slot and any TPM/ITPM/OTPM reservation
        when the request is rejected after the pre-call hook acquired them
        but before the LLM call ran (e.g. a downstream guardrail/auth hook
        raised). Without this, those resources are stranded —
        async_log_failure_event is a litellm completion-level callback and
        never fires for proxy-side rejections, so a leaked slot would occupy
        the gauge for the full PARALLEL_REQUEST_SLOT_TTL_SECONDS.

        Idempotent: the slot release clears the acquisition marker (and slot
        removal is a no-op ZREM on a second run), and the reservation refund
        is guarded by TPM_RESERVATION_RELEASED_KEY (shared across the
        TPM/ITPM/OTPM buckets) — if both this hook and
        async_log_failure_event end up running in the same flow, only the
        first release/refund applies.
        """
        try:
            acquisition = self._get_parallel_slot_acquisition(kwargs=request_data)
            if acquisition is not None:
                await self._release_parallel_request_slots(
                    acquisition=acquisition,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )
                self._clear_parallel_slot_marker(request_data)

            if self._is_reservation_released(kwargs=request_data):
                return
            reserved_tokens = self._get_reserved_tokens_from_kwargs(kwargs=request_data)
            itpm_reserved = self._get_reserved_itpm_tokens_from_kwargs(kwargs=request_data)
            otpm_reserved = self._get_reserved_otpm_tokens_from_kwargs(kwargs=request_data)
            if reserved_tokens <= 0 and itpm_reserved <= 0 and otpm_reserved <= 0:
                return

            ops: List[RedisPipelineIncrementOperation] = []

            if reserved_tokens > 0:
                # Refund directly against the descriptors we reserved
                # against — the pre-call hook stashes them in the
                # request-data metadata channels before success/failure
                # callbacks run. Excludes the project ITPM/OTPM descriptors,
                # which are reserved from different amounts and refunded
                # separately below.
                stashed = self._lookup_stashed_value(
                    kwargs=request_data,
                    standard_logging_metadata=None,
                    key=RATE_LIMIT_DESCRIPTORS_KEY,
                )
                descriptors: List[RateLimitDescriptor] = stashed if isinstance(stashed, list) else []
                for descriptor in descriptors:
                    if descriptor["key"] in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY):
                        continue
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

            if itpm_reserved > 0:
                itpm_scopes = self._get_reserved_itpm_scopes_from_kwargs(kwargs=request_data)
                ops.extend(
                    self._build_reservation_aware_tpm_ops(
                        targets=list(itpm_scopes),
                        reserved_scopes=itpm_scopes,
                        actual_tokens=0,
                        reserved_tokens=itpm_reserved,
                    )
                )
            if otpm_reserved > 0:
                otpm_scopes = self._get_reserved_otpm_scopes_from_kwargs(kwargs=request_data)
                ops.extend(
                    self._build_reservation_aware_tpm_ops(
                        targets=list(otpm_scopes),
                        reserved_scopes=otpm_scopes,
                        actual_tokens=0,
                        reserved_tokens=otpm_reserved,
                    )
                )

            if ops:
                verbose_proxy_logger.debug(
                    f"Releasing reserved tokens on proxy-level rejection: "
                    f"tpm={reserved_tokens}, itpm={itpm_reserved}, otpm={otpm_reserved}"
                )
                await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                    increment_list=ops,
                    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                )
            self._mark_reservation_released(request_data)
        except Exception as e:
            verbose_proxy_logger.exception(f"Error releasing TPM reservation on post-call failure: {e}")
        return None
