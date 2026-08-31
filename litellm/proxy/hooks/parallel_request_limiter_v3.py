"""
This is a rate limiter implementation based on a similar one by Envoy proxy.

This is currently in development and not yet ready for production.
"""

import asyncio
import binascii
import os
import uuid
from collections.abc import Callable, Mapping, Sequence, Set
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Final,
    Literal,
    Protocol,
    TypeAlias,
    TypedDict,
)

from typing_extensions import NotRequired, ReadOnly

from litellm import DualCache
from litellm._logging import verbose_proxy_logger
from litellm.constants import DYNAMIC_RATE_LIMIT_ERROR_THRESHOLD_PER_MINUTE, INTERNAL_CALL_ORIGIN_METADATA_KEY
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_str_from_messages,
)
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    ESTIMATED_OUTPUT_TOKENS_FIELD,
    get_estimated_output_tokens,
    get_key_tag_rpm_limit,
    get_model_rate_limit_from_metadata,
)
from litellm.proxy.auth.budget_throttle import throttled_limit
from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body
from litellm.proxy.common_utils.proxy_rate_limit_error import (
    ProxyRateLimitError,
    map_v3_rate_limit_type,
)
from litellm.proxy.hooks.batch_enqueued_tokens import (
    BATCH_ENQUEUED_REFUND_STATUSES,
    BatchEnqueuedTokenReservation,
    BatchEnqueuedTokenStore,
    batch_response_view,
    canonical_provider_batch_id,
)
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.types.caching import RedisPipelineIncrementOperation
from litellm.types.llms.openai import BaseLiteLLMOpenAIResponseObject, ResponseAPIUsage
from litellm.types.utils import (
    CallTypes,
    EmbeddingResponse,
    ModelResponse,
    RerankResponse,
    TextCompletionResponse,
    Usage,
)

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache
    from litellm.types.agents import AgentResponse
    from litellm.types.caching import RedisPipelineIncrementOperation

    Span = _Span | Any
    InternalUsageCache = _InternalUsageCache
else:
    Span = Any
    InternalUsageCache = Any


BATCH_RATE_LIMITER_SCRIPT: Final = """
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

CHECK_AND_INCREMENT_BY_N_SCRIPT: Final = """
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
-- Return on success:
--     { 0, new_counter_1, window_start_1, new_counter_2, window_start_2, ... }
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

    local blocked
    if increment > 0 then
        blocked = current_counter + increment > limit
    else
        blocked = current_counter >= limit
    end
    if blocked then
        return { 1, i, current_counter, limit }
    end

    descriptor_state[i] = { window_expired, current_counter, window_start }
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
    local active_window_start

    if window_expired then
        active_window_start = now
        redis.call('SET', window_key, tostring(now))
        redis.call('SET', counter_key, increment)
        redis.call('EXPIRE', window_key, window_size)
        if ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(results, increment)
    else
        active_window_start = tonumber(descriptor_state[i][3])
        local new_counter = redis.call('INCRBY', counter_key, increment)
        local current_ttl = redis.call('TTL', counter_key)
        if current_ttl == -1 and ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(results, new_counter)
    end
    table.insert(results, active_window_start)
end

return results
"""

WINDOW_GUARDED_TOKEN_INCREMENT_SCRIPT: Final = """
local results = {}
for i = 1, #KEYS, 2 do
    local window_key = KEYS[i]
    local counter_key = KEYS[i + 1]
    local arg_base = ((i - 1) / 2) * 3 + 1
    local expected_window_start = ARGV[arg_base]
    local increment = tonumber(ARGV[arg_base + 1])
    local ttl = tonumber(ARGV[arg_base + 2])
    local active_window_start = redis.call('GET', window_key)

    if active_window_start and active_window_start == expected_window_start then
        local new_counter = redis.call('INCRBY', counter_key, increment)
        local current_ttl = redis.call('TTL', counter_key)
        if current_ttl == -1 and ttl > 0 then
            redis.call('EXPIRE', counter_key, ttl)
        end
        table.insert(results, 1)
        table.insert(results, new_counter)
    else
        table.insert(results, 0)
        table.insert(results, tonumber(redis.call('GET', counter_key) or 0))
    end
end
return results
"""

PARALLEL_ACQUIRE_SCRIPT: Final = """
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

PARALLEL_RELEASE_SCRIPT: Final = """
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

PARALLEL_COUNT_SCRIPT: Final = """
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

TOKEN_INCREMENT_SCRIPT: Final = """
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
REDIS_CLUSTER_SLOTS: Final = 16384
REDIS_NODE_HASHTAG_NAME: Final = "all_keys"

# TPM token reservation tuning constants.
# When max_tokens is not specified in the request we still need to reserve
# *some* output budget; these define that fallback estimate.
DEFAULT_MAX_TOKENS_ESTIMATE: Final = 4096
DEFAULT_CHARS_PER_TOKEN: Final = 4
# Fraction of the available output budget reserved as the upfront floor when
# the request omits max_tokens. Applied to both DEFAULT_MAX_TOKENS_ESTIMATE
# (baseline floor) and to the smallest configured TPM limit (capped floor for
# small per-tenant TPM caps).
_TPM_FLOOR_FRACTION: Final = 4
# Both embeddings and the Responses API put their prompt in data["input"],
# but only embeddings have no output tokens. Every "is this an embedding"
# check on data["input"] must exclude these call types, or a Responses call
# gets misclassified as an embedding and skips output-token reservation/caps.
RESPONSES_API_CALL_TYPES: Final = ("aresponses", "responses")
EMBEDDING_API_CALL_TYPES: Final = ("aembedding", "embedding")
TEXT_COMPLETION_API_CALL_TYPES: Final = ("atext_completion", "text_completion")
RERANK_API_CALL_TYPES: Final = (CallTypes.rerank.value, CallTypes.arerank.value)
GOOGLE_GENAI_NATIVE_CALL_TYPES: Final = (
    CallTypes.generate_content.value,
    CallTypes.agenerate_content.value,
    CallTypes.generate_content_stream.value,
    CallTypes.agenerate_content_stream.value,
)
RESPONSES_API_MIN_OUTPUT_TOKENS: Final = 16
# litellm.token_counter has no per-type handling for "input_audio" content
# blocks (unlike images, which use use_default_image_token_count) -- it
# silently contributes 0 tokens for them. When the block carries a base64
# payload, the estimate is derived from the decoded byte count; when the
# block is a reference without a payload (or the payload is missing), this
# flat per-block floor is used instead.
DEFAULT_AUDIO_TOKEN_ESTIMATE: Final = 300
# Conservative bytes-per-token assumption for size-based audio estimation:
# equivalent to 8 kHz mono PCM-16 (16 000 bytes/s) at 10 tokens/s. Choosing
# the lowest reasonable bitrate means we never under-reserve for higher-
# quality audio recorded at the same wall-clock duration.
_AUDIO_BYTES_PER_TOKEN: Final = 1600
# Descriptor "key" values for project-scoped ITPM/OTPM. Distinct from
# "model_per_project" (the combined-TPM descriptor) so both can be enforced
# on the same project+model simultaneously without colliding on cache keys.
PROJECT_ITPM_DESCRIPTOR_KEY: Final = "model_per_project_itpm"
PROJECT_OTPM_DESCRIPTOR_KEY: Final = "model_per_project_otpm"
# How long an acquired slot counts toward the in-flight total before it is
# considered leaked (worker crashed without any release callback firing) and
# pruned. Also the longest request duration the gauge can track: a request
# running longer than this stops occupying its slot.
PARALLEL_REQUEST_SLOT_TTL_SECONDS: Final = 3600


CacheCounterValue: TypeAlias = int | float | str | bytes

CacheCounterValues: TypeAlias = Sequence[CacheCounterValue | None]

ParallelGaugeCacheValue: TypeAlias = dict[str, object] | int | float | str | bytes


class RateLimitDescriptorRateLimitObject(TypedDict, total=False):
    requests_per_unit: int | None
    tokens_per_unit: int | None
    max_parallel_requests: int | None
    window_size: int | None


class RateLimitDescriptor(TypedDict):
    key: str
    value: str
    rate_limit: RateLimitDescriptorRateLimitObject | None


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
    # Only populated by the atomic_check_and_increment_by_n path. A caller
    # matching a status back to its descriptor must key on (descriptor_key,
    # descriptor_value) when this is present, not descriptor_key alone --
    # e.g. a batch charging several models' project ITPM/OTPM in one call
    # produces multiple statuses sharing the same descriptor_key.
    descriptor_value: NotRequired[ReadOnly[str]]


class RateLimitResponse(TypedDict):
    overall_code: str
    statuses: list[RateLimitStatus]
    reservation_windows: NotRequired[ReadOnly[frozenset[tuple[str, str, Literal["redis", "local"]]]]]


class ReservationAwareIncrementOperation(RedisPipelineIncrementOperation):
    window_key: NotRequired[str]
    expected_window_start: NotRequired[str]
    reservation_backend: NotRequired[Literal["redis", "local"]]


class RateLimitResponseWithDescriptors(TypedDict):
    descriptors: list[RateLimitDescriptor]
    response: RateLimitResponse


class _RateLimitDescriptorSink(Protocol):
    def append(self, descriptor: RateLimitDescriptor, /) -> None: ...


class WindowKeyMetadata(TypedDict):
    requests_limit: int | None
    tokens_limit: int | None
    window_size: int
    descriptor_key: str


class AtomicCounterMeta(TypedDict):
    descriptor_key: str
    descriptor_value: ReadOnly[str]
    current_limit: int
    rate_limit_type: Literal["requests", "tokens"]
    window_key: str
    counter_key: str
    increment: int
    ttl: int
    window_size: int


class AtomicCounterState(TypedDict):
    window_expired: bool
    current: int
    window_start: ReadOnly[str]


DescriptorAtomicGroup: TypeAlias = tuple[list[str], list[int], list[AtomicCounterMeta]]


class CallTypeRateLimiter(Protocol):
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],
        call_type: str,
    ) -> Exception | str | dict[str, object] | None: ...


@dataclass(slots=True)
class RequestRateLimiterStash:
    """
    Per-request bookkeeping the pre-call hook hands to the success/failure/
    disconnect callbacks. Lives on a ContextVar instead of the request body so
    it never reaches provider-facing ``metadata`` channels.

    A single mutable instance is shared by every context forked from the
    request task (the SDK call, streaming generators, and the logging worker's
    captured context all see the same object), which is what makes the
    ``reservation_released`` flag and ``parallel_slot`` clearing effective
    across sibling callbacks: the first release wins, later callbacks observe
    the cleared state.

    Because the stash is context-inherited, nested LiteLLM calls made inside
    the request (LLM-judge guardrails, silent experiments) would also see it
    from their own logging callbacks. ``owner_litellm_call_id`` pins the stash
    to the proxy request's ``litellm_call_id`` so those callbacks can tell the
    owning request's events apart from a nested call's: router retries and
    fallbacks reuse the request's call id and keep access, while nested calls
    mint fresh ids and are ignored.
    """

    owner_litellm_call_id: str | None = None
    rate_limit_response: RateLimitResponse | None = None
    parallel_slot: ParallelSlotAcquisition | None = None
    reserved_tokens: int = 0
    reserved_model: str | None = None
    reserved_scopes: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    itpm_reserved_tokens: int = 0
    itpm_reserved_scopes: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    itpm_reserved_window_identities: frozenset[tuple[str, str, Literal["redis", "local"]]] = field(
        default_factory=frozenset
    )
    otpm_reserved_tokens: int = 0
    otpm_reserved_scopes: frozenset[tuple[str, str]] = field(default_factory=frozenset)
    otpm_reserved_window_identities: frozenset[tuple[str, str, Literal["redis", "local"]]] = field(
        default_factory=frozenset
    )
    batch_enqueued_reservation: BatchEnqueuedTokenReservation | None = None
    reservation_released: bool = False


_request_stash: Final[ContextVar[RequestRateLimiterStash | None]] = ContextVar(
    "litellm_v3_rate_limiter_request_stash", default=None
)


def get_request_stash() -> RequestRateLimiterStash | None:
    return _request_stash.get()


def get_or_create_request_stash() -> RequestRateLimiterStash:
    stash = _request_stash.get()
    if stash is None:
        stash = RequestRateLimiterStash()
        _request_stash.set(stash)
    return stash


def claim_request_stash_for_data(data: dict) -> RequestRateLimiterStash:
    stash: Final = get_or_create_request_stash()
    owner_call_id: Final = data.get("litellm_call_id")
    if isinstance(owner_call_id, str):
        stash.owner_litellm_call_id = owner_call_id
    return stash


def get_request_stash_for_call(litellm_call_id: str | None) -> RequestRateLimiterStash | None:
    stash: Final = _request_stash.get()
    if stash is None:
        return None
    if stash.owner_litellm_call_id is None or litellm_call_id is None:
        return stash
    return stash if litellm_call_id == stash.owner_litellm_call_id else None


def _call_id_from_callback_kwargs(kwargs: object) -> str | None:
    if not isinstance(kwargs, dict):
        return None
    call_id: Final = kwargs.get("litellm_call_id")
    return call_id if isinstance(call_id, str) else None


def _parse_output_cap_value(raw_value: object) -> int | None:
    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float, str)):
        return None
    try:
        return int(float(raw_value))
    except (ValueError, OverflowError):
        return None


class _PROXY_MaxParallelRequestsHandler_v3(CustomLogger):
    def __init__(
        self,
        internal_usage_cache: InternalUsageCache,
        time_provider: Callable[[], datetime] | None = None,
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
            self.window_guarded_token_increment_script = (
                self.internal_usage_cache.dual_cache.redis_cache.async_register_script(
                    WINDOW_GUARDED_TOKEN_INCREMENT_SCRIPT
                )
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
            self.window_guarded_token_increment_script = None
            self.parallel_acquire_script = None
            self.parallel_release_script = None
            self.parallel_count_script = None

        self.window_size = int(os.getenv("LITELLM_RATE_LIMIT_WINDOW_SIZE", 60))

        # When disabled, TPM is enforced post-call from actual usage (pre-v1.82
        # behavior) instead of reserving an estimated budget upfront, shedding
        # the extra per-request Redis Lua round-trip and the global-lock
        # in-memory fallback that the reservation path incurs.
        self.tpm_reservation_enabled = os.getenv("LITELLM_TPM_TOKEN_RESERVATION_ENABLED", "true").lower() == "true"

        # Batch rate limiter (lazy loaded)
        self._batch_rate_limiter: CallTypeRateLimiter | None = None
        self.batch_enqueued_token_store = BatchEnqueuedTokenStore(internal_usage_cache=internal_usage_cache)

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

    def _get_batch_rate_limiter(self) -> CallTypeRateLimiter | None:
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
                verbose_proxy_logger.debug("Could not load batch rate limiter: %s", e)
        return self._batch_rate_limiter

    def _get_current_time(self) -> datetime:
        """Return the current time for rate limiting calculations."""
        return self._time_provider()

    @staticmethod
    def no_max_tokens_output_floor(
        min_configured_tpm_limit: int | None,
    ) -> int:
        """Output-budget floor used when the request omits max_tokens.

        Capped at a fraction of the smallest configured TPM limit so a small
        per-tenant cap can't be tripped by the floor alone. Returns the
        baseline floor when no limit is provided.
        """
        baseline: Final = DEFAULT_MAX_TOKENS_ESTIMATE // _TPM_FLOOR_FRACTION
        if min_configured_tpm_limit is None:
            return baseline
        return min(baseline, max(1, min_configured_tpm_limit // _TPM_FLOOR_FRACTION))

    @staticmethod
    def _is_embedding_request(data: object, call_type: str | None) -> bool:
        if call_type in EMBEDDING_API_CALL_TYPES:
            return True
        if call_type in RESPONSES_API_CALL_TYPES:
            return False
        if call_type:
            return False
        if not isinstance(data, dict):
            return False
        return data.get("input") is not None

    @staticmethod
    def _translate_google_genai_native_request(
        data: object,
        call_type: str | None,
    ) -> Mapping[str, object] | None:
        contents: Final = data.get("contents") if isinstance(data, dict) else None
        if (
            not isinstance(data, dict)
            or call_type not in GOOGLE_GENAI_NATIVE_CALL_TYPES
            or not isinstance(contents, (dict, list))
        ):
            return None
        from litellm.google_genai.adapters.transformation import GoogleGenAIAdapter

        config: Final = data.get("config") if "config" in data else data.get("generationConfig")
        return GoogleGenAIAdapter().translate_generate_content_to_completion(
            model=data.get("model") if isinstance(data.get("model"), str) else "",
            contents=contents,
            config=config if isinstance(config, dict) else None,
            systemInstruction=data.get("systemInstruction"),
            system_instruction=data.get("system_instruction"),
            tools=data.get("tools"),
            toolConfig=data.get("toolConfig"),
            tool_config=data.get("tool_config"),
        )

    @staticmethod
    def _get_explicit_output_cap(data: object, call_type: str | None) -> int | None:
        if not isinstance(data, dict):
            return None
        if call_type in GOOGLE_GENAI_NATIVE_CALL_TYPES:
            config: Final = data.get("config") if "config" in data else data.get("generationConfig")
            google_cap_values: Final = tuple(
                parsed
                for field in ("maxOutputTokens", "max_output_tokens")
                if isinstance(config, dict)
                for parsed in (_parse_output_cap_value(config.get(field)),)
                if parsed is not None
            )
            return max(google_cap_values, default=None)
        if call_type in RESPONSES_API_CALL_TYPES:
            responses_cap: Final = _parse_output_cap_value(data.get("max_output_tokens"))
            if responses_cap is None:
                return None
            return max(RESPONSES_API_MIN_OUTPUT_TOKENS, responses_cap)
        if call_type in EMBEDDING_API_CALL_TYPES:
            return None
        fields: Final = (
            ("max_tokens", "max_completion_tokens")
            if call_type
            else ("max_tokens", "max_completion_tokens", "max_output_tokens")
        )
        output_cap_values: Final = tuple(
            parsed for field in fields for parsed in (_parse_output_cap_value(data.get(field)),) if parsed is not None
        )
        return max(output_cap_values, default=None)

    @classmethod
    def _has_explicit_output_cap(cls, data: object, call_type: str | None) -> bool:
        """Whether the caller explicitly set an output-token cap.

        Checked via ``is not None`` (not truthiness) so an explicit 0 --
        a legitimate zero-output request -- counts as explicit.
        """
        return cls._get_explicit_output_cap(data, call_type) is not None

    @staticmethod
    def get_output_candidate_count(data: object, call_type: str | None = None) -> int:
        if not isinstance(data, Mapping):
            return 1
        config: Final = (
            (data.get("config") if "config" in data else data.get("generationConfig"))
            if call_type in GOOGLE_GENAI_NATIVE_CALL_TYPES
            else None
        )
        candidate_values: Final = (
            data.get("n"),
            data.get("best_of"),
            config.get("candidateCount") if isinstance(config, dict) else None,
            config.get("candidate_count") if isinstance(config, dict) else None,
        )
        candidate_count = 1  # rebind-ok: running maximum across candidate-count aliases
        for value in candidate_values:
            try:
                candidate_count = max(candidate_count, int(value or 1))
            except (TypeError, ValueError, OverflowError):
                continue
        return candidate_count

    @staticmethod
    def _apply_implicit_output_cap(
        data: object,
        min_configured_limit: int | None,
        call_type: str | None,
        configured_output_tokens: int | None = None,
    ) -> None:
        """Hard-cap generation length when the request has no explicit cap.

        Guards against an unbounded response overshooting a small TPM/OTPM
        budget before post-call reconciliation runs. Skips requests that
        already set an explicit cap and embeddings, which have no generation
        budget. The Responses API only honors ``max_output_tokens`` (its
        underlying chat-completion transformation ignores ``max_tokens``), so
        the cap must be written to that field for Responses call types.

        ``configured_output_tokens`` is the operator-declared per-tenant
        estimate; when it exceeds the safety floor, the cap is raised to that
        value instead of clamping every tenant to the same floor.
        """
        if not isinstance(data, dict):
            return
        base_capped_floor: Final = _PROXY_MaxParallelRequestsHandler_v3.no_max_tokens_output_floor(min_configured_limit)
        capped_floor: Final = (
            max(base_capped_floor, RESPONSES_API_MIN_OUTPUT_TOKENS)
            if call_type in RESPONSES_API_CALL_TYPES
            else base_capped_floor
        )
        baseline_floor: Final = DEFAULT_MAX_TOKENS_ESTIMATE // _TPM_FLOOR_FRACTION
        is_embedding: Final = _PROXY_MaxParallelRequestsHandler_v3._is_embedding_request(data, call_type)
        if (
            capped_floor >= baseline_floor
            or _PROXY_MaxParallelRequestsHandler_v3._has_explicit_output_cap(data, call_type)
            or is_embedding
        ):
            return
        effective_cap: Final = max(capped_floor, configured_output_tokens or 0)
        if call_type in GOOGLE_GENAI_NATIVE_CALL_TYPES:
            config_field: Final = "config" if "config" in data or "generationConfig" not in data else "generationConfig"
            config: Final = data.get(config_field)
            if config is None or isinstance(config, dict):
                data[config_field] = {  # rebind-ok: routed request needs cap  # mutable-ok: downstream needs dict
                    **(config or {}),  # mutable-ok: downstream native routing requires a mutable request config
                    "maxOutputTokens": effective_cap,
                }
            return
        cap_field: Final = "max_output_tokens" if call_type in RESPONSES_API_CALL_TYPES else "max_tokens"
        existing_cap: Final = data.get(cap_field)
        if existing_cap is None or effective_cap < existing_cap:
            data[cap_field] = effective_cap  # rebind-ok: downstream routing requires the bounded output cap

    def _estimate_tokens_for_request(
        self,
        data: dict,
        model: str | None = None,
        min_configured_tpm_limit: int | None = None,
        call_type: str | None = None,
        configured_output_tokens: int | None = None,
    ) -> int:
        """
        Estimate total tokens this request will consume so we can reserve them
        upfront (input + output budget):
        estimated = input_tokens + max_tokens.

        Supports chat (messages), completions (prompt), embeddings (input),
        and the Responses API (also `input`, disambiguated from embeddings
        via ``call_type``).

        ``min_configured_tpm_limit`` is the smallest ``tokens_per_unit`` among
        the TPM-bearing descriptors this request will be charged against. When
        provided, the no-``max_tokens`` output-budget floor is capped at a
        fraction of that limit so small TPM caps remain usable. Omit to
        preserve the unconstrained floor.

        ``configured_output_tokens`` is the operator-declared estimate resolved
        from key or team metadata. When provided it replaces the heuristic
        floor entirely, so the reservation reflects what this tenant's model
        actually emits rather than one constant shared by every tenant.
        """
        estimated_input_tokens, max_tokens_estimate = self._estimate_input_and_output_tokens(
            data=data,
            min_configured_tpm_limit=min_configured_tpm_limit,
            call_type=call_type,
            configured_output_tokens=configured_output_tokens,
        )
        total_estimated: Final = estimated_input_tokens + max_tokens_estimate

        verbose_proxy_logger.debug(
            "TPM reservation estimate: input=%s, max_tokens=%s, total=%s",
            estimated_input_tokens,
            max_tokens_estimate,
            total_estimated,
        )

        return total_estimated

    def _estimate_input_and_output_tokens(
        self,
        data: object,
        min_configured_tpm_limit: int | None = None,
        call_type: str | None = None,
        configured_output_tokens: int | None = None,
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

        ``call_type`` disambiguates embeddings from the Responses API: both
        put their prompt in ``data["input"]``, but only embeddings have no
        output tokens. Unset (the default) preserves the historical
        "any `input` means zero output" behavior for callers that don't have
        a call type to pass.

        ``configured_output_tokens`` is the operator-declared estimate resolved
        from key or team metadata. When provided it replaces the heuristic
        floor entirely, so the reservation reflects what this tenant's model
        actually emits rather than one constant shared by every tenant.
        """
        if not isinstance(data, dict):
            return 0, 0
        translated_data: Final = self._translate_google_genai_native_request(data, call_type)
        estimable_data: Final = translated_data if translated_data is not None else data
        selected_fields: Final[tuple[object | None, object | None, object | None]] = (
            (None, None, estimable_data.get("input"))
            if call_type in RESPONSES_API_CALL_TYPES or call_type in EMBEDDING_API_CALL_TYPES
            else (None, estimable_data.get("prompt"), None)
            if call_type in TEXT_COMPLETION_API_CALL_TYPES
            else (estimable_data.get("messages"), None, None)
            if call_type
            else (
                estimable_data.get("messages"),
                estimable_data.get("prompt"),
                estimable_data.get("input"),
            )
        )
        messages, prompt, input_text = selected_fields

        total_chars: Final = (
            len(get_str_from_messages(messages))
            if isinstance(messages, list) and messages
            else len(prompt)
            if isinstance(prompt, str)
            else sum(len(str(item)) for item in prompt)
            if isinstance(prompt, list)
            else len(input_text)
            if isinstance(input_text, str)
            else sum(len(str(item)) for item in input_text)
            if isinstance(input_text, list)
            else 0
        )

        estimated_input_tokens: Final = max(1, total_chars // DEFAULT_CHARS_PER_TOKEN) if total_chars > 0 else 0

        explicit_max_tokens: Final = self._get_explicit_output_cap(data, call_type)
        is_embedding: Final = self._is_embedding_request(data, call_type)

        base_output_floor: Final = self.no_max_tokens_output_floor(min_configured_tpm_limit)
        output_floor: Final = (
            max(base_output_floor, RESPONSES_API_MIN_OUTPUT_TOKENS)
            if call_type in RESPONSES_API_CALL_TYPES
            else base_output_floor
        )
        max_tokens_estimate: Final = (
            0
            if is_embedding or (explicit_max_tokens is None and total_chars == 0 and configured_output_tokens is None)
            else explicit_max_tokens
            if explicit_max_tokens is not None
            else configured_output_tokens
            if configured_output_tokens is not None
            else max(estimated_input_tokens, output_floor)
        )

        return estimated_input_tokens, max_tokens_estimate * self.get_output_candidate_count(data, call_type)

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
        keys: list[str],
        now_int: int,
        window_size: int,
    ) -> CacheCounterValues:
        """
        Implement sliding window rate limiting logic using in-memory cache operations.
        This follows the same logic as the Redis Lua script but uses async cache operations.
        """
        results: Final[list[CacheCounterValue | None]] = []

        # Process each window/counter pair
        for i in range(0, len(keys), 2):
            window_key = keys[i]
            counter_key = keys[i + 1]
            increment_value = 1

            # Get the window start time
            window_start: CacheCounterValue | None = await self.internal_usage_cache.async_get_cache(
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
                current_counter: CacheCounterValue | None = await self.internal_usage_cache.async_get_cache(
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
        counter_key: Final = f"{{{key}:{value}}}:{rate_limit_type}"

        return counter_key

    def is_cache_list_over_limit(
        self,
        keys_to_fetch: list[str],
        cache_values: CacheCounterValues,
        key_metadata: dict[str, WindowKeyMetadata],
    ) -> RateLimitResponse:
        """
        Check if the cache values are over the limit.
        """
        statuses: Final[list[RateLimitStatus]] = []
        overall_code = "OK"

        for i in range(0, len(cache_values), 2):
            item_code = "OK"
            window_key = keys_to_fetch[i]
            counter_key = keys_to_fetch[i + 1]
            counter_value = cache_values[i + 1]
            requests_limit = key_metadata[window_key]["requests_limit"]
            tokens_limit = key_metadata[window_key]["tokens_limit"]

            # Determine which limit to use for current_limit and limit_remaining
            current_limit: int | None = None
            rate_limit_type: Literal["requests", "tokens", "max_parallel_requests"] | None = None
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
        start: Final = key.find("{")
        if start != -1:
            end: Final = key.find("}", start + 1)
            if end != -1 and end != start + 1:
                key = key[start + 1 : end]

        # Compute CRC16 and mod 16384
        crc: Final = binascii.crc_hqx(key.encode("utf-8"), 0)
        return crc % REDIS_CLUSTER_SLOTS

    def _group_keys_by_hash_tag(self, keys: list[str]) -> dict[str, list[str]]:
        """
        Group keys by their Redis hash tag to ensure cluster compatibility.

        For Redis clusters, uses slot calculation to group keys that belong to the same slot.
        For regular Redis, no grouping is needed - all keys can be processed together.
        """
        groups: Final[dict[str, list[str]]] = {}

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

    async def _batch_get_counter_values(
        self,
        keys: list[str],
        parent_otel_span: Span | None,
        local_only: bool,
    ) -> CacheCounterValues | None:
        """Typed view over the DualCache batch read of window/counter keys."""
        return await self.internal_usage_cache.async_batch_get_cache(
            keys=keys,
            parent_otel_span=parent_otel_span,
            local_only=local_only,
        )

    async def _batch_get_gauge_values(
        self,
        keys: list[str],
        parent_otel_span: Span | None,
    ) -> Sequence[ParallelGaugeCacheValue | None] | None:
        """Typed view over the DualCache batch read of parallel-request gauges."""
        return await self.internal_usage_cache.async_batch_get_cache(
            keys=keys,
            parent_otel_span=parent_otel_span,
            local_only=True,
        )

    async def _execute_redis_batch_rate_limiter_script(
        self,
        keys_to_fetch: list[str],
        now_int: int,
    ) -> CacheCounterValues:
        """
        Execute Redis operations grouped by hash tag for cluster compatibility.

        Args:
            keys_to_fetch: List[str] - List of keys to fetch
            now_int: int - Current timestamp

        Returns:
            List of cache values
        """
        if self.batch_rate_limiter_script is None:
            return []

        key_groups: Final = self._group_keys_by_hash_tag(keys_to_fetch)
        all_cache_values: Final[list[CacheCounterValue | None]] = []

        for hash_tag, group_keys in key_groups.items():
            try:
                group_cache_values: CacheCounterValues = await self.batch_rate_limiter_script(
                    keys=group_keys,
                    args=[now_int, self.window_size],  # Use integer timestamp
                )
                all_cache_values.extend(group_cache_values)
            except Exception as e:
                verbose_proxy_logger.warning("Redis Lua script failed for hash tag %s: %s", hash_tag, e)
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
        descriptors: Sequence[RateLimitDescriptor],
        parent_otel_span: Span | None = None,
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

        current_time: Final = self._get_current_time()
        now: Final = current_time.timestamp()
        now_int: Final = int(now)  # Convert to integer for Redis Lua script

        keys_to_fetch, key_metadata, gauges = self._collect_windowed_keys_and_gauges(
            descriptors=descriptors,
            skip_tpm_check=skip_tpm_check,
        )

        windowed_response = RateLimitResponse(overall_code="OK", statuses=[])
        if keys_to_fetch:
            ## CHECK IN-MEMORY CACHE
            cache_values = await self._batch_get_counter_values(  # rebind-ok: refreshed by the Redis read below when the in-memory pass is under limit
                keys=keys_to_fetch,
                parent_otel_span=parent_otel_span,
                local_only=True,
            )

            if cache_values is not None:
                rate_limit_response: Final = self.is_cache_list_over_limit(keys_to_fetch, cache_values, key_metadata)
                if rate_limit_response["overall_code"] == "OVER_LIMIT":
                    return rate_limit_response

            ## IF under limit in-memory, check Redis
            if read_only:
                # READ-ONLY MODE: Just read current values without incrementing
                cache_values = await self._batch_get_counter_values(  # rebind-ok: read-only mode replaces the in-memory snapshot with Redis values
                    keys=keys_to_fetch,
                    parent_otel_span=parent_otel_span,
                    local_only=False,  # Check Redis too
                )

                # For keys that don't exist yet, set them to 0
                if cache_values is None:
                    cache_values = [  # rebind-ok: missing keys default to a zeroed window snapshot
                        str(now_int) if key.endswith(":window") else 0 for key in keys_to_fetch
                    ]
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

        gauge_response: Final = await self._check_parallel_request_gauges(
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
        descriptors: Sequence[RateLimitDescriptor],
        skip_tpm_check: bool,
    ) -> tuple[list[str], dict[str, WindowKeyMetadata], list[ParallelRequestGauge]]:
        """
        Split descriptors into the windowed (window_key, counter_key) fetch
        list with its per-window metadata, and the concurrency gauges for
        descriptors carrying a max_parallel_requests limit.
        """
        keys_to_fetch: Final[list[str]] = []
        key_metadata: Final[dict[str, WindowKeyMetadata]] = {}
        gauges: Final[list[ParallelRequestGauge]] = []
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

    def _gauge_in_flight_from_cache_value(self, raw_value: ParallelGaugeCacheValue | None) -> int:
        """
        In-flight count from a cached gauge value: a dict of slot_id ->
        acquire timestamp when the in-memory registry is authoritative, or
        the mirrored integer count from the last Redis script result.
        """
        if raw_value is None:
            return 0
        if isinstance(raw_value, dict):
            cutoff: Final = self._get_current_time().timestamp() - PARALLEL_REQUEST_SLOT_TTL_SECONDS
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
        gauge_keys: Final = [gauge["counter_key"] for gauge in gauges]

        if read_only:
            if self.parallel_count_script is not None:
                try:
                    raw_counts: Final[list[CacheCounterValue]] = await self.parallel_count_script(
                        keys=gauge_keys,
                        args=[PARALLEL_REQUEST_SLOT_TTL_SECONDS for _ in gauges],
                    )
                    counts = [max(0, int(value)) for value in raw_counts]
                except Exception as e:  # noqa: BLE001 - any Redis/Lua failure degrades to the local mirror, never a 500
                    verbose_proxy_logger.warning("parallel_count_script failed, using local mirror: %s", e)
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

        local_counts: Final = await self._read_local_gauge_counts(gauge_keys, parent_otel_span)
        for gauge, in_flight in zip(gauges, local_counts):
            if in_flight >= gauge["limit"]:
                return RateLimitResponse(
                    overall_code="OVER_LIMIT",
                    statuses=[self._gauge_status(gauge, in_flight, "OVER_LIMIT")],
                )

        if self.parallel_acquire_script is not None:
            try:
                raw: Final[list[CacheCounterValue]] = await self.parallel_acquire_script(
                    keys=gauge_keys,
                    args=[
                        arg for gauge in gauges for arg in (gauge["limit"], PARALLEL_REQUEST_SLOT_TTL_SECONDS, slot_id)
                    ],
                )
            except Exception as e:  # noqa: BLE001 - any Redis/Lua failure degrades to in-memory enforcement, never a 500
                verbose_proxy_logger.warning("parallel_acquire_script failed, falling back to in-memory gauge: %s", e)
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
        values: Final = await self._batch_get_gauge_values(
            keys=gauge_keys,
            parent_otel_span=parent_otel_span,
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
        now: Final = self._get_current_time().timestamp()
        cutoff: Final = now - PARALLEL_REQUEST_SLOT_TTL_SECONDS
        states: Final[list[tuple[dict[str, float] | None, int]]] = []
        for gauge in gauges:
            raw_value: ParallelGaugeCacheValue | None = await self.internal_usage_cache.async_get_cache(
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

        statuses: Final = []
        for gauge, (registry, in_flight) in zip(gauges, states):
            new_value: dict[str, float] | int = {**registry, slot_id: now} if registry is not None else in_flight + 1
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
        counter_keys: Final = acquisition["counter_keys"]
        slot_id: Final = acquisition["slot_id"]
        if not counter_keys or not slot_id:
            return
        if self.parallel_release_script is not None:
            try:
                raw: Final[list[CacheCounterValue]] = await self.parallel_release_script(
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
                verbose_proxy_logger.warning("parallel_release_script failed, falling back to in-memory release: %s", e)

        async with self._check_and_increment_lock:
            for counter_key in counter_keys:
                raw_value: ParallelGaugeCacheValue | None = await self.internal_usage_cache.async_get_cache(
                    key=counter_key,
                    litellm_parent_otel_span=parent_otel_span,
                    local_only=True,
                )
                if isinstance(raw_value, dict):
                    if slot_id not in raw_value:
                        continue
                    new_value: dict[str, object] | int = {key: ts for key, ts in raw_value.items() if key != slot_id}
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
        descriptors: list[RateLimitDescriptor],
        increments: list[dict[Literal["requests", "tokens"], int]],
        parent_otel_span: Span | None = None,
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
        descriptor_groups: Final[list[DescriptorAtomicGroup]] = []
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

        flat_meta: Final[list[AtomicCounterMeta]] = [
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
        increment_amounts: dict[Literal["requests", "tokens"], int],
    ) -> DescriptorAtomicGroup:
        """
        Build (KEYS, ARGV, per-counter meta) for a single descriptor's Lua
        call. All keys returned share the descriptor's {key:value} hash tag.
        """
        descriptor_key: Final = descriptor["key"]
        descriptor_value: Final = descriptor["value"]
        rate_limit: Final[RateLimitDescriptorRateLimitObject] = (
            descriptor.get("rate_limit") or RateLimitDescriptorRateLimitObject()
        )
        window_size: Final = rate_limit.get("window_size") or self.window_size
        window_key: Final = f"{{{descriptor_key}:{descriptor_value}}}:window"

        keys: Final[list[str]] = []
        args: Final[list[int]] = []
        meta: Final[list[AtomicCounterMeta]] = []

        rate_limit_types: Final[tuple[Literal["requests", "tokens"], ...]] = ("requests", "tokens")
        for rlt in rate_limit_types:
            if rlt == "requests":
                limit_value = rate_limit.get("requests_per_unit")
                inc_amount = int(increment_amounts.get("requests", 0) or 0)
            else:
                limit_value = rate_limit.get("tokens_per_unit")
                inc_amount = int(increment_amounts.get("tokens", 0) or 0)
            if limit_value is None or inc_amount < 0:
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
                    "descriptor_value": descriptor_value,
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
        descriptor_groups: list[DescriptorAtomicGroup],
        parent_otel_span: Span | None = None,
    ) -> RateLimitResponse:
        """
        Run Lua check-and-increment one descriptor at a time so each call's
        keys co-locate on a single Redis Cluster slot. On OVER_LIMIT for
        descriptor i, refund descriptors 0..i-1's increments. On Lua failure
        mid-loop, refund applied increments and fall back to in-memory.
        """
        if not descriptor_groups:
            return RateLimitResponse(
                overall_code="OK",
                statuses=[],  # mutable-ok: response contract requires a status list
            )
        applied: Final[list[list[AtomicCounterMeta]]] = []
        statuses: Final[list[RateLimitStatus]] = []
        raw: list[CacheCounterValue]

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
                    "atomic_check_and_increment_by_n: Redis Lua execution failed (%s: %s). Refunding %s prior descriptors and falling back to in-memory enforcement — counters will diverge from Redis until window expires (window_size=%ss).",
                    type(e).__name__,
                    e,
                    len(applied),
                    self.window_size,
                )
                await self._refund_applied_descriptor_groups(applied)
                flat_meta: list[AtomicCounterMeta] = [m for _k, _a, group_meta in descriptor_groups for m in group_meta]
                async with self._check_and_increment_lock:
                    return await self._atomic_check_and_increment_in_memory(
                        per_counter_meta=flat_meta,
                        parent_otel_span=parent_otel_span,
                    )

            response = self._build_atomic_response(raw, meta)
            if response["overall_code"] == "OVER_LIMIT":
                await self._refund_applied_descriptor_groups(applied)
                return response
            if len(descriptor_groups) == 1:
                return response
            applied.append(meta)
            statuses.extend(response["statuses"])

        return RateLimitResponse(
            overall_code="OK",
            statuses=statuses,
            reservation_windows=frozenset(),
        )

    async def _refund_applied_descriptor_groups(
        self,
        applied: list[list[AtomicCounterMeta]],
    ) -> None:
        """
        Decrement counters for descriptor groups already applied via Lua.
        Best-effort: refund failures are logged but not raised — the original
        OVER_LIMIT / fallback decision is what matters to the caller.
        """
        if not applied:
            return
        redis_cache: Final = self.internal_usage_cache.dual_cache.redis_cache
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
                        "Failed to refund %s on cross-descriptor rollback: %s", entry["counter_key"], e
                    )

    def _build_atomic_response(
        self,
        raw: list[CacheCounterValue],
        per_counter_meta: list[AtomicCounterMeta],
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

        status_code: Final = int(raw[0])
        if status_code == 1:
            # Over limit: { 1, counter_index (1-based), current_counter, limit }
            descriptor_index: Final = int(raw[1]) - 1
            current_counter: Final = int(raw[2])
            limit: Final = int(raw[3])
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
                        descriptor_value=meta["descriptor_value"],
                    )
                ],
            )

        statuses: Final[list[RateLimitStatus]] = []
        for index, meta in enumerate(per_counter_meta):
            new_counter = raw[1 + index * 2]
            statuses.append(
                RateLimitStatus(
                    code="OK",
                    current_limit=meta["current_limit"],
                    limit_remaining=max(0, meta["current_limit"] - int(new_counter)),
                    rate_limit_type=meta["rate_limit_type"],
                    descriptor_key=meta["descriptor_key"],
                    descriptor_value=meta["descriptor_value"],
                )
            )
        return RateLimitResponse(
            overall_code="OK",
            statuses=statuses,
            reservation_windows=frozenset(
                (
                    meta["counter_key"],
                    str(int(raw[2 + index * 2])),
                    "redis",
                )
                for index, meta in enumerate(per_counter_meta)
            ),
        )

    async def _atomic_check_and_increment_in_memory(
        self,
        per_counter_meta: list[AtomicCounterMeta],
        parent_otel_span: Span | None = None,
    ) -> RateLimitResponse:
        """In-memory all-or-nothing check-and-increment. Caller holds lock.

        Reads/writes the LOCAL DualCache (`local_only=True`) — note this is
        a different store from Redis. When this fallback fires after a Lua
        failure, in-memory counters will diverge from Redis until each key's
        window expires (TTL bounds divergence).
        """
        # Use a single 'now' for the duration of this critical section so all
        # descriptors evaluate window expiry consistently.
        now_int: Final = int(self._get_current_time().timestamp())

        # Pass 1: read state, validate.
        descriptor_state: Final[list[AtomicCounterState]] = []
        for meta in per_counter_meta:
            window_size = meta["window_size"]
            window_start: CacheCounterValue | None = await self.internal_usage_cache.async_get_cache(
                key=meta["window_key"],
                litellm_parent_otel_span=parent_otel_span,
                local_only=True,
            )
            window_expired = window_start is None or (now_int - int(window_start)) >= window_size
            raw_counter: CacheCounterValue | None = (
                None
                if window_expired
                else await self.internal_usage_cache.async_get_cache(
                    key=meta["counter_key"],
                    litellm_parent_otel_span=parent_otel_span,
                    local_only=True,
                )
            )
            current_counter = 0 if window_expired else int(raw_counter or 0)
            over_limit = (
                current_counter + meta["increment"] > meta["current_limit"]
                if meta["increment"] > 0
                else current_counter >= meta["current_limit"]
            )
            if over_limit:
                return RateLimitResponse(
                    overall_code="OVER_LIMIT",
                    statuses=[
                        RateLimitStatus(
                            code="OVER_LIMIT",
                            current_limit=meta["current_limit"],
                            limit_remaining=max(0, meta["current_limit"] - current_counter),
                            rate_limit_type=meta["rate_limit_type"],
                            descriptor_key=meta["descriptor_key"],
                            descriptor_value=meta["descriptor_value"],
                        )
                    ],
                )
            descriptor_state.append(
                {  # mutable-ok: local atomic-counter state is updated during pass two
                    "window_expired": window_expired,
                    "current": current_counter,
                    "window_start": str(now_int if window_expired else int(window_start)),
                }
            )

        # Pass 2: apply increments.
        statuses: Final[list[RateLimitStatus]] = []
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
                    descriptor_value=meta["descriptor_value"],
                )
            )
        return RateLimitResponse(
            overall_code="OK",
            statuses=statuses,
            reservation_windows=frozenset(
                (meta["counter_key"], state["window_start"], "local")
                for meta, state in zip(per_counter_meta, descriptor_state)
            ),
        )

    async def reserve_tpm_tokens(
        self,
        descriptors: list[RateLimitDescriptor],
        estimated_tokens: int,
        parent_otel_span: Span | None = None,
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
        tpm_descriptors: Final[list[RateLimitDescriptor]] = [
            RateLimitDescriptor(
                key=d["key"],
                value=d["value"],
                rate_limit=RateLimitDescriptorRateLimitObject(
                    tokens_per_unit=(d.get("rate_limit") or {}).get("tokens_per_unit"),
                    window_size=(d.get("rate_limit") or {}).get("window_size"),
                ),
            )
            for d in descriptors
            if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
            and (d.get("rate_limit") or {}).get("tokens_per_unit") is not None  # mutable-ok: optional descriptor
        ]
        if not tpm_descriptors:
            return RateLimitResponse(overall_code="OK", statuses=[])

        increments: Final[list[dict[Literal["requests", "tokens"], int]]] = [
            {"tokens": estimated_tokens} for _ in tpm_descriptors
        ]
        return await self.atomic_check_and_increment_by_n(
            descriptors=tpm_descriptors,
            increments=increments,
            parent_otel_span=parent_otel_span,
        )

    async def _refund_reserved_tokens(
        self,
        scopes: Sequence[tuple[str, str]],
        amount: int,
        reservation_windows: frozenset[tuple[str, str, Literal["redis", "local"]]] = frozenset(),
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
        if not reservation_windows:
            await self.async_increment_tokens_with_ttl_preservation(
                pipeline_operations=self._build_reservation_aware_tpm_ops(
                    targets=scopes,
                    reserved_scopes=frozenset(scopes),
                    actual_tokens=0,
                    reserved_tokens=amount,
                ),
                parent_otel_span=parent_otel_span,
            )
            return
        pipeline_operations: Final = self._build_project_reservation_ops(
            targets=scopes,
            reserved_scopes=frozenset(scopes),
            actual_tokens=0,
            reserved_tokens=amount,
            reservation_window_identities=reservation_windows,
        )
        await self.async_increment_reservation_aware_tokens(
            pipeline_operations=pipeline_operations,
            parent_otel_span=parent_otel_span,
        )

    async def reserve_io_tokens(
        self,
        descriptors: Sequence[RateLimitDescriptor],
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
        itpm_descriptors: Final = [  # mutable-ok: atomic limiter API requires lists
            d for d in descriptors if d["key"] == PROJECT_ITPM_DESCRIPTOR_KEY
        ]
        otpm_descriptors: Final = [  # mutable-ok: atomic limiter API requires lists
            d for d in descriptors if d["key"] == PROJECT_OTPM_DESCRIPTOR_KEY
        ]

        if not itpm_descriptors and not otpm_descriptors:
            return RateLimitResponse(overall_code="OK", statuses=[]), 0, 0  # mutable-ok: response contract uses a list

        itpm_response: Final = (
            await self.atomic_check_and_increment_by_n(
                descriptors=itpm_descriptors,
                increments=[  # mutable-ok: atomic limiter API requires mutable increment records
                    {"tokens": estimated_input_tokens}  # mutable-ok: atomic limiter increment record
                    for _ in itpm_descriptors
                ],
                parent_otel_span=parent_otel_span,
            )
            if itpm_descriptors
            else None
        )
        if itpm_response is not None and itpm_response["overall_code"] == "OVER_LIMIT":
            return itpm_response, 0, 0
        itpm_reserved: Final = estimated_input_tokens if itpm_response is not None else 0

        if otpm_descriptors:
            otpm_response: Final = await self.atomic_check_and_increment_by_n(
                descriptors=otpm_descriptors,
                increments=[  # mutable-ok: atomic limiter API requires mutable increment records
                    {"tokens": estimated_output_tokens}  # mutable-ok: atomic limiter increment record
                    for _ in otpm_descriptors
                ],
                parent_otel_span=parent_otel_span,
            )
            if otpm_response["overall_code"] == "OVER_LIMIT":
                if itpm_reserved > 0:
                    await self._refund_reserved_tokens(
                        scopes=[  # mutable-ok: reservation rollback accepts collected scopes
                            (d["key"], d["value"]) for d in itpm_descriptors
                        ],
                        amount=itpm_reserved,
                        reservation_windows=itpm_response.get("reservation_windows", frozenset()),
                        parent_otel_span=parent_otel_span,
                    )
                return otpm_response, 0, 0
            statuses: Final = (
                [  # mutable-ok: response contract uses a list
                    *itpm_response["statuses"],
                    *otpm_response["statuses"],
                ]
                if itpm_response is not None
                else otpm_response["statuses"]
            )
            return (
                RateLimitResponse(
                    overall_code="OK",
                    statuses=statuses,
                    reservation_windows=(
                        (
                            itpm_response.get("reservation_windows", frozenset())
                            if itpm_response is not None
                            else frozenset()
                        )
                        | otpm_response.get("reservation_windows", frozenset())
                    ),
                ),
                itpm_reserved,
                estimated_output_tokens,
            )

        assert itpm_response is not None
        return itpm_response, itpm_reserved, 0

    async def enforce_project_io_token_quota_for_frame(
        self,
        user_api_key_dict: UserAPIKeyAuth | None,
        requested_model: str | None,
        estimated_input_tokens: int,
        estimated_output_tokens: int,
    ) -> None:
        """Reserve one WebSocket ``response.create`` frame's tokens against
        the caller's project ITPM/OTPM quota.

        The Responses WebSocket connection-level pre-call hook only runs once
        per connection, but a connection accepts many ``response.create``
        frames over its lifetime. Without this, a project caller could send
        unlimited high-token generations after a single minimal reservation.
        There is no per-frame post-call hook to reconcile against, so --
        like the batch rate limiter -- this charges the estimate immediately
        and never refunds it.
        """
        if user_api_key_dict is None:
            return
        descriptors: Final[list[RateLimitDescriptor]] = []  # mutable-ok: descriptor helper appends in place
        self.add_project_io_token_rate_limit_descriptors_from_metadata(
            user_api_key_dict=user_api_key_dict,
            requested_model=requested_model,
            descriptors=descriptors,
        )
        if not descriptors:
            return
        response, _itpm_reserved, _otpm_reserved = await self.reserve_io_tokens(
            descriptors=descriptors,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )
        if response["overall_code"] == "OVER_LIMIT":
            self._handle_rate_limit_error(response, descriptors, requested_model)

    def create_organization_rate_limit_descriptor(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None = None
    ) -> list[RateLimitDescriptor]:
        descriptors: Final[list[RateLimitDescriptor]] = []

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
            _tpm_limit_for_team_model: Final = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "organization_metadata", "model_tpm_limit") or {}
            )
            _rpm_limit_for_team_model: Final = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "organization_metadata", "model_rpm_limit") or {}
            )

            should_check_rate_limit = False
            if requested_model in _tpm_limit_for_team_model or requested_model in _rpm_limit_for_team_model:
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
        requested_model: str | None,
        descriptors: list[RateLimitDescriptor],
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
        should_check_rate_limit: Final = (
            requested_model in _tpm_limit_for_key_model or requested_model in _rpm_limit_for_key_model
        )

        if not should_check_rate_limit:
            return

        # Get model-specific limits
        model_specific_tpm_limit: Final[int | None] = _tpm_limit_for_key_model.get(requested_model)
        model_specific_rpm_limit: Final[int | None] = _rpm_limit_for_key_model.get(requested_model)

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

        tag_rpm_limit: Final = get_key_tag_rpm_limit(user_api_key_dict) or {}
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
        mcp_server_name: str | None,
        descriptors: list[RateLimitDescriptor],
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

        mcp_rpm_limit: Final = get_key_mcp_rpm_limit(user_api_key_dict)
        if not mcp_rpm_limit:
            return

        server_rpm_limit: Final = mcp_rpm_limit.get(mcp_server_name)
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
        mcp_server_name: str | None,
        descriptors: list[RateLimitDescriptor],
    ) -> None:
        """
        Add a per-MCP-server rpm descriptor for the team, if a limit is
        configured for the server being called.
        """
        from litellm.proxy.auth.auth_utils import get_team_mcp_rpm_limit

        if not mcp_server_name:
            return

        # Which teams' buckets does this call charge? A key is pinned to exactly one team. A keyless
        # MCP-admitted subject reaches servers through SEVERAL teams at once and has no team_id, so
        # without the second source below its calls charged no team bucket at all and it outran every
        # team's mcp_rpm_limit. Every applicable team is charged rather than one being picked: the
        # limiter enforces all descriptors, so each team's own ceiling binds on a call made through
        # its grant, and there is no arbitrary attribution when several teams grant the same server.
        team_limits: Final[list[tuple[str | None, dict[str, int] | None]]] = []
        if user_api_key_dict.team_id:
            team_limits.append((user_api_key_dict.team_id, get_team_mcp_rpm_limit(user_api_key_dict)))
        for source_team_id, source_limit in (user_api_key_dict.mcp_source_team_rpm_limits or {}).items():
            team_limits.append((source_team_id, source_limit))

        for team_id, mcp_rpm_limit in team_limits:
            if not team_id or not mcp_rpm_limit:
                continue
            server_rpm_limit = mcp_rpm_limit.get(mcp_server_name)
            if server_rpm_limit is None:
                continue
            descriptors.append(
                RateLimitDescriptor(
                    key="mcp_per_team",
                    value=f"{team_id}:{mcp_server_name}",
                    rate_limit={
                        "requests_per_unit": server_rpm_limit,
                        "tokens_per_unit": None,
                        "window_size": self.window_size,
                    },
                )
            )

    def _should_enforce_rate_limit(
        self,
        limit_type: str | None,
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
        limit_value: int | None,
        limit_type: str | None,
        model_has_failures: bool,
    ) -> int | None:
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
        rpm_limit_type: str | None,
        tpm_limit_type: str | None,
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

    def _get_agent_from_registry(self, agent_id: str) -> "AgentResponse | None":
        """Look up an agent from the in-memory registry by ID."""
        from litellm.proxy.agent_endpoints.agent_registry import global_agent_registry

        return global_agent_registry.get_agent_by_id(agent_id=agent_id)

    def _get_resolved_agent_id(self, user_api_key_dict: UserAPIKeyAuth, data: dict) -> str | None:
        """
        Resolve the agent_id from either the API key or request metadata.
        Key-level agent_id takes precedence over metadata/header-supplied agent_id.
        """
        key_agent_id: Final = getattr(user_api_key_dict, "agent_id", None)
        if key_agent_id:
            return key_agent_id
        metadata: Final = data.get("metadata") or {}
        return metadata.get("agent_id")

    def _get_session_id_from_data(self, data: dict) -> str | None:
        """Extract session_id from request metadata or litellm_session_id."""
        session_id = data.get("litellm_session_id")
        if session_id:
            return str(session_id)
        metadata: Final = data.get("metadata") or {}
        session_id = metadata.get("session_id")
        if session_id:
            return str(session_id)
        litellm_metadata: Final = data.get("litellm_metadata") or {}
        session_id = litellm_metadata.get("session_id")
        if session_id:
            return str(session_id)
        return None

    def _create_agent_rate_limit_descriptors(
        self,
        agent_id: str,
        data: dict,
    ) -> list[RateLimitDescriptor]:
        """
        Create rate limit descriptors for agent-level and session-level limits.

        Agent-level: caps total RPM/TPM across all sessions for a given agent.
        Session-level: caps RPM/TPM within a single session (identified by session_id).
        """
        descriptors: Final[list[RateLimitDescriptor]] = []

        agent: Final = self._get_agent_from_registry(agent_id)
        if agent is None:
            return descriptors

        agent_rpm: Final = getattr(agent, "rpm_limit", None)
        agent_tpm: Final = getattr(agent, "tpm_limit", None)
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

        session_rpm: Final = getattr(agent, "session_rpm_limit", None)
        session_tpm: Final = getattr(agent, "session_tpm_limit", None)
        if session_rpm is not None or session_tpm is not None:
            session_id: Final = self._get_session_id_from_data(data)
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
        rpm_limit_type: str | None,
        tpm_limit_type: str | None,
        model_has_failures: bool,
        call_type: str | None = None,
    ) -> list[RateLimitDescriptor]:
        """
        Create all rate limit descriptors for the request.

        Returns list of descriptors for API key, user, team, team member, end user,
        model-specific, agent, and agent-session limits.
        """
        from litellm.proxy.auth.auth_utils import (
            get_team_model_rpm_limit,
            get_team_model_tpm_limit,
        )

        descriptors: Final = []

        # API Key rate limits
        if user_api_key_dict.api_key and (
            user_api_key_dict.rpm_limit is not None
            or user_api_key_dict.tpm_limit is not None
            or user_api_key_dict.max_parallel_requests is not None
        ):
            throttle_pct: Final = user_api_key_dict.budget_throttle_pct
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
            team_member_value: Final = f"{user_api_key_dict.team_id}:{user_api_key_dict.user_id}"
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
        requested_model: Final = data.get("model", None)
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
            mcp_server_name: Final = data.get("mcp_server_name", None)
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
            _tpm_limit_for_team_model: Final = get_team_model_tpm_limit(user_api_key_dict) or {}
            _rpm_limit_for_team_model: Final = get_team_model_rpm_limit(user_api_key_dict) or {}
            should_check_rate_limit = False
            if requested_model in _tpm_limit_for_team_model or requested_model in _rpm_limit_for_team_model:
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
        resolved_agent_id: Final = self._get_resolved_agent_id(user_api_key_dict, data)

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
        parent_otel_span: Span | None = None,
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
            model_list: Final = llm_router.get_model_list(model_name=model)
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
                        "[Dynamic Rate Limit] Deployment %s has %s failures in current minute - enforcing rate limits for model %s",
                        deployment_id,
                        failure_count,
                        model,
                    )
                    return True

            verbose_proxy_logger.debug(
                "[Dynamic Rate Limit] No failures detected for model %s - allowing dynamic exceeding", model
            )
            return False

        except Exception as e:
            verbose_proxy_logger.debug("Error checking model failure status: %s, defaulting to enforce limits", e)
            # Fail safe: enforce limits if we can't check
            return True

    def get_rate_limiter_for_call_type(self, call_type: str) -> CallTypeRateLimiter | None:
        """Get the rate limiter for the call type."""
        if call_type == "acreate_batch":
            batch_limiter: Final = self._get_batch_rate_limiter()
            return batch_limiter
        return None

    def _add_team_model_rate_limit_descriptor_from_metadata(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str | None,
        descriptors: list[RateLimitDescriptor],
    ) -> None:
        """Add team model rate limit descriptor from team_metadata if applicable."""
        if (
            get_model_rate_limit_from_metadata(user_api_key_dict, "team_metadata", "model_rpm_limit") is not None
            or get_model_rate_limit_from_metadata(user_api_key_dict, "team_metadata", "model_tpm_limit") is not None
        ):
            _tpm_limit_for_team_model: Final = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "team_metadata", "model_tpm_limit") or {}
            )
            _rpm_limit_for_team_model: Final = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "team_metadata", "model_rpm_limit") or {}
            )
            should_check_rate_limit: Final = (
                requested_model in _tpm_limit_for_team_model or requested_model in _rpm_limit_for_team_model
            )

            if should_check_rate_limit and requested_model is not None:
                model_specific_tpm_limit: Final = _tpm_limit_for_team_model.get(requested_model)
                model_specific_rpm_limit: Final = _rpm_limit_for_team_model.get(requested_model)
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
        requested_model: str | None,
        descriptors: list[RateLimitDescriptor],
    ) -> None:
        """Add project model rate limit descriptor from project_metadata if applicable."""
        if (
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_rpm_limit") is not None
            or get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_tpm_limit") is not None
        ):
            _tpm_limit_for_project_model: Final = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_tpm_limit") or {}
            )
            _rpm_limit_for_project_model: Final = (
                get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_rpm_limit") or {}
            )
            should_check_rate_limit: Final = (
                requested_model in _tpm_limit_for_project_model or requested_model in _rpm_limit_for_project_model
            )

            if should_check_rate_limit and requested_model is not None:
                model_specific_tpm_limit: Final = _tpm_limit_for_project_model.get(requested_model)
                model_specific_rpm_limit: Final = _rpm_limit_for_project_model.get(requested_model)
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

    def add_project_io_token_rate_limit_descriptors_from_metadata(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        requested_model: str | None,
        descriptors: _RateLimitDescriptorSink,
    ) -> None:
        """Add project-scoped ITPM/OTPM descriptors from project_metadata.

        Enforced independently of, and alongside, the combined ``model_per_project``
        TPM descriptor above -- these give Bedrock Mantle-style separate input/output
        token quotas at the project level.
        """
        if requested_model is None or user_api_key_dict.project_id is None:
            return

        itpm_limit_for_project_model: Final = (
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_itpm_limit")
            or {}  # mutable-ok: metadata helper returns an optional mapping
        )
        otpm_limit_for_project_model: Final = (
            get_model_rate_limit_from_metadata(user_api_key_dict, "project_metadata", "model_otpm_limit")
            or {}  # mutable-ok: metadata helper returns an optional mapping
        )

        model_itpm_limit: Final = itpm_limit_for_project_model.get(requested_model)
        model_otpm_limit: Final = otpm_limit_for_project_model.get(requested_model)

        if model_itpm_limit is None and model_otpm_limit is None:
            return

        descriptor_value: Final = f"{user_api_key_dict.project_id}:{requested_model}"
        if model_itpm_limit is not None:
            descriptors.append(
                RateLimitDescriptor(
                    key=PROJECT_ITPM_DESCRIPTOR_KEY,
                    value=descriptor_value,
                    rate_limit={  # mutable-ok: descriptor TypedDict requires a runtime dict
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
                    rate_limit={  # mutable-ok: descriptor TypedDict requires a runtime dict
                        "requests_per_unit": None,
                        "tokens_per_unit": model_otpm_limit,
                        "window_size": self.window_size,
                    },
                )
            )

    def _handle_rate_limit_error(
        self,
        response: RateLimitResponse,
        descriptors: list[RateLimitDescriptor],
        requested_model: str | None = None,
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

    @staticmethod
    def _estimate_audio_block_tokens(block: object) -> int:
        """
        Token estimate for one ``input_audio`` content block.

        When the block carries a base64 ``data`` payload, the estimate comes
        from the decoded byte count (``len(b64) * 3 // 4 // _AUDIO_BYTES_PER_TOKEN``),
        assuming the lowest reasonable audio bitrate so we never under-reserve
        for higher-quality recordings of the same duration.

        When no payload is present (reference-only block or missing ``data``),
        falls back to ``DEFAULT_AUDIO_TOKEN_ESTIMATE``.
        """
        if not isinstance(block, dict):
            return DEFAULT_AUDIO_TOKEN_ESTIMATE
        input_audio: Final = block.get("input_audio")
        b64_data: Final = input_audio.get("data") if isinstance(input_audio, dict) else None
        if b64_data and isinstance(b64_data, str):
            decoded_bytes: Final = len(b64_data) * 3 // 4
            return max(decoded_bytes // _AUDIO_BYTES_PER_TOKEN, DEFAULT_AUDIO_TOKEN_ESTIMATE)
        return DEFAULT_AUDIO_TOKEN_ESTIMATE

    @classmethod
    def _estimate_audio_content_tokens(cls, messages: object) -> int:
        """
        Sum of per-block audio token estimates across all ``messages``.
        Returns 0 when there are no ``input_audio`` blocks, which the caller
        uses to skip the (relatively expensive) strip pass.
        """
        if not isinstance(messages, list):
            return 0
        return sum(
            cls._estimate_audio_block_tokens(block)
            for message in messages
            if isinstance(message, dict)
            for content in (message.get("content"),)
            if isinstance(content, list)
            for block in content
            if isinstance(block, dict) and block.get("type") == "input_audio"
        )

    @staticmethod
    def _strip_audio_content_blocks(messages: object) -> object:
        """
        Drop ``input_audio`` content blocks before passing ``messages`` to
        ``token_counter``, which raises ``ValueError`` on them (no per-type
        handling, unlike images). The audio contribution is added back
        separately via ``DEFAULT_AUDIO_TOKEN_ESTIMATE`` so the rest of the
        message (text/images/tools) still gets counted accurately instead of
        the whole call falling back to the cheap char-count estimate.
        """
        if not isinstance(messages, list):
            return messages
        sanitized: Final[list[object]] = []  # mutable-ok: token_counter requires a list of message dicts
        for message in messages:
            if not isinstance(message, dict):
                sanitized.append(message)
                continue
            content = message.get("content")
            if not isinstance(content, list):
                sanitized.append(message)
                continue
            filtered_content = [  # mutable-ok: token_counter requires list content blocks
                block for block in content if not (isinstance(block, dict) and block.get("type") == "input_audio")
            ]
            sanitized.append(  # mutable-ok: token_counter requires mutable message dicts
                {**message, "content": filtered_content}  # mutable-ok: token_counter requires message dicts
            )
        return sanitized

    @staticmethod
    def _responses_input_to_chat_messages(data: object) -> Sequence[object]:
        """
        Convert a Responses API ``input`` (string or list of input items) into
        chat-completion-style messages via the standard LiteLLM transformation
        (the same one guardrails use, e.g. ``purview_dlp.py``), so multimodal
        ``input_image``/``input_text`` content blocks get counted by
        ``token_counter``'s ``messages`` path instead of silently contributing
        zero tokens via its ``text`` path, which only joins plain strings.
        """
        from litellm.responses.litellm_completion_transformation.transformation import (
            LiteLLMCompletionResponsesConfig,
        )

        if not isinstance(data, dict):
            return ()
        return LiteLLMCompletionResponsesConfig.transform_responses_api_input_to_messages(
            input=data.get("input") or "",
            responses_api_request=data,
        )

    @staticmethod
    def _count_pretokenized_embedding_input(value: object) -> int | None:
        if not isinstance(value, list):
            return None
        if all(isinstance(token, int) for token in value):
            return len(value)
        if all(
            isinstance(token_ids, list) and all(isinstance(token, int) for token in token_ids) for token_ids in value
        ):
            return sum(len(token_ids) for token_ids in value)
        return None

    @staticmethod
    def _rerank_input_to_text(data: Mapping[str, object]) -> str:
        documents: Final = data.get("documents")
        document_items: Final[Sequence[object]] = documents if isinstance(documents, list) else ()  # pyright: ignore[reportUnknownVariableType]  # rerank documents are validated runtime JSON
        input_parts: Final[tuple[object, ...]] = (  # pyright: ignore[reportUnknownVariableType]  # list narrowing preserves unknown JSON element types
            data.get("query"),
            *document_items,
        )
        return "\n".join(
            str(part)  # pyright: ignore[reportUnknownArgumentType]  # accepted document dicts have provider-defined fields
            for part in input_parts  # pyright: ignore[reportUnknownVariableType]  # runtime JSON list elements remain unknown after list narrowing
            if isinstance(part, (str, dict))
        )

    def _estimate_precise_input_tokens(self, data: object, model: str | None, call_type: str | None = None) -> int:
        """
        Model-aware input token estimate for the project ITPM reservation,
        using ``litellm.token_counter`` -- the same approach the
        deployment-level itpm/otpm check uses in
        ``io_token_rate_limit_check.py``. Unlike the cheap char-count
        estimate the combined-TPM path uses, this accounts for image/tool
        content and derives per-``input_audio``-block estimates from the
        base64 payload size (assuming the lowest reasonable bitrate so
        longer recordings always reserve proportionally more), so a burst
        of multimodal, tool-heavy, or audio-heavy requests can't each
        reserve only the one-token floor and blow past ITPM before
        post-call reconciliation catches up.

        For the Responses API, ``input`` is converted to chat messages first
        (via ``_responses_input_to_chat_messages``) so its own multimodal
        content blocks are counted the same way; ``token_counter``'s ``text``
        argument can only see plain strings in a list, not content blocks.

        Falls back to the cheap char-count estimate if ``token_counter``
        can't resolve a tokenizer for this model (e.g. an unrecognized
        custom model name) or otherwise raises -- the audio add-on still
        applies on top of the fallback.
        """
        from litellm import token_counter

        if not isinstance(data, dict):
            return 0
        is_responses_request: Final = call_type in RESPONSES_API_CALL_TYPES
        translated_request: Final = (
            None if is_responses_request else self._translate_google_genai_native_request(data, call_type)
        )
        is_embedding_request: Final = self._is_embedding_request(data, call_type)
        embedding_text: Final = data.get("input") if is_embedding_request else None
        pretokenized_input_tokens: Final = (
            self._count_pretokenized_embedding_input(embedding_text) if is_embedding_request else None
        )
        if pretokenized_input_tokens is not None:
            return pretokenized_input_tokens

        prompt: Final = data.get("prompt")
        fallback_text: Final = prompt if prompt is not None else data.get("input")
        selected_inputs: Final[tuple[object | None, object | None, object | None, object | None]] = (
            (self._responses_input_to_chat_messages(data), None, data.get("tools"), data.get("tool_choice"))
            if is_responses_request
            else (
                translated_request.get("messages"),
                None,
                translated_request.get("tools"),
                translated_request.get("tool_choice"),
            )
            if translated_request is not None
            else (None, embedding_text, data.get("tools"), data.get("tool_choice"))
            if is_embedding_request
            else (None, self._rerank_input_to_text(data), data.get("tools"), data.get("tool_choice"))
            if call_type in RERANK_API_CALL_TYPES
            else (None, prompt, data.get("tools"), data.get("tool_choice"))
            if call_type in TEXT_COMPLETION_API_CALL_TYPES
            else (data.get("messages"), fallback_text, data.get("tools"), data.get("tool_choice"))
        )
        messages, selected_text, countable_tools, countable_tool_choice = selected_inputs

        audio_token_estimate: Final = self._estimate_audio_content_tokens(messages)
        countable_messages: Final = self._strip_audio_content_blocks(messages) if audio_token_estimate > 0 else messages

        try:
            estimate: Final = max(
                0,
                int(
                    token_counter(
                        model=model or "",
                        messages=countable_messages,
                        text=selected_text,
                        tools=countable_tools,
                        tool_choice=countable_tool_choice,
                        use_default_image_token_count=True,
                    )
                ),
            )
            return estimate + audio_token_estimate
        except Exception:  # noqa: BLE001  # tokenizer failures degrade to the cheap estimate
            if call_type in RERANK_API_CALL_TYPES and isinstance(selected_text, str):
                return max(0, len(selected_text) // DEFAULT_CHARS_PER_TOKEN)
            estimated_input_tokens, _ = self._estimate_input_and_output_tokens(data=data, call_type=call_type)
            return estimated_input_tokens + audio_token_estimate

    async def _reserve_project_io_tokens_or_raise(
        self,
        descriptors: Sequence[RateLimitDescriptor],
        data: object,
        requested_model: str | None,
        user_api_key_dict: UserAPIKeyAuth,
        tpm_reservation_scopes: Sequence[tuple[str, str]],
        tpm_reservation_amount: int,
        call_type: str | None = None,
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
        if not isinstance(data, dict):
            return
        stash: Final = claim_request_stash_for_data(data)
        io_token_descriptors: Final = [  # mutable-ok: reservation API requires descriptor lists
            d for d in descriptors if d["key"] in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
        ]
        if not io_token_descriptors:
            return

        configured_otpm_limits: Final = [  # mutable-ok: min calculation materializes validated limits
            int(v)
            for d in io_token_descriptors
            if d["key"] == PROJECT_OTPM_DESCRIPTOR_KEY
            for v in [  # mutable-ok: comprehension binds the optional descriptor value
                (d.get("rate_limit") or {}).get(  # mutable-ok: optional descriptor fallback
                    "tokens_per_unit"
                )
            ]
            if v is not None
        ]
        min_configured_otpm_limit: Final = min(configured_otpm_limits) if configured_otpm_limits else None
        _, raw_estimated_output_tokens = self._estimate_input_and_output_tokens(
            data=data,
            min_configured_tpm_limit=min_configured_otpm_limit,
            call_type=call_type,
        )
        raw_estimated_input_tokens: Final = self._estimate_precise_input_tokens(
            data=data, model=requested_model, call_type=call_type
        )
        estimated_input_tokens: Final = max(raw_estimated_input_tokens, 1)
        estimated_output_tokens: Final = (
            raw_estimated_output_tokens
            if self._has_explicit_output_cap(data, call_type)
            else max(raw_estimated_output_tokens, 1)
        )

        # Hard-cap generation length so an unbounded response can't overshoot
        # the OTPM budget before post-call reconciliation runs, mirroring the
        # combined-TPM floor cap in the caller.
        self._apply_implicit_output_cap(
            data=data,
            min_configured_limit=min_configured_otpm_limit,
            call_type=call_type,
        )

        io_response, itpm_reserved, otpm_reserved = await self.reserve_io_tokens(
            descriptors=io_token_descriptors,
            estimated_input_tokens=estimated_input_tokens,
            estimated_output_tokens=estimated_output_tokens,
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )

        if io_response["overall_code"] == "OVER_LIMIT":
            # A combined-TPM reservation may have already succeeded above for
            # this same request; refund it too, or its counter stays inflated
            # until the window's TTL expires. Mark it released so the
            # ProxyRateLimitError we're about to raise doesn't get refunded
            # a second time when async_post_call_failure_hook sees the same
            # (still-stashed) reservation and refunds it again.
            if tpm_reservation_amount > 0:
                await self._refund_reserved_tokens(
                    scopes=tpm_reservation_scopes,
                    amount=tpm_reservation_amount,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )
                stash.reservation_released = True
            acquisition: Final = stash.parallel_slot
            if acquisition is not None:
                await self._release_parallel_request_slots(
                    acquisition=acquisition,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )
                stash.parallel_slot = None
            self._handle_rate_limit_error(
                response=io_response,
                descriptors=descriptors,
                requested_model=requested_model,
            )

        if itpm_reserved > 0:
            itpm_scopes: Final = tuple(
                (d["key"], d["value"]) for d in io_token_descriptors if d["key"] == PROJECT_ITPM_DESCRIPTOR_KEY
            )
            stash.itpm_reserved_tokens = itpm_reserved
            stash.itpm_reserved_scopes = frozenset(itpm_scopes)
            stash.itpm_reserved_window_identities = frozenset(
                (counter_key, window_start, backend)
                for counter_key, window_start, backend in io_response.get("reservation_windows", frozenset())
                if "model_per_project_itpm" in counter_key
            )
        if otpm_reserved > 0:
            otpm_scopes: Final = tuple(
                (d["key"], d["value"]) for d in io_token_descriptors if d["key"] == PROJECT_OTPM_DESCRIPTOR_KEY
            )
            stash.otpm_reserved_tokens = otpm_reserved
            stash.otpm_reserved_scopes = frozenset(otpm_scopes)
            stash.otpm_reserved_window_identities = frozenset(
                (counter_key, window_start, backend)
                for counter_key, window_start, backend in io_response.get("reservation_windows", frozenset())
                if "model_per_project_otpm" in counter_key
            )

        if stash.rate_limit_response is not None:
            stash.rate_limit_response["statuses"].extend(io_response["statuses"])
        elif io_response["statuses"]:
            stash.rate_limit_response = io_response

        verbose_proxy_logger.debug(
            "ITPM/OTPM tokens reserved: itpm=%s, otpm=%s for model %s",
            itpm_reserved,
            otpm_reserved,
            requested_model,
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

        stash: Final = claim_request_stash_for_data(data)

        #########################################################
        # Check if the call type has a specific rate limiter
        # eg. for Batch APIs we need to use the batch rate limiter to read the input file and count the tokens and requests
        #########################################################
        call_type_specific_rate_limiter: Final = self.get_rate_limiter_for_call_type(call_type=call_type)
        if call_type_specific_rate_limiter:
            return await call_type_specific_rate_limiter.async_pre_call_hook(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type=call_type,
            )

        # Get rate limit types from metadata
        metadata: Final = user_api_key_dict.metadata or {}
        rpm_limit_type: Final = metadata.get("rpm_limit_type")
        tpm_limit_type: Final = metadata.get("tpm_limit_type")

        # For dynamic mode, check if the model has recent failures
        model_has_failures = False
        requested_model: Final = data.get("model", None)

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
        descriptors: Final = self._create_rate_limit_descriptors(
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
        self.add_project_io_token_rate_limit_descriptors_from_metadata(
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
            # this pass enforces TPM directly from the post-call counters --
            # except for project ITPM/OTPM descriptors, which are excluded
            # then because _reserve_project_io_tokens_or_raise below charges
            # them unconditionally and counting them here too would
            # double-charge every request.
            parallel_counter_keys: Final = [
                self.create_rate_limit_keys(d["key"], d["value"], "max_parallel_requests")
                for d in descriptors
                if (d.get("rate_limit") or {}).get("max_parallel_requests") is not None
            ]
            parallel_slot_id: Final = uuid.uuid4().hex if parallel_counter_keys else None

            first_pass_descriptors: Final = (
                descriptors
                if self.tpm_reservation_enabled
                else tuple(
                    d for d in descriptors if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
                )
            )
            response: Final = await self.should_rate_limit(
                descriptors=first_pass_descriptors,
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
                stash.rate_limit_response = response
                if parallel_slot_id is not None:
                    stash.parallel_slot = ParallelSlotAcquisition(
                        slot_id=parallel_slot_id,
                        counter_keys=parallel_counter_keys,
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
            configured_tpm_limits: Final = [
                int(v)
                for d in descriptors
                if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
                for v in [(d.get("rate_limit") or {}).get("tokens_per_unit")]
                if v is not None
            ]
            has_tpm_limits: Final = bool(configured_tpm_limits)

            # Populated on a successful combined-TPM reservation below, so the
            # project ITPM/OTPM block further down can roll it back if a
            # different bucket in the same request subsequently hits its
            # limit. Stays empty/0 whenever no combined-TPM reservation was
            # made (or it was over limit, in which case execution never
            # reaches the ITPM/OTPM block -- `_handle_rate_limit_error` raises).
            tpm_reservation_scopes: Sequence[tuple[str, str]] = ()  # rebind-ok: set after successful reservation
            tpm_reservation_amount = 0  # rebind-ok: set after successful reservation

            if has_tpm_limits and self.tpm_reservation_enabled:
                min_configured_tpm_limit: Final = min(configured_tpm_limits)

                configured_output_tokens: Final = get_estimated_output_tokens(
                    user_api_key_dict=user_api_key_dict,
                    model_name=requested_model,
                )

                # When the configured TPM cap is small enough to constrain the
                # no-max_tokens floor, also hard-cap the model output so
                # concurrent unbounded generations can't spend past the limit
                # before post-call reconciliation runs.
                self._apply_implicit_output_cap(
                    data=data,
                    min_configured_limit=min_configured_tpm_limit,
                    call_type=call_type,
                    configured_output_tokens=configured_output_tokens,
                )

                # Floor at 1 token so contentless requests (/responses,
                # tool-call continuations, empty messages) still flow
                # through the atomic counter and get backpressure when at
                # limit. Without this floor, N concurrent contentless
                # requests would all pass pre-call with no enforcement.
                # Post-call reconciliation refunds the over-reservation
                # delta when actual usage comes in below the floor.
                estimated_tokens: Final = max(
                    self._estimate_tokens_for_request(
                        data=data,
                        model=requested_model,
                        min_configured_tpm_limit=min_configured_tpm_limit,
                        call_type=call_type,
                        configured_output_tokens=configured_output_tokens,
                    ),
                    1,
                )

                if configured_output_tokens is not None and estimated_tokens > min_configured_tpm_limit:
                    verbose_proxy_logger.debug(
                        "Reserving %s tokens for model %s (declared %s=%s plus the input estimate) exceeds the "
                        "smallest TPM limit this request is charged against (%s), so it cannot be admitted even "
                        "against an empty window. Lower the declared estimate or raise the TPM limit.",
                        estimated_tokens,
                        requested_model,
                        ESTIMATED_OUTPUT_TOKENS_FIELD,
                        configured_output_tokens,
                        min_configured_tpm_limit,
                    )

                tpm_response: Final = await self.reserve_tpm_tokens(
                    descriptors=descriptors,
                    estimated_tokens=estimated_tokens,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )

                if tpm_response["overall_code"] == "OVER_LIMIT":
                    acquisition: Final = stash.parallel_slot
                    if acquisition is not None:
                        await self._release_parallel_request_slots(
                            acquisition=acquisition,
                            parent_otel_span=user_api_key_dict.parent_otel_span,
                        )
                        stash.parallel_slot = None
                    self._handle_rate_limit_error(
                        response=tpm_response,
                        descriptors=descriptors,
                        requested_model=requested_model,
                    )
                else:
                    # Capture the exact (key, value) scopes the reservation
                    # incremented so post-call reconciliation only applies
                    # the (actual - reserved) delta to those — unreserved
                    # scopes get charged the full actual usage instead.
                    stash.reserved_tokens = estimated_tokens
                    stash.reserved_model = requested_model
                    stash.reserved_scopes = frozenset(
                        (d["key"], d["value"])
                        for d in descriptors
                        if d["key"] not in (PROJECT_ITPM_DESCRIPTOR_KEY, PROJECT_OTPM_DESCRIPTOR_KEY)
                        and (d.get("rate_limit") or {}).get(  # mutable-ok: optional descriptor fallback
                            "tokens_per_unit"
                        )
                        is not None
                    )
                    tpm_reservation_scopes = tuple(  # rebind-ok: record successful reservation scopes
                        stash.reserved_scopes
                    )
                    tpm_reservation_amount = estimated_tokens  # rebind-ok: record successful reservation amount

                    # Merge TPM statuses into the stored rate-limit response
                    # so x-ratelimit-{key}-remaining-tokens / -limit-tokens
                    # headers reach the client. Without this, the RPM-only
                    # response from should_rate_limit (skip_tpm_check=True)
                    # silently drops all token headers.
                    stored_response: Final = stash.rate_limit_response
                    if stored_response is not None:
                        stored_response["statuses"].extend(tpm_response["statuses"])

                    verbose_proxy_logger.debug(
                        "TPM tokens reserved: %s for model %s", estimated_tokens, requested_model
                    )
            await self._reserve_project_io_tokens_or_raise(
                descriptors=descriptors,
                data=data,
                requested_model=requested_model,
                user_api_key_dict=user_api_key_dict,
                tpm_reservation_scopes=tpm_reservation_scopes,
                tpm_reservation_amount=tpm_reservation_amount,
                call_type=call_type,
            )

    def _create_pipeline_operations(
        self,
        key: str,
        value: str,
        rate_limit_type: Literal["requests", "tokens", "max_parallel_requests"],
        total_tokens: int,
    ) -> list["RedisPipelineIncrementOperation"]:
        """
        Create pipeline operations for TPM increments
        """
        pipeline_operations: Final[list[RedisPipelineIncrementOperation]] = []
        counter_key: Final = self.create_rate_limit_keys(
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
        self, usage: Any | None, rate_limit_type: Literal["output", "input", "total"]
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
                    prompt_details: Final = usage.get("prompt_tokens_details") or {}
                    if isinstance(prompt_details, dict):
                        cached_tokens = prompt_details.get("cached_tokens", 0) or 0

        # Subtract cached tokens for input/total (providers don't count them)
        if cached_tokens > 0:
            total_tokens = max(0, total_tokens - cached_tokens)

        return total_tokens

    @staticmethod
    def _aggregate_only_total_tokens(usage: Usage | ResponseAPIUsage | Mapping[str, object] | None) -> int:
        """Total for usage that carries no input/output split, else 0.

        A source that can only report one number for the whole request (a
        pass-through target pricing its own multi-model call) charges that
        number under every ``token_rate_limit_type``. Splitting it is
        impossible, and reading 0 out of it would leave the window
        uncharged, which is how pass-through traffic slips past a TPM limit
        it is supposed to share.
        """
        if usage is None:
            return 0
        token_counts: Final = (
            (usage.prompt_tokens or 0, usage.completion_tokens or 0, usage.total_tokens or 0)
            if isinstance(usage, Usage)
            else (usage.input_tokens or 0, usage.output_tokens or 0, usage.total_tokens or 0)
            if isinstance(usage, ResponseAPIUsage)
            else (
                usage.get("prompt_tokens") or usage.get("input_tokens") or 0,
                usage.get("completion_tokens") or usage.get("output_tokens") or 0,
                usage.get("total_tokens") or 0,
            )
        )
        prompt_tokens, completion_tokens, total_tokens = token_counts
        if prompt_tokens or completion_tokens or not isinstance(total_tokens, int):
            return 0
        return total_tokens

    @staticmethod
    def _response_usage(
        response_obj: object,
    ) -> Usage | ResponseAPIUsage | Mapping[str, object] | None:
        if isinstance(response_obj, (Usage, ResponseAPIUsage)):
            return response_obj
        if isinstance(
            response_obj,
            (ModelResponse, EmbeddingResponse, TextCompletionResponse, BaseLiteLLMOpenAIResponseObject),
        ):
            usage: Final = getattr(response_obj, "usage", None)
            return usage if isinstance(usage, (Usage, ResponseAPIUsage, dict)) else None
        if isinstance(response_obj, dict):
            nested_usage: Final = response_obj.get("usage")
            if isinstance(nested_usage, (Usage, ResponseAPIUsage, dict)):
                return nested_usage
            return response_obj
        return None

    async def _execute_token_increment_script(
        self,
        pipeline_operations: list["RedisPipelineIncrementOperation"],
    ) -> None:
        """
        Execute token increment script grouped by hash tag for cluster compatibility.
        """
        if self.token_increment_script is None:
            return

        # Group operations by hash tag for Redis cluster compatibility
        operation_keys: Final = [op["key"] for op in pipeline_operations]
        key_groups: Final = self._group_keys_by_hash_tag(operation_keys)

        for _hash_tag, group_keys in key_groups.items():
            # Get operations for this hash tag group
            group_operations = [op for op in pipeline_operations if op["key"] in group_keys]

            keys = []
            args = []

            for op in group_operations:
                # Convert None TTL to 0 for Lua script
                ttl_value = op["ttl"] if op["ttl"] is not None else 0

                verbose_proxy_logger.debug(
                    "Executing TTL-preserving increment for key=%s, increment=%s, ttl=%s",
                    op["key"],
                    op["increment_value"],
                    ttl_value,
                )
                keys.append(op["key"])
                args.extend([op["increment_value"], ttl_value])

            await self.token_increment_script(
                keys=keys,
                args=args,
            )

    async def async_increment_tokens_with_ttl_preservation(
        self,
        pipeline_operations: list["RedisPipelineIncrementOperation"],
        parent_otel_span: Span | None = None,
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
                "Successfully executed TTL-preserving increment for %s keys", len(pipeline_operations)
            )

        except Exception as e:
            verbose_proxy_logger.warning("TTL preservation failed, falling back to regular pipeline: %s", e)
            # Fallback to regular pipeline on error
            await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                increment_list=pipeline_operations,
                litellm_parent_otel_span=parent_otel_span,
            )

    async def _apply_local_window_guarded_token_increments(
        self,
        operations: Sequence[ReservationAwareIncrementOperation],
        parent_otel_span: Span | None = None,
    ) -> None:
        async with self._check_and_increment_lock:
            for operation in operations:
                window_key = operation.get("window_key")
                expected_window_start = operation.get("expected_window_start")
                if window_key is None or expected_window_start is None:
                    continue
                active_window_start = await self.internal_usage_cache.async_get_cache(
                    key=window_key,
                    litellm_parent_otel_span=parent_otel_span,
                    local_only=True,
                )
                if active_window_start is None or str(active_window_start) != expected_window_start:
                    continue
                current_counter = (
                    await self.internal_usage_cache.async_get_cache(
                        key=operation["key"],
                        litellm_parent_otel_span=parent_otel_span,
                        local_only=True,
                    )
                    or 0
                )
                await self.internal_usage_cache.async_set_cache(
                    key=operation["key"],
                    value=float(current_counter) + operation["increment_value"],
                    ttl=operation["ttl"],
                    litellm_parent_otel_span=parent_otel_span,
                    local_only=True,
                )

    async def _apply_redis_window_guarded_token_increments(
        self,
        operations: Sequence[ReservationAwareIncrementOperation],
        parent_otel_span: Span | None = None,
    ) -> None:
        for operation in operations:
            window_key = operation.get("window_key")
            expected_window_start = operation.get("expected_window_start")
            if window_key is None or expected_window_start is None:
                continue
            if self.window_guarded_token_increment_script is not None:
                try:
                    await self.window_guarded_token_increment_script(
                        keys=[  # mutable-ok: Redis script interface requires a key list
                            window_key,
                            operation["key"],
                        ],
                        args=[  # mutable-ok: Redis script interface requires an argument list
                            expected_window_start,
                            operation["increment_value"],
                            operation["ttl"] or 0,
                        ],
                    )
                    continue
                except Exception as e:  # noqa: BLE001  # Redis failures use the plain increment fallback
                    verbose_proxy_logger.warning(
                        "Window-guarded token adjustment failed for %s: %s",
                        operation["key"],
                        e,
                    )
            if operation["increment_value"] > 0:
                await self.internal_usage_cache.async_increment_cache(
                    key=operation["key"],
                    value=operation["increment_value"],
                    litellm_parent_otel_span=parent_otel_span,
                    ttl=operation["ttl"],
                )

    async def async_increment_reservation_aware_tokens(
        self,
        pipeline_operations: Sequence[ReservationAwareIncrementOperation],
        parent_otel_span: Span | None = None,
    ) -> None:
        for operation in pipeline_operations:
            if operation.get("window_key") is None or operation.get("expected_window_start") is None:
                await self.internal_usage_cache.async_increment_cache(
                    key=operation["key"],
                    value=operation["increment_value"],
                    litellm_parent_otel_span=parent_otel_span,
                    ttl=operation["ttl"],
                )
        local_guarded_operations: Final = tuple(
            operation
            for operation in pipeline_operations
            if operation.get("window_key") is not None
            and operation.get("expected_window_start") is not None
            and operation.get("reservation_backend") == "local"
        )
        redis_guarded_operations: Final = tuple(
            operation
            for operation in pipeline_operations
            if operation.get("window_key") is not None
            and operation.get("expected_window_start") is not None
            and operation.get("reservation_backend") != "local"
        )
        if local_guarded_operations:
            await self._apply_local_window_guarded_token_increments(
                operations=local_guarded_operations,
                parent_otel_span=parent_otel_span,
            )
        if redis_guarded_operations:
            await self._apply_redis_window_guarded_token_increments(
                operations=redis_guarded_operations,
                parent_otel_span=parent_otel_span,
            )

    def get_rate_limit_type(self) -> Literal["output", "input", "total"]:
        from litellm.proxy.proxy_server import general_settings

        specified_rate_limit_type: Final = general_settings.get("token_rate_limit_type", "total")
        if specified_rate_limit_type not in [
            "output",
            "input",
            "total",
        ]:
            return "total"  # default to total
        return specified_rate_limit_type

    @staticmethod
    def _merge_ratelimit_statuses_into_additional_headers(
        additional_headers: dict[str, object],
        statuses: list[RateLimitStatus],
    ) -> dict[str, object]:
        """
        Return ``additional_headers`` extended with
        ``x-ratelimit-{descriptor_key}-{remaining|limit}-{rate_limit_type}``
        entries. Non-mutating so callers pick their own target dict.
        """
        merged: Final[dict[str, object]] = dict(additional_headers)
        for status in statuses:
            prefix = f"x-ratelimit-{status['descriptor_key']}"
            merged[f"{prefix}-remaining-{status['rate_limit_type']}"] = status["limit_remaining"]
            merged[f"{prefix}-limit-{status['rate_limit_type']}"] = status["current_limit"]
        return merged

    @staticmethod
    def _resolve_rerank_token_usage(response_obj: object) -> tuple[int, int, bool] | None:
        if not isinstance(response_obj, RerankResponse) or response_obj.meta is None:
            return None

        rerank_tokens: Final = response_obj.meta.get("tokens")  # pyright: ignore[reportUnknownMemberType]  # TypedDict's optional generic metadata widens get overloads
        if rerank_tokens is not None:
            input_tokens: Final = rerank_tokens.get("input_tokens") or 0  # pyright: ignore[reportUnknownMemberType]  # token fields are typed integers despite the generic get overload
            output_tokens: Final = rerank_tokens.get("output_tokens") or 0  # pyright: ignore[reportUnknownMemberType]  # token fields are typed integers despite the generic get overload
            if input_tokens or output_tokens:
                return max(0, input_tokens), max(0, output_tokens), True

        billed_units: Final = response_obj.meta.get("billed_units")  # pyright: ignore[reportUnknownMemberType]  # TypedDict's optional generic metadata widens get overloads
        if billed_units is not None:
            total_tokens: Final = billed_units.get("total_tokens") or 0  # pyright: ignore[reportUnknownMemberType]  # billed total is a typed integer despite the generic get overload
            if total_tokens:
                return max(0, total_tokens), 0, True
        return None

    def _resolve_io_token_reconcile_usage(
        self,
        response_obj: object,
    ) -> tuple[int, int, bool]:
        """
        Resolve ``(billable_input_tokens, completion_tokens, usage_resolved)``
        for ITPM/OTPM reconciliation. Cache-read tokens are excluded from
        billable input -- Bedrock Mantle doesn't count them toward ITPM --
        but they're untouched everywhere else (cost/usage logging still sees
        the full prompt token count).
        """
        rerank_usage: Final = self._resolve_rerank_token_usage(response_obj)
        if rerank_usage is not None:
            return rerank_usage

        usage: Final = self._response_usage(response_obj)

        if isinstance(usage, Usage):
            prompt_tokens: Final = usage.prompt_tokens or 0
            completion_tokens: Final = usage.completion_tokens or 0
            cached_tokens: Final = (
                getattr(usage.prompt_tokens_details, "cached_tokens", 0) or 0
                if usage.prompt_tokens_details is not None
                else 0
            )
            if prompt_tokens == 0 and completion_tokens == 0:
                return 0, 0, False
            return max(0, prompt_tokens - cached_tokens), completion_tokens, True

        if isinstance(usage, ResponseAPIUsage):
            response_input_tokens: Final = usage.input_tokens or 0
            response_output_tokens: Final = usage.output_tokens or 0
            response_cached_tokens: Final = (
                usage.input_tokens_details.cached_tokens or 0 if usage.input_tokens_details is not None else 0
            )
            if response_input_tokens == 0 and response_output_tokens == 0:
                return 0, 0, False
            return max(0, response_input_tokens - response_cached_tokens), response_output_tokens, True

        if isinstance(usage, Mapping):
            raw_prompt_tokens: Final = usage.get("prompt_tokens") or usage.get("input_tokens") or 0
            raw_completion_tokens: Final = usage.get("completion_tokens") or usage.get("output_tokens") or 0
            mapped_prompt_tokens: Final = raw_prompt_tokens if isinstance(raw_prompt_tokens, int) else 0
            mapped_completion_tokens: Final = raw_completion_tokens if isinstance(raw_completion_tokens, int) else 0
            prompt_details: Final = usage.get("prompt_tokens_details") or usage.get("input_tokens_details")
            raw_cached_tokens: Final = (
                (prompt_details.get("cached_tokens", 0) if isinstance(prompt_details, dict) else 0)
                or usage.get("cache_read_input_tokens")
                or 0
            )
            mapped_cached_tokens: Final = raw_cached_tokens if isinstance(raw_cached_tokens, int) else 0
            if mapped_prompt_tokens == 0 and mapped_completion_tokens == 0:
                return 0, 0, False
            return max(0, mapped_prompt_tokens - mapped_cached_tokens), mapped_completion_tokens, True

        return 0, 0, False

    def _build_io_token_reservation_ops(
        self,
        kwargs: object,
        response_obj: object,
    ) -> Sequence[RedisPipelineIncrementOperation]:
        """
        Reconcile project ITPM/OTPM reservations to actual usage on success:
        ITPM to billable input tokens, OTPM to actual completion tokens.
        Reuses ``_build_reservation_aware_tpm_ops``'s delta pattern -- ITPM/OTPM
        are stored in the same ":tokens" cache bucket as combined TPM, just
        under distinct scope keys, so the reservation-aware increment math is
        identical; only the usage fields being reconciled against differ.
        """
        if not isinstance(kwargs, dict):
            return ()
        stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(kwargs))
        if stash is None:
            return ()

        itpm_reserved: Final = stash.itpm_reserved_tokens
        otpm_reserved: Final = stash.otpm_reserved_tokens
        if itpm_reserved <= 0 and otpm_reserved <= 0:
            return ()

        response_usage: Final = self._resolve_io_token_reconcile_usage(response_obj)
        combined_usage: Final = self._resolve_io_token_reconcile_usage(kwargs.get("combined_usage_object"))
        aggregate_total: Final = self._aggregate_only_total_tokens(
            self._response_usage(response_obj)
        ) or self._aggregate_only_total_tokens(self._response_usage(kwargs.get("combined_usage_object")))

        if not response_usage[2] and not combined_usage[2] and aggregate_total <= 0 and not stash.reservation_released:
            return ()
        resolved_usage: Final = (
            response_usage
            if response_usage[2]
            else combined_usage
            if combined_usage[2]
            else (aggregate_total, aggregate_total, True)
            if aggregate_total > 0
            else (itpm_reserved, otpm_reserved, False)
        )
        billable_input, completion_tokens, _ = resolved_usage

        if stash.reservation_released or (
            not stash.itpm_reserved_window_identities and not stash.otpm_reserved_window_identities
        ):
            return self._build_reservation_aware_tpm_ops(
                targets=tuple(stash.itpm_reserved_scopes),
                reserved_scopes=frozenset() if stash.reservation_released else stash.itpm_reserved_scopes,
                actual_tokens=billable_input,
                reserved_tokens=0 if stash.reservation_released else itpm_reserved,
            ) + self._build_reservation_aware_tpm_ops(
                targets=tuple(stash.otpm_reserved_scopes),
                reserved_scopes=frozenset() if stash.reservation_released else stash.otpm_reserved_scopes,
                actual_tokens=completion_tokens,
                reserved_tokens=0 if stash.reservation_released else otpm_reserved,
            )

        itpm_ops: Final[Sequence[ReservationAwareIncrementOperation]] = (
            self._build_project_reservation_ops(
                targets=tuple(stash.itpm_reserved_scopes),
                reserved_scopes=frozenset() if stash.reservation_released else stash.itpm_reserved_scopes,
                actual_tokens=billable_input,
                reserved_tokens=itpm_reserved,
                reservation_window_identities=stash.itpm_reserved_window_identities,
            )
            if itpm_reserved > 0
            else ()
        )
        otpm_ops: Final[Sequence[ReservationAwareIncrementOperation]] = (
            self._build_project_reservation_ops(
                targets=tuple(stash.otpm_reserved_scopes),
                reserved_scopes=frozenset() if stash.reservation_released else stash.otpm_reserved_scopes,
                actual_tokens=completion_tokens,
                reserved_tokens=otpm_reserved,
                reservation_window_identities=stash.otpm_reserved_window_identities,
            )
            if otpm_reserved > 0
            else ()
        )
        return tuple((*itpm_ops, *otpm_ops))

    def _collect_tpm_scope_targets(
        self,
        standard_logging_metadata: dict[str, Any],
        kwargs: Any,
        model_group: str | None,
    ) -> list[tuple[str, str]]:
        """
        Enumerate every (scope_key, scope_value) pair that *might* carry a
        TPM counter for this request — independent of whether each scope had
        a configured TPM limit at pre-call. Reservation awareness happens at
        the emitter; this helper just lists the candidate scopes so callers
        can split reserved-vs-unreserved.
        """
        user_api_key: Final = standard_logging_metadata.get("user_api_key_hash")
        user_api_key_user_id: Final = standard_logging_metadata.get("user_api_key_user_id")
        user_api_key_team_id: Final = standard_logging_metadata.get("user_api_key_team_id")
        user_api_key_organization_id: Final = standard_logging_metadata.get("user_api_key_org_id")
        user_api_key_project_id: Final = standard_logging_metadata.get("user_api_key_project_id")
        user_api_key_end_user_id: Final = (
            kwargs.get("user") if isinstance(kwargs, dict) else None
        ) or standard_logging_metadata.get("user_api_key_end_user_id")
        agent_id: Final = standard_logging_metadata.get("agent_id")
        session_id: Final = standard_logging_metadata.get("session_id") or standard_logging_metadata.get("trace_id")

        targets: Final[list[tuple[str, str]]] = []
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
        targets: Sequence[tuple[str, str]],
        reserved_scopes: Set[tuple[str, str]],
        actual_tokens: int,
        reserved_tokens: int,
    ) -> list[RedisPipelineIncrementOperation]:
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
        ops: Final[list[RedisPipelineIncrementOperation]] = []
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

    def _build_project_reservation_op(
        self,
        scope: tuple[str, str],
        reserved_scopes: Set[tuple[str, str]],
        actual_tokens: int,
        reserved_tokens: int,
        reservation_window_identities: frozenset[tuple[str, str, Literal["redis", "local"]]],
    ) -> ReservationAwareIncrementOperation | None:
        scope_key, scope_value = scope
        is_reserved_scope: Final = scope in reserved_scopes
        increment: Final = actual_tokens - reserved_tokens if is_reserved_scope else actual_tokens
        if increment == 0:
            return None
        counter_key: Final = self.create_rate_limit_keys(scope_key, scope_value, "tokens")
        window_identity: Final = next(
            (
                (window_start, backend)
                for identity_counter_key, window_start, backend in reservation_window_identities
                if identity_counter_key == counter_key
            ),
            None,
        )
        if not is_reserved_scope or window_identity is None:
            return ReservationAwareIncrementOperation(
                key=counter_key,
                increment_value=increment,
                ttl=self.window_size,
            )
        return ReservationAwareIncrementOperation(
            key=counter_key,
            increment_value=increment,
            ttl=self.window_size,
            window_key=f"{{{scope_key}:{scope_value}}}:window",
            expected_window_start=window_identity[0],
            reservation_backend=window_identity[1],
        )

    def _build_project_reservation_ops(
        self,
        targets: Sequence[tuple[str, str]],
        reserved_scopes: Set[tuple[str, str]],
        actual_tokens: int,
        reserved_tokens: int,
        reservation_window_identities: frozenset[tuple[str, str, Literal["redis", "local"]]],
    ) -> tuple[ReservationAwareIncrementOperation, ...]:
        return tuple(
            operation
            for scope in targets
            if (
                operation := self._build_project_reservation_op(
                    scope=scope,
                    reserved_scopes=reserved_scopes,
                    actual_tokens=actual_tokens,
                    reserved_tokens=reserved_tokens,
                    reservation_window_identities=reservation_window_identities,
                )
            )
            is not None
        )

    def _build_success_event_pipeline_operations(
        self,
        kwargs: Any,
        response_obj: Any,
        rate_limit_type: Literal["output", "input", "total"],
    ) -> list[RedisPipelineIncrementOperation]:
        """Build Redis pipeline increment ops for TPM / parallel-request counters."""
        from litellm.litellm_core_utils.core_helpers import get_litellm_metadata_from_kwargs
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )

        # Get metadata from standard_logging_object - this correctly handles both
        # 'metadata' and 'litellm_metadata' fields from litellm_params
        standard_logging_object: Final = kwargs.get("standard_logging_object") or {}
        request_metadata: Final = get_litellm_metadata_from_kwargs(kwargs)
        if request_metadata.get(INTERNAL_CALL_ORIGIN_METADATA_KEY):
            # Internal sub-calls bill spend to the caller but are not the caller's
            # traffic; charging them here would let background evals eat TPM headroom.
            return []
        standard_logging_metadata: Final = standard_logging_object.get("metadata") or {}

        model_group: Final = get_model_group_from_litellm_kwargs(kwargs)

        # Get total tokens from response. Responses LiteLLM does not model
        # (e.g. pass-through, whose usage is reported by the upstream rather
        # than parsed out of the body) carry their usage in
        # ``combined_usage_object`` instead, and would otherwise never charge
        # the TPM window.
        _usage: Usage | dict | None = None
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
        else:
            _combined_usage: Final = kwargs.get("combined_usage_object")
            if isinstance(_combined_usage, Usage):
                _usage = _combined_usage
        total_tokens = self._get_total_tokens_from_usage(usage=_usage, rate_limit_type=rate_limit_type)
        if total_tokens == 0:
            total_tokens = self._aggregate_only_total_tokens(usage=_usage)

        stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(kwargs))
        reserved_tokens: Final = stash.reserved_tokens if stash is not None else 0
        reserved_model: Final = stash.reserved_model if stash is not None else None
        reserved_scopes: Final[frozenset[tuple[str, str]]] = stash.reserved_scopes if stash is not None else frozenset()
        # Reconciliation must target the same model-scoped counter that the
        # pre-call reservation incremented. If a reservation was made,
        # ``reserved_model`` is authoritative; otherwise fall back to the
        # router's ``model_group`` (covers the no-reservation charge path).
        reconcile_model: Final = reserved_model or model_group

        pipeline_operations: Final[list[RedisPipelineIncrementOperation]] = []

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
        targets: Final = self._collect_tpm_scope_targets(
            standard_logging_metadata=standard_logging_metadata,
            kwargs=kwargs,
            model_group=reconcile_model,
        )
        if reserved_tokens > 0 and total_tokens < reserved_tokens:
            verbose_proxy_logger.debug(
                "Releasing unused TPM budget on success: reserved=%s, actual=%s, release=%s",
                reserved_tokens,
                total_tokens,
                reserved_tokens - total_tokens,
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

        rate_limit_type: Final = self.get_rate_limit_type()

        litellm_parent_otel_span: Final[Span | None] = _get_parent_otel_span_from_kwargs(kwargs)
        try:
            verbose_proxy_logger.debug("INSIDE parallel request limiter ASYNC SUCCESS LOGGING")

            stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(kwargs))
            acquisition: Final = stash.parallel_slot if stash is not None else None
            if stash is not None and acquisition is not None:
                await self._release_parallel_request_slots(
                    acquisition=acquisition,
                    parent_otel_span=litellm_parent_otel_span,
                )
                stash.parallel_slot = None

            pipeline_operations: Final = self._build_success_event_pipeline_operations(
                kwargs=kwargs,
                response_obj=response_obj,
                rate_limit_type=rate_limit_type,
            )
            if pipeline_operations:
                await self.async_increment_tokens_with_ttl_preservation(
                    pipeline_operations=pipeline_operations,
                    parent_otel_span=litellm_parent_otel_span,
                )
            io_token_operations: Final = self._build_io_token_reservation_ops(
                kwargs=kwargs,
                response_obj=response_obj,
            )
            if io_token_operations:
                if isinstance(io_token_operations, list):
                    await self.async_increment_tokens_with_ttl_preservation(
                        pipeline_operations=io_token_operations,
                        parent_otel_span=litellm_parent_otel_span,
                    )
                else:
                    await self.async_increment_reservation_aware_tokens(
                        pipeline_operations=io_token_operations,
                        parent_otel_span=litellm_parent_otel_span,
                    )

        except Exception as e:
            verbose_proxy_logger.exception("Error in rate limit success event: %s", e)

    async def async_logging_hook(
        self,
        kwargs: dict,
        result: object,
        call_type: str,
    ) -> tuple[dict, object]:
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
        kwargs: object,
        response_obj: object,
    ) -> None:
        """
        Copy the stashed ``RateLimitResponse`` into the SLP's
        ``hidden_params.additional_headers`` and the response object's
        ``_hidden_params.additional_headers`` (when the latter is a dict).
        """
        if not isinstance(kwargs, dict):
            return

        stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(kwargs))
        rate_limit_response: Final = stash.rate_limit_response if stash is not None else None
        statuses: Final = rate_limit_response["statuses"] if rate_limit_response is not None else []
        if not statuses:
            return

        standard_logging_object: Final = kwargs.get("standard_logging_object")
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

        response_hidden: Final = getattr(response_obj, "_hidden_params", None)
        if isinstance(response_hidden, dict):
            existing = response_hidden.get("additional_headers")
            response_hidden["additional_headers"] = self._merge_ratelimit_statuses_into_additional_headers(
                additional_headers=existing if isinstance(existing, dict) else {},
                statuses=statuses,
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
            litellm_parent_otel_span: Final[Span | None] = _get_parent_otel_span_from_kwargs(kwargs)

            pipeline_operations: Final[list[RedisPipelineIncrementOperation]] = []

            stash: Final = get_request_stash_for_call(_call_id_from_callback_kwargs(kwargs))
            acquisition: Final = stash.parallel_slot if stash is not None else None
            if stash is not None and acquisition is not None:
                await self._release_parallel_request_slots(
                    acquisition=acquisition,
                    parent_otel_span=litellm_parent_otel_span,
                )
                stash.parallel_slot = None

            # Skip the reservation refund if async_post_call_failure_hook
            # already released it (proxy-level rejection that also bubbles up
            # here as an LLM-error callback). max_parallel_requests is its
            # own counter and is always decremented per call.
            reserved_tokens, itpm_reserved, otpm_reserved = (
                (0, 0, 0)
                if stash is None or stash.reservation_released
                else (stash.reserved_tokens, stash.itpm_reserved_tokens, stash.otpm_reserved_tokens)
            )

            if stash is not None and reserved_tokens > 0:
                verbose_proxy_logger.debug("Releasing reserved TPM tokens on failure: %s", reserved_tokens)
                # Refund only against the scopes the reservation actually
                # charged. _build_reservation_aware_tpm_ops with
                # actual_tokens=0 emits -reserved on reserved scopes and 0
                # on unreserved (skipped), so unreserved scopes can't drift
                # negative.
                pipeline_operations.extend(
                    self._build_reservation_aware_tpm_ops(
                        targets=list(stash.reserved_scopes),
                        reserved_scopes=stash.reserved_scopes,
                        actual_tokens=0,
                        reserved_tokens=reserved_tokens,
                    )
                )

            # Refund project ITPM/OTPM reservations the same way -- full
            # refund, since a failed call has no billable usage to reconcile
            # against.
            itpm_operations: Final = (
                self._build_project_reservation_ops(
                    targets=tuple(stash.itpm_reserved_scopes),
                    reserved_scopes=stash.itpm_reserved_scopes,
                    actual_tokens=0,
                    reserved_tokens=itpm_reserved,
                    reservation_window_identities=stash.itpm_reserved_window_identities,
                )
                if stash is not None and itpm_reserved > 0 and stash.itpm_reserved_window_identities
                else self._build_reservation_aware_tpm_ops(
                    targets=tuple(stash.itpm_reserved_scopes),
                    reserved_scopes=stash.itpm_reserved_scopes,
                    actual_tokens=0,
                    reserved_tokens=itpm_reserved,
                )
                if stash is not None and itpm_reserved > 0
                else ()
            )

            otpm_operations: Final = (
                self._build_project_reservation_ops(
                    targets=tuple(stash.otpm_reserved_scopes),
                    reserved_scopes=stash.otpm_reserved_scopes,
                    actual_tokens=0,
                    reserved_tokens=otpm_reserved,
                    reservation_window_identities=stash.otpm_reserved_window_identities,
                )
                if stash is not None and otpm_reserved > 0 and stash.otpm_reserved_window_identities
                else self._build_reservation_aware_tpm_ops(
                    targets=tuple(stash.otpm_reserved_scopes),
                    reserved_scopes=stash.otpm_reserved_scopes,
                    actual_tokens=0,
                    reserved_tokens=otpm_reserved,
                )
                if stash is not None and otpm_reserved > 0
                else ()
            )

            if pipeline_operations:
                await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                    increment_list=pipeline_operations,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                )
            for project_operations in (itpm_operations, otpm_operations):
                if isinstance(project_operations, list):
                    await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                        increment_list=project_operations,
                        litellm_parent_otel_span=litellm_parent_otel_span,
                    )
                elif project_operations:
                    await self.async_increment_reservation_aware_tokens(
                        pipeline_operations=project_operations,
                        parent_otel_span=litellm_parent_otel_span,
                    )
            if stash is not None and (reserved_tokens > 0 or itpm_reserved > 0 or otpm_reserved > 0):
                stash.reservation_released = True
        except Exception as e:
            verbose_proxy_logger.exception("Error in rate limit failure event: %s", e)

    async def async_release_max_parallel_requests_on_disconnect(
        self,
        user_api_key_dict: UserAPIKeyAuth,
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
        TTL prunes it. The stashed acquisition's presence (not the key
        object's current max_parallel_requests configuration, which can
        change mid-request) decides whether there is anything to release.
        """
        stash: Final = get_request_stash()
        if stash is None or stash.parallel_slot is None:
            return

        await self._release_parallel_request_slots(
            acquisition=stash.parallel_slot,
            parent_otel_span=None,
        )
        stash.parallel_slot = None

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response):
        """
        Post-call hook to update rate limit headers in the response.
        """
        try:
            from pydantic import BaseModel

            stash: Final = get_request_stash()
            litellm_proxy_rate_limit_response: Final = stash.rate_limit_response if stash is not None else None

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

                    _additional_headers: Final = self._merge_ratelimit_statuses_into_additional_headers(
                        additional_headers=_hidden_params.get("additional_headers", {}) or {},
                        statuses=litellm_proxy_rate_limit_response["statuses"],
                    )

                    setattr(
                        response,
                        "_hidden_params",
                        {**_hidden_params, "additional_headers": _additional_headers},
                    )

        except Exception as e:
            verbose_proxy_logger.exception("Error in rate limit post-call hook: %s", e)

        try:
            await self._handle_batch_enqueued_post_call(user_api_key_dict=user_api_key_dict, response=response)
        except Exception as e:  # noqa: BLE001  # post-call batch accounting must never fail the response
            verbose_proxy_logger.exception("Error in batch enqueued-token post-call hook: %s", e)

    async def _handle_batch_enqueued_post_call(self, user_api_key_dict: UserAPIKeyAuth, response: object) -> None:
        view: Final = batch_response_view(response)
        if view is None:
            return
        span: Final = user_api_key_dict.parent_otel_span
        stash: Final = get_request_stash()
        if stash is not None and stash.batch_enqueued_reservation is not None:
            await self.batch_enqueued_token_store.save_reservation(
                batch_id=canonical_provider_batch_id(view.id),
                reservation=stash.batch_enqueued_reservation,
                litellm_parent_otel_span=span,
            )
            stash.batch_enqueued_reservation = None
        if view.status.lower() in BATCH_ENQUEUED_REFUND_STATUSES:
            popped: Final = await self.batch_enqueued_token_store.pop_reservation(
                batch_id=canonical_provider_batch_id(view.id),
                litellm_parent_otel_span=span,
            )
            if popped is not None:
                await self.batch_enqueued_token_store.refund(reservation=popped, litellm_parent_otel_span=span)

    async def async_post_call_failure_hook(
        self,
        request_data: dict,
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        """
        Release the parallel-request slot and any TPM/ITPM/OTPM reservation
        when the request is rejected after the pre-call hook acquired them
        but before the LLM call ran (e.g. a downstream guardrail/auth hook
        raised). Without this, those resources are stranded —
        async_log_failure_event is a litellm completion-level callback and
        never fires for proxy-side rejections, so a leaked slot would occupy
        the gauge for the full PARALLEL_REQUEST_SLOT_TTL_SECONDS.

        Idempotent: the slot release clears the stashed acquisition (and slot
        removal is a no-op ZREM on a second run), and the TPM/ITPM/OTPM
        refund is guarded by the stash's ``reservation_released`` flag — if
        both this hook and async_log_failure_event end up running in the same
        flow, only the first release/refund applies.
        """
        try:
            stash: Final = get_request_stash()
            if stash is None:
                return
            if stash.parallel_slot is not None:
                await self._release_parallel_request_slots(
                    acquisition=stash.parallel_slot,
                    parent_otel_span=user_api_key_dict.parent_otel_span,
                )
                stash.parallel_slot = None

            if stash.batch_enqueued_reservation is not None:
                await self.batch_enqueued_token_store.refund(
                    reservation=stash.batch_enqueued_reservation,
                    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                )
                stash.batch_enqueued_reservation = None

            if stash.reservation_released:
                return
            reserved_tokens: Final = stash.reserved_tokens
            itpm_reserved: Final = stash.itpm_reserved_tokens
            otpm_reserved: Final = stash.otpm_reserved_tokens
            if reserved_tokens <= 0 and itpm_reserved <= 0 and otpm_reserved <= 0:
                return

            combined_ops: Final = (
                self._build_reservation_aware_tpm_ops(
                    targets=tuple(stash.reserved_scopes),
                    reserved_scopes=stash.reserved_scopes,
                    actual_tokens=0,
                    reserved_tokens=reserved_tokens,
                )
                if reserved_tokens > 0
                else ()
            )
            itpm_ops: Final = (
                self._build_project_reservation_ops(
                    targets=tuple(stash.itpm_reserved_scopes),
                    reserved_scopes=stash.itpm_reserved_scopes,
                    actual_tokens=0,
                    reserved_tokens=itpm_reserved,
                    reservation_window_identities=stash.itpm_reserved_window_identities,
                )
                if itpm_reserved > 0 and stash.itpm_reserved_window_identities
                else self._build_reservation_aware_tpm_ops(
                    targets=tuple(stash.itpm_reserved_scopes),
                    reserved_scopes=stash.itpm_reserved_scopes,
                    actual_tokens=0,
                    reserved_tokens=itpm_reserved,
                )
                if itpm_reserved > 0
                else ()
            )
            otpm_ops: Final = (
                self._build_project_reservation_ops(
                    targets=tuple(stash.otpm_reserved_scopes),
                    reserved_scopes=stash.otpm_reserved_scopes,
                    actual_tokens=0,
                    reserved_tokens=otpm_reserved,
                    reservation_window_identities=stash.otpm_reserved_window_identities,
                )
                if otpm_reserved > 0 and stash.otpm_reserved_window_identities
                else self._build_reservation_aware_tpm_ops(
                    targets=tuple(stash.otpm_reserved_scopes),
                    reserved_scopes=stash.otpm_reserved_scopes,
                    actual_tokens=0,
                    reserved_tokens=otpm_reserved,
                )
                if otpm_reserved > 0
                else ()
            )
            if combined_ops or itpm_ops or otpm_ops:
                verbose_proxy_logger.debug(
                    "Releasing reserved tokens on proxy-level rejection: tpm=%s, itpm=%s, otpm=%s",
                    reserved_tokens,
                    itpm_reserved,
                    otpm_reserved,
                )
            if combined_ops:
                await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                    increment_list=combined_ops,
                    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                )
            for project_ops in (itpm_ops, otpm_ops):
                if isinstance(project_ops, list):
                    await self.internal_usage_cache.dual_cache.async_increment_cache_pipeline(
                        increment_list=project_ops,
                        litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                    )
                elif project_ops:
                    await self.async_increment_reservation_aware_tokens(
                        pipeline_operations=project_ops,
                        parent_otel_span=user_api_key_dict.parent_otel_span,
                    )
            stash.reservation_released = True
        except Exception as e:
            verbose_proxy_logger.exception("Error releasing TPM reservation on post-call failure: %s", e)
        return
