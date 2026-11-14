from asyncio import Future
from collections.abc import Callable, Mapping, Sequence
from typing import Generic, Literal, Never, NotRequired, Protocol, TypedDict, TypeVar, final

import httpx
from pydantic import JsonValue
from typing_extensions import disjoint_base

from litellm.redis.store_settings import RedisBreakerSettings, RedisStoreSettings
from litellm.redis.stores.router_session_pins.keys import PinKey
from litellm.rust_bridge.execution import Execution
from litellm.rust_bridge.public_call import NativeCall

_T_co = TypeVar("_T_co", covariant=True)

class RustBridgeDeclined(Exception): ...
class RustUpstreamError(Exception): ...
class ForkedAfterNativeRuntimeStarted(RuntimeError): ...
class ProcessReservedForForking(RuntimeError): ...
class NativeRedisStoreError(Exception): ...

def trace_encode_error(message: str) -> bytes: ...
def bundled_model_cost_map() -> bytes: ...
def native_redis_breaker_states() -> list[str]: ...
def unsupported_tls_settings(ssl_security_level: str | None = None, ssl_ecdh_curve: str | None = None) -> list[str]: ...
@final
class NativeCostMapSource:
    def __new__(cls, url: str, *, use_http_settings: bool) -> NativeCostMapSource: ...
    def fetch(self, policy: NativeFetchPolicy) -> Future[NativeFetched | NativeFetchFailed]: ...
    def fetch_blocking(
        self, policy: NativeFetchPolicy, resume_from: NativeFetchFailed | None = None
    ) -> NativeFetched | NativeFetchFailed: ...

@final
class NativeFetchPolicy:
    def __new__(
        cls,
        *,
        max_attempts: int,
        max_wait_seconds: float,
        timeout_seconds: float,
        min_delay_seconds: float,
    ) -> NativeFetchPolicy: ...

@final
class NativeFetched:
    @property
    def body(self) -> bytes: ...
    @property
    def etag(self) -> str | None: ...

@final
class NativeFetchFailed:
    @property
    def reason(self) -> str: ...
    @property
    def retryable(self) -> bool: ...
    @property
    def retry_after(self) -> float | None: ...

@final
class NativeInstalled:
    @property
    def rows(self) -> dict[str, dict[str, object]]: ...
    @property
    def model_count(self) -> int: ...
    @property
    def fallback_generalizations(self) -> JsonValue: ...

@final
class NativeRegistered:
    @property
    def unmatched(self) -> dict[str, Mapping[str, object]]: ...

@final
class NativeTableSwapped: ...

@final
class NativeRejected:
    @property
    def reason(self) -> str: ...
    @property
    def integrity(self) -> bool: ...

@final
class NativeProvenance:
    @property
    def revision(self) -> str: ...
    @property
    def etag(self) -> str | None: ...

@final
class NativeModelCostMap:
    @staticmethod
    def bundled(min_model_count: int, max_shrink_ratio: float) -> NativeModelCostMap: ...
    def install(self, body: bytes, etag: str | None = None) -> NativeInstalled | NativeRejected: ...
    def install_bundled(self) -> NativeInstalled: ...
    @staticmethod
    def finalize(body: bytes) -> NativeInstalled | NativeRejected: ...
    def register(
        self,
        rows: dict[str, dict[str, object]] | None,
        entries: Mapping[str, Mapping[str, object]],
        persist: bool,
        custom_llm_provider: str | None = None,
    ) -> NativeRegistered | NativeTableSwapped: ...
    def reapply(self, rows: dict[str, dict[str, object]]) -> NativeRegistered | NativeTableSwapped: ...
    def registered_row(self, key: str) -> dict[str, object] | None: ...
    def registered_keys(self) -> list[str]: ...
    def loaded(self) -> dict[str, dict[str, object]]: ...
    def find_key(self, rows: dict[str, dict[str, object]], name: str) -> str | None: ...
    def keys_in(self, rows: dict[str, dict[str, object]], provider: str) -> list[str]: ...
    def invalidate(self) -> None: ...
    def snapshot(self) -> NativeModelCostMapState: ...
    def restore(self, state: NativeModelCostMapState) -> None: ...
    @property
    def provenance(self) -> NativeProvenance: ...

@final
class NativeModelCostMapState: ...

@final
class NativeClickHouseSpendConfig:
    def __new__(cls, database: str, url: str, retention_days: int) -> NativeClickHouseSpendConfig: ...

@final
class NativeClickHouseSpendStorage:
    def __new__(cls, config: NativeClickHouseSpendConfig) -> NativeClickHouseSpendStorage: ...
    def ensure_schema(self) -> Future[None]: ...
    def insert_rows(self, rows: Sequence[Mapping[str, object]]) -> Future[None]: ...

class _NativeHealthState(TypedDict):
    is_healthy: bool
    timestamp: float
    reason: str

class _NativeCooldown(TypedDict):
    exception_received: str
    status_code: str
    timestamp: float
    cooldown_time: float

@final
class NativeAdmission:
    def __new__(cls, _uninstantiable: Never, /) -> Never: ...
    def succeeded(self) -> None: ...
    def failed(self, timeout: bool) -> None: ...

@final
class NativeRedisBreaker:
    def __new__(cls, settings: RedisBreakerSettings) -> NativeRedisBreaker: ...
    @staticmethod
    def shared(redis_kwargs: Mapping[str, object], settings: RedisBreakerSettings) -> NativeRedisBreaker: ...
    def admit(self) -> NativeAdmission | None: ...
    def failed(self, timeout: bool) -> None: ...
    @property
    def state(self) -> Literal["closed", "open", "half_open"]: ...

@final
class NativeRedisClient:
    def __new__(cls, _uninstantiable: Never, /) -> Never: ...
    @staticmethod
    def connect(settings: RedisStoreSettings) -> NativeRedisClient | None: ...
    @property
    def default_ttl_seconds(self) -> float: ...

@final
class NativeRequestRedis:
    def __new__(cls, post_call_deadline_seconds: float) -> NativeRequestRedis: ...
    def flush_pre(self) -> Future[None]: ...
    def flush_post(self) -> Future[None]: ...

@final
class NativePending(Generic[_T_co]):
    def resolve(self) -> Future[_T_co]: ...

@disjoint_base
class NativeSend:
    @final
    class Now(NativeSend):
        __match_args__: tuple[()] = ()

        def __new__(cls) -> NativeSend.Now: ...

    @final
    class PreCall(NativeSend):
        __match_args__: tuple[Literal["_0"]] = ("_0",)

        def __new__(cls, _0: NativeRequestRedis) -> NativeSend.PreCall: ...

    @final
    class PostCall(NativeSend):
        __match_args__: tuple[Literal["_0"]] = ("_0",)

        def __new__(cls, _0: NativeRequestRedis) -> NativeSend.PostCall: ...

@final
class NativeRouterCooldowns:
    def __new__(cls, client: NativeRedisClient | None) -> NativeRouterCooldowns: ...
    def attach(self, client: NativeRedisClient) -> bool: ...
    @property
    def has_redis(self) -> bool: ...
    def put(self, model_id: str, cooldown: Mapping[str, object], ttl_seconds: float) -> None: ...
    def increment_allowed_fails(self, model_id: str, suffix: str | None, ttl_seconds: float) -> int: ...
    def async_put(self, model_id: str, cooldown: Mapping[str, object], ttl_seconds: float) -> Future[None]: ...
    def async_increment_allowed_fails(self, model_id: str, suffix: str | None, ttl_seconds: float) -> Future[int]: ...
    def active(
        self, model_ids: Sequence[str], send: NativeSend
    ) -> NativePending[list[tuple[str, _NativeCooldown]]]: ...
    def active_sync(self, model_ids: Sequence[str]) -> list[tuple[str, _NativeCooldown]]: ...
    def minimum(self, model_ids: Sequence[str]) -> float | None: ...
    def delete_local(self, model_id: str) -> None: ...
    def flush_local_counters(self) -> None: ...
    def snapshot(self) -> dict[str, _NativeCooldown]: ...

@final
class NativeDeploymentHealth:
    def __new__(cls, client: NativeRedisClient | None) -> NativeDeploymentHealth: ...
    @property
    def has_redis(self) -> bool: ...
    def attach(self, client: NativeRedisClient) -> bool: ...
    def get(self) -> Future[dict[str, _NativeHealthState] | None]: ...
    def get_sync(self) -> dict[str, _NativeHealthState] | None: ...
    def get_shared_sync(self) -> dict[str, _NativeHealthState] | None: ...
    def set_sync(self, states: Mapping[str, object], ttl_seconds: float) -> None: ...

@final
class NativeSensitiveRoutePins:
    def __new__(cls, client: NativeRedisClient | None) -> NativeSensitiveRoutePins: ...
    def get(self, tenant: str, session_id: str, default_ttl_seconds: float) -> Future[str | None]: ...
    def set(self, tenant: str, session_id: str, model: str, ttl_seconds: float) -> Future[None]: ...

class _UsageCounter(Protocol):
    @property
    def scope(self) -> Literal["global_router", "deployment"]: ...
    @property
    def deployment_id(self) -> str: ...
    @property
    def model(self) -> str | None: ...
    @property
    def window(self) -> str: ...
    @property
    def kind(self) -> Literal["tpm", "rpm", "itpm", "otpm", "tpd", "rpd"]: ...

class _CounterIncrement(Protocol):
    @property
    def counter(self) -> _UsageCounter: ...
    @property
    def amount(self) -> float: ...
    @property
    def ttl(self) -> float | None: ...

@final
class NativeUsageRead:
    @property
    def read(self) -> NativePending[list[float | None]]: ...
    def release(self) -> None: ...

@final
class NativeRouterSessionPins:
    def __new__(
        cls, client: NativeRedisClient | None, clock: Callable[[], float] | None = None
    ) -> NativeRouterSessionPins: ...
    def attach(self, client: NativeRedisClient) -> bool: ...
    @property
    def has_redis(self) -> bool: ...
    def get(self, key: PinKey) -> Future[object]: ...
    def get_local(self, key: PinKey) -> object: ...
    def set_local(self, key: PinKey, value: object, ttl_seconds: float) -> None: ...
    def delete_local(self, key: PinKey) -> None: ...
    def set(self, key: PinKey, value: object, ttl_seconds: float) -> Future[None]: ...
    def get_shared(self, key: PinKey) -> Future[object]: ...
    def read(self, key: PinKey, ttl_seconds: float, send: NativeSend) -> NativePending[object]: ...
    def claim(
        self,
        key: PinKey,
        value: object,
        ttl_seconds: float,
        eligible_values: Sequence[object] | None,
        preserve_legacy_model_ids: bool,
    ) -> Future[object]: ...
    def claim_local(
        self,
        key: PinKey,
        value: object,
        ttl_seconds: float,
        eligible_values: Sequence[object] | None,
        preserve_legacy_model_ids: bool,
    ) -> object: ...

@final
class NativeRouterUsage:
    def __new__(cls, client: NativeRedisClient | None) -> NativeRouterUsage: ...
    def attach(self, client: NativeRedisClient) -> bool: ...
    @property
    def has_redis(self) -> bool: ...
    def increment(self, counter: _UsageCounter, value: float, ttl_seconds: int) -> Future[float | None]: ...
    def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> Future[None]: ...
    def get_many(self, counters: Sequence[_UsageCounter]) -> Future[list[float | None]]: ...
    def increment_sync(self, counter: _UsageCounter, value: float, ttl_seconds: int) -> float | None: ...
    def increment_deployment_sync(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None: ...
    def get_many_sync(self, counters: Sequence[_UsageCounter]) -> list[float | None]: ...
    def usage(self, counters: Sequence[_UsageCounter], send: NativeSend) -> NativeUsageRead: ...
    def increment_pipeline(self, increments: Sequence[_CounterIncrement]) -> Future[list[float] | None]: ...
    def increment_post_call(self, request: NativeRequestRedis, increments: Sequence[_CounterIncrement]) -> None: ...
    def increment_remote(self, increments: Sequence[_CounterIncrement]) -> Future[list[float] | None]: ...
    def increment_local(self, increment: _CounterIncrement) -> float: ...
    def get_local(self, counter: _UsageCounter) -> float | None: ...
    def get_local_many(self, counters: Sequence[_UsageCounter]) -> list[float | None]: ...
    def set_local(self, counter: _UsageCounter, value: float) -> None: ...
    def current(self, counter: _UsageCounter, limit: float) -> Future[float | None]: ...
    def current_sync(self, counter: _UsageCounter, limit: float) -> float | None: ...
    def shared_or_local(self, counter: _UsageCounter) -> Future[float | None]: ...
    def flush_local_counters(self) -> None: ...
    def snapshot(self) -> dict[str, float]: ...

class _RateLimitScope(Protocol):
    @property
    def descriptor_key(self) -> str: ...
    @property
    def descriptor_value(self) -> str: ...

class _RateLimitCounter(Protocol):
    @property
    def scope(self) -> _RateLimitScope: ...
    @property
    def kind(self) -> Literal["requests", "tokens"]: ...

class _RateLimitEntry(Protocol):
    @property
    def scope(self) -> _RateLimitScope: ...
    @property
    def suffix(self) -> str: ...

class _RateLimitIncrement(Protocol):
    @property
    def counter(self) -> _RateLimitCounter: ...
    @property
    def amount(self) -> int | float: ...
    @property
    def ttl(self) -> int | None: ...

class _RateLimitAtomicCounter(Protocol):
    @property
    def counter(self) -> _RateLimitCounter: ...
    @property
    def limit(self) -> int: ...
    @property
    def increment(self) -> int: ...
    @property
    def ttl(self) -> int: ...
    @property
    def window_size(self) -> int: ...

_RateLimitLocal = int | float | str | dict[str, float]

@final
class WindowCount:
    def __new__(cls, window_start: str, count: int) -> WindowCount: ...
    @property
    def window_start(self) -> str: ...
    @property
    def count(self) -> int: ...

@final
class AtomicApplied:
    def __new__(cls, counters: Sequence[tuple[int, int]]) -> AtomicApplied: ...
    @property
    def counters(self) -> list[tuple[int, int]]: ...

@final
class AtomicOverLimit:
    def __new__(cls, index: int, current: int, limit: int) -> AtomicOverLimit: ...
    @property
    def index(self) -> int: ...
    @property
    def current(self) -> int: ...
    @property
    def limit(self) -> int: ...

@final
class GuardedReply:
    def __new__(cls, applied: bool, count: int) -> GuardedReply: ...
    @property
    def applied(self) -> bool: ...
    @property
    def count(self) -> int: ...

@final
class GaugesAcquired:
    def __new__(cls, in_flight: Sequence[int]) -> GaugesAcquired: ...
    @property
    def in_flight(self) -> list[int]: ...

@final
class GaugeOverLimit:
    def __new__(cls, index: int, in_flight: int, limit: int) -> GaugeOverLimit: ...
    @property
    def index(self) -> int: ...
    @property
    def in_flight(self) -> int: ...
    @property
    def limit(self) -> int: ...

@final
class NativeRateLimits:
    def __new__(
        cls,
        client: NativeRedisClient | None,
        backfill_ttl_seconds: float | None = None,
    ) -> NativeRateLimits: ...
    @property
    def has_redis(self) -> bool: ...
    def slot_groups(self, scopes: Sequence[_RateLimitScope], force: bool) -> list[list[int]]: ...
    def get_local(self, entry: _RateLimitEntry) -> _RateLimitLocal | None: ...
    def set_local(self, entry: _RateLimitEntry, value: _RateLimitLocal, ttl_seconds: float | None) -> None: ...
    def get(
        self,
        entry: _RateLimitEntry,
        *,
        backfill_ttl_seconds: float | None = None,
        request: NativeRequestRedis | None = None,
    ) -> Future[_RateLimitLocal | None]: ...
    def sliding_window(
        self, counters: Sequence[_RateLimitCounter], now: int, window_size: int, send: NativeSend
    ) -> NativePending[list[WindowCount]] | None: ...
    def read_counters(
        self, counters: Sequence[_RateLimitCounter], send: NativeSend
    ) -> NativePending[list[str | None]] | None: ...
    def check_and_increment(
        self, counters: Sequence[_RateLimitAtomicCounter], send: NativeSend
    ) -> NativePending[AtomicApplied | AtomicOverLimit] | None: ...
    def guarded_increment(
        self,
        counter: _RateLimitCounter,
        expected_window_start: str,
        amount: float,
        ttl_seconds: int | None,
        send: NativeSend,
    ) -> NativePending[GuardedReply] | None: ...
    def parallel_acquire(
        self, gauges: Sequence[tuple[_RateLimitScope, int]], slot_ttl_seconds: int, slot_id: str, send: NativeSend
    ) -> NativePending[GaugesAcquired | GaugeOverLimit] | None: ...
    def parallel_release(
        self, gauges: Sequence[_RateLimitScope], slot_id: str, send: NativeSend
    ) -> NativePending[list[int]] | None: ...
    def parallel_count(
        self, gauges: Sequence[_RateLimitScope], slot_ttl_seconds: int, send: NativeSend
    ) -> NativePending[list[int]] | None: ...
    def token_increment(
        self, increments: Sequence[_RateLimitIncrement], send: NativeSend
    ) -> NativePending[list[float]] | None: ...
    def token_increment_post_call(
        self, request: NativeRequestRedis, increments: Sequence[_RateLimitIncrement]
    ) -> bool: ...
    def parallel_release_post_call(
        self, request: NativeRequestRedis, gauges: Sequence[_RateLimitScope], slot_id: str
    ) -> bool: ...
    def increment(self, increment: _RateLimitIncrement) -> Future[int | float]: ...
    def increment_many(self, increments: Sequence[_RateLimitIncrement]) -> Future[None]: ...
    def increment_many_post_call(
        self, request: NativeRequestRedis, increments: Sequence[_RateLimitIncrement]
    ) -> None: ...

@final
class LocalSizes:
    def __new__(cls, objects: int, key_objects: int) -> LocalSizes: ...
    @property
    def objects(self) -> int: ...
    @property
    def key_objects(self) -> int: ...

@final
class NativeAuthCache:
    def __new__(
        cls,
        client: NativeRedisClient | None,
        *,
        memory_entries: int,
        memory_ttl_seconds: float | None = None,
    ) -> NativeAuthCache: ...
    @property
    def has_redis(self) -> bool: ...
    def get(self, key: str, request: NativeRequestRedis | None = None) -> Future[object]: ...
    def reconfigure(self, *, memory_entries: int, memory_ttl_seconds: float | None = None) -> None: ...
    def get_sync(self, key: str) -> object: ...
    def set(self, key: str, value: object, ttl_seconds: float | None = None) -> Future[None]: ...
    def set_sync(self, key: str, value: object, ttl_seconds: float | None = None) -> None: ...
    def lease(self, key: str) -> Future[str | None]: ...
    def set_leased(
        self, key: str, lease: str | None, value: object, ttl_seconds: float | None = None
    ) -> Future[bool]: ...
    def declare_set(
        self, key: str, value: object, ttl_seconds: float | None, send: NativeSend
    ) -> NativePending[None]: ...
    def declare_delete(self, key: str, send: NativeSend) -> NativePending[None]: ...
    def set_many(self, entries: Sequence[tuple[str, object]], ttl_seconds: float | None = None) -> Future[None]: ...
    def write_back(self, entries: Sequence[tuple[str, object, float | None]], send: NativeSend) -> Future[None]: ...
    def delete(self, keys: Sequence[str]) -> Future[None]: ...
    def delete_sync(self, keys: Sequence[str]) -> None: ...
    def get_many(self, keys: Sequence[str], send: NativeSend) -> NativePending[list[object]] | None: ...
    def increment_if_present(self, key: str, amount: float) -> Future[float | None]: ...
    def get_local(self, key: str) -> object: ...
    def set_local(self, key: str, value: object, ttl_seconds: float | None = None) -> None: ...
    def delete_local(self, key: str) -> None: ...
    def flush_local(self) -> None: ...
    def len_local(self) -> LocalSizes: ...

@final
class SpendRead:
    def __new__(cls, value: float | None, from_redis: bool) -> SpendRead: ...
    @property
    def value(self) -> float | None: ...
    @property
    def from_redis(self) -> bool: ...

@final
class PostCallSettlement:
    @property
    def settled(self) -> bool: ...

@final
class NativeSpendCounters:
    def __new__(cls, client: NativeRedisClient | None) -> NativeSpendCounters: ...
    def attach(self, client: NativeRedisClient) -> bool: ...
    @property
    def has_redis(self) -> bool: ...
    @property
    def redis_ttl_seconds(self) -> float | None: ...
    def set_committed_ttl(self, ttl_seconds: float) -> None: ...
    def read(self, counter: str) -> Future[SpendRead]: ...
    def read_many(
        self, counters: Sequence[str], send: NativeSend
    ) -> NativePending[list[float | None] | None] | None: ...
    def increment(self, counter: str, amount: float) -> Future[float]: ...
    def increment_many(self, increments: Sequence[tuple[str, float]]) -> Future[list[float]]: ...
    def increment_many_post_call(
        self, request: NativeRequestRedis, increments: Sequence[tuple[str, float]]
    ) -> PostCallSettlement | None: ...
    def set_max(self, counter: str, spend: float) -> Future[float | None]: ...
    def seed(self, counter: str, spend: float, *, increment_if_lost: bool) -> Future[float]: ...
    def set(self, counter: str, spend: float, ttl_seconds: float | None = None) -> Future[None]: ...
    def delete(self, counters: Sequence[str]) -> Future[None]: ...
    def refresh_ttl(self, counter: str) -> Future[bool]: ...
    def get_local(self, counter: str) -> float | None: ...
    def set_local(self, counter: str, spend: float, ttl_seconds: float | None = None) -> None: ...
    def delete_local(self, counter: str) -> None: ...
    def increment_local(self, counter: str, amount: float) -> float: ...
    def flush_local(self) -> None: ...
    def snapshot(self) -> dict[str, float]: ...

@final
class NativeDatabase:
    @staticmethod
    def connect() -> Future[NativeDatabase]: ...
    def close(self) -> Future[None]: ...

@final
class NativeTokenRedactor:
    def __new__(cls, minimum_custom_key_length: int, word_ranges: Sequence[tuple[int, int]]) -> NativeTokenRedactor: ...
    def redact(self, text: str, replacement: str) -> str | None: ...

@final
class NativeDiagnosticProcessor:
    def __new__(cls, minimum_custom_key_length: int) -> NativeDiagnosticProcessor: ...
    def redact_text(self, text: str) -> str: ...
    def redact_structured_text(self, key: str | None, text: str) -> str: ...
    def redact_client_message(self, text: str) -> str: ...
    def process_diagnostic(
        self,
        message: str,
        exception: str | None,
        stack: str | None,
        leaves: Sequence[tuple[str | None, str]],
        policy: tuple[bool, int, int],
    ) -> tuple[str, str | None, str | None, list[str], bool]: ...
    def scrub_access_arguments(self, arguments: Sequence[str]) -> list[str]: ...

def ocr(
    call: NativeCall,
) -> Execution: ...
def aocr(
    call: NativeCall,
) -> Execution: ...
def ocr_health_check_document(model: str, custom_llm_provider: str | None) -> dict[str, object]: ...
def ocr_passthrough_response(model: str, endpoint: str, body: bytes) -> dict[str, object] | None: ...
def embedding(
    call: NativeCall,
) -> Execution: ...
def aembedding(
    call: NativeCall,
) -> Execution: ...
def transcription(
    call: NativeCall,
) -> dict[str, object]: ...
def atranscription(
    call: NativeCall,
) -> Future[dict[str, object]]: ...
def completion(
    call: NativeCall,
) -> Execution: ...
def acompletion(
    call: NativeCall,
) -> Execution: ...
def responses(
    call: NativeCall,
) -> Execution: ...
def aresponses(
    call: NativeCall,
) -> Execution: ...
def messages(
    call: NativeCall,
) -> Execution: ...
def amessages(
    call: NativeCall,
) -> Execution: ...

@final
class ResponsesWebSocketConnection:
    def __new__(cls, _uninstantiable: Never, /) -> Never: ...
    @classmethod
    def connect(
        cls,
        url: str,
        headers: Mapping[str, str] | None = None,
        timeout_seconds: float | None = None,
    ) -> Future[ResponsesWebSocketConnection]: ...
    def send_text(self, text: str) -> Future[None]: ...
    def recv_text(self) -> Future[str | None]: ...
    def close(self) -> Future[None]: ...

@final
class TokenCounter:
    @staticmethod
    def from_tokenizer(tokenizer: Tokenizer, fast: bool = False) -> TokenCounter: ...
    def acount_request(self, body: bytes) -> Future[dict[str, object]]: ...

@final
class Tokenizer:
    @staticmethod
    def from_tiktoken(encoding: str) -> Tokenizer: ...
    @staticmethod
    def from_json(tokenizer_json: str) -> Tokenizer: ...
    @staticmethod
    def from_pretrained(
        identifier: str,
        revision: str = "main",
        token: str | None = None,
    ) -> Tokenizer: ...
    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: Sequence[int], skip_special_tokens: bool = True) -> str: ...
    def count(self, text: str, fast: bool = False) -> int: ...
    # tiktoken encodings
    def encode_special(self, text: str, allowed: Sequence[str]) -> list[int]: ...
    def encode_with_unstable(self, text: str, allowed: Sequence[str]) -> tuple[list[int], list[list[int]]]: ...
    def encode_single_token(self, piece: bytes) -> int: ...
    def special_tokens(self) -> dict[str, int]: ...
    def max_token_value(self) -> int: ...
    def is_special_token(self, token: int) -> bool: ...
    def token_byte_values(self) -> list[bytes]: ...
    def decode_bytes(self, ids: Sequence[int]) -> bytes: ...
    # Hugging Face tokenizers
    def to_json(self, pretty: bool = False) -> str: ...
    def token_to_id(self, token: str) -> int | None: ...
    def id_to_token(self, id: int) -> str | None: ...
    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]: ...
    def get_vocab_size(self, with_added_tokens: bool = True) -> int: ...
    def added_tokens_decoder(self) -> list[tuple[int, tuple[str, bool, bool, bool, bool, bool]]]: ...
    def padding(self) -> dict[str, object] | None: ...
    def truncation(self) -> dict[str, object] | None: ...
    def num_special_tokens_to_add(self, is_pair: bool) -> int: ...
    def encode_special_tokens(self) -> bool: ...
    def encode_huggingface(
        self,
        sequence: str | Sequence[str],
        pair: str | Sequence[str] | None = None,
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        fast: bool = False,
    ) -> HuggingFaceEncoding: ...
    def encode_batch_huggingface(
        self,
        inputs: Sequence[tuple[str | Sequence[str], str | Sequence[str] | None]],
        is_pretokenized: bool = False,
        add_special_tokens: bool = True,
        fast: bool = False,
    ) -> list[HuggingFaceEncoding]: ...
    @property
    def name(self) -> str: ...

@final
class HuggingFaceEncoding:
    def __new__(cls, json: str | None = None) -> HuggingFaceEncoding: ...
    @staticmethod
    def merge(encodings: Sequence[HuggingFaceEncoding], growing_offsets: bool = True) -> HuggingFaceEncoding: ...
    def __len__(self) -> int: ...
    def __reduce__(self) -> tuple[type[HuggingFaceEncoding], tuple[str]]: ...
    def word_to_tokens(self, word_index: int, sequence_index: int = 0) -> tuple[int, int] | None: ...
    def word_to_chars(self, word_index: int, sequence_index: int = 0) -> tuple[int, int] | None: ...
    def token_to_sequence(self, token_index: int) -> int | None: ...
    def token_to_chars(self, token_index: int) -> tuple[int, int] | None: ...
    def token_to_word(self, token_index: int) -> int | None: ...
    def char_to_token(self, char_pos: int, sequence_index: int = 0) -> int | None: ...
    def char_to_word(self, char_pos: int, sequence_index: int = 0) -> int | None: ...
    def set_sequence_id(self, sequence_id: int) -> None: ...
    def pad(
        self,
        length: int,
        direction: str = "right",
        pad_id: int = 0,
        pad_type_id: int = 0,
        pad_token: str = "[PAD]",
    ) -> None: ...
    def truncate(self, max_length: int, stride: int = 0, direction: str = "right") -> None: ...
    @property
    def ids(self) -> list[int]: ...
    @property
    def tokens(self) -> list[str]: ...
    @property
    def offsets(self) -> list[tuple[int, int]]: ...
    @property
    def type_ids(self) -> list[int]: ...
    @property
    def attention_mask(self) -> list[int]: ...
    @property
    def special_tokens_mask(self) -> list[int]: ...
    @property
    def word_ids(self) -> list[int | None]: ...
    @property
    def sequence_ids(self) -> list[int | None]: ...
    @property
    def overflowing(self) -> list[HuggingFaceEncoding]: ...
    @property
    def n_sequences(self) -> int: ...

def gil_stats() -> dict[str, int]: ...
def process_state_started() -> bool: ...
def reserve_process_for_forking() -> None: ...

__all__ = [
    "AtomicApplied",
    "AtomicOverLimit",
    "ForkedAfterNativeRuntimeStarted",
    "GaugeOverLimit",
    "GaugesAcquired",
    "GuardedReply",
    "HuggingFaceEncoding",
    "LocalSizes",
    "NativeAdmission",
    "NativeAuthCache",
    "NativeClickHouseSpendConfig",
    "NativeClickHouseSpendStorage",
    "NativeCostMapSource",
    "NativeDatabase",
    "NativeDeploymentHealth",
    "NativeDiagnosticProcessor",
    "NativeFetchFailed",
    "NativeFetchPolicy",
    "NativeFetched",
    "NativeInstalled",
    "NativeModelCostMap",
    "NativeModelCostMapState",
    "NativePending",
    "NativeProvenance",
    "NativeRateLimits",
    "NativeRedisBreaker",
    "NativeRedisClient",
    "NativeRedisStoreError",
    "NativeRegistered",
    "NativeRejected",
    "NativeRequestRedis",
    "NativeRouterCooldowns",
    "NativeRouterSessionPins",
    "NativeRouterUsage",
    "NativeSend",
    "NativeSensitiveRoutePins",
    "NativeSpendCounters",
    "NativeTableSwapped",
    "NativeTokenRedactor",
    "NativeUsageRead",
    "PostCallSettlement",
    "ProcessReservedForForking",
    "ResponsesWebSocketConnection",
    "RouterCall",
    "RouterCoreHandle",
    "RustBridgeDeclined",
    "RustUpstreamError",
    "SpendRead",
    "TokenCounter",
    "Tokenizer",
    "WindowCount",
    "acompletion",
    "aembedding",
    "amessages",
    "aocr",
    "aresponses",
    "atranscription",
    "bundled_model_cost_map",
    "completion",
    "embedding",
    "gil_stats",
    "messages",
    "native_redis_breaker_states",
    "ocr",
    "ocr_health_check_document",
    "ocr_passthrough_response",
    "process_state_started",
    "reserve_process_for_forking",
    "responses",
    "trace_encode_error",
    "transcription",
    "unsupported_tls_settings",
]

@final
class _SecretManagerRuntime:
    @staticmethod
    def from_config(
        system: str,
        environment: Mapping[str, str],
        settings: Mapping[str, object] | None = None,
        enterprise_enabled: bool = False,
    ) -> _SecretManagerRuntime: ...
    @staticmethod
    def from_client(client: object) -> _SecretManagerRuntime | None: ...
    @property
    def system(self) -> str: ...
    def read_secret(self, name: str, settings: Mapping[str, object] | None = None) -> JsonValue: ...
    def read_secret_async(self, name: str, settings: Mapping[str, object] | None = None) -> Future[JsonValue]: ...
    def async_write_secret(
        self,
        secret_name: str,
        secret_value: str,
        description: str | None = None,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
        tags: object = None,
    ) -> Future[dict[str, JsonValue]]: ...
    def async_delete_secret(
        self,
        secret_name: str,
        recovery_window_in_days: int | None = None,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Future[dict[str, JsonValue]]: ...
    def async_rotate_secret(
        self,
        current_secret_name: str,
        new_secret_name: str,
        new_secret_value: str,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
    ) -> Future[dict[str, JsonValue]]: ...
    def sync_read_secret(
        self,
        secret_name: str,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
        primary_secret_name: str | None = None,
    ) -> JsonValue: ...
    def async_read_secret(
        self,
        secret_name: str,
        optional_params: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None = None,
        primary_secret_name: str | None = None,
    ) -> Future[JsonValue]: ...

# The FailureKind names an attempt outcome's `kind` takes (router-core `FailureKind`).
_FailureKind = Literal[
    "RateLimitError",
    "ContextWindowExceededError",
    "ContentPolicyViolationError",
    "AuthenticationError",
    "PermissionDeniedError",
    "NotFoundError",
    "BadRequestError",
    "Timeout",
    "APIConnectionError",
    "InternalServerError",
    "ServiceUnavailableError",
    "BadGatewayError",
    "GuardrailIntervention",
    "Declined",
    "CallerCredentialAuthenticationError",
    "CallerCredentialRateLimitError",
    "APIError",
]

# One event a call reported since the last step (router-core `RouterEvent`), carried on the next step.
class _Event(TypedDict):
    kind: str

# Every step `RouterCall.next` returns carries the events reported since the last one.
class _Step(TypedDict):
    events: list[_Event]

class _AttemptStep(_Step):
    kind: Literal["attempt"]
    deployment_id: str
    # The deployment as a `model_list` entry, with its id in `model_info`.
    deployment: dict[str, JsonValue]
    # The name this attempt serves: the requested model or the fallback target.
    model: str
    model_group: str
    target_order: int | None
    candidate_ids: list[str]
    retry: int
    fallback_depth: int
    fallback_level: int
    timeout: float | None
    # Request params the fallback walk's dict targets merged in.
    params: dict[str, JsonValue] | None
    # The walk level whose request metadata the attempt carries.
    scope: int
    max_retries: int
    model_group_size: int

class _SleepStep(_Step):
    kind: Literal["sleep"]
    seconds: float

class _AllowFallbackStep(_Step):
    kind: Literal["allow_fallback"]
    target: str

class _RefuseFallbacksStep(_Step):
    kind: Literal["refuse_fallbacks"]
    # The walk level whose chain the dict targets are.
    level: int
    targets: list[dict[str, JsonValue]]

class _SelectStep(_Step):
    kind: Literal["select"]
    stage: Literal[
        "plugins",
        "strategy",
        "tags",
        "pre_call",
        "text_only",
        "callback_filters",
        "routing_plugins",
        "provider_request",
    ]
    strategy: str | None
    model_group: str
    # The name the request resolved from, which strategies key their state by.
    model: str
    # Whose selector a strategy reads: "default", "group" (with `group`) or "override".
    scope: str
    group: str | None
    # The deployments the name resolved to, before health filters.
    pool: list[str]
    candidates: list[str]
    # `routing_plugins` only: whether the plugins run.
    run: NotRequired[bool]
    # `callback_filters` only: the ids the request already failed on.
    excluded_ids: NotRequired[list[str]]
    skipped_ids: NotRequired[list[str]]

# Pre-routing before a pick resolves `model`: answered with the model group to resolve instead
# (a string), a custom strategy's deployment document (a mapping), or None.
class _PreRouteStep(_Step):
    kind: Literal["pre_route"]
    model: str
    fallback_depth: int
    retry: int

# One deployment's per-minute usage counters: `model` is the model its usage is counted under,
# `minute` is `HH-MM` in UTC.
class _UsageWindow(TypedDict):
    deployment_id: str
    model: str
    minute: str

# One RouterCache op, answered from the router's stores. It names the state it touches, never a
# Redis key. Fields by op: cooldowns (`ids`) -> one stored cooldown or None per id; put_cooldown
# (`id`, `cooldown`, `ttl`) -> None; increment_allowed_fails (`id`, `suffix`, `ttl`) -> the new
# count; add_usage (`window`, `tokens`, `requests`) -> None; add_request (`window`) -> None;
# usage (`window`) -> [tokens, requests], each a number or None; health_states -> each checked
# deployment's `{is_healthy: bool, timestamp: float}` by id, or None; outcomes (`id`) -> this
# worker's [successes, fails] of the current minute; count_success (`id`) and count_failure (`id`)
# -> None. `ttl` is in seconds.
class _CacheStep(_Step):
    kind: Literal["cache"]
    op: Literal[
        "cooldowns",
        "put_cooldown",
        "increment_allowed_fails",
        "add_usage",
        "add_request",
        "usage",
        "health_states",
        "outcomes",
        "count_success",
        "count_failure",
    ]
    ids: NotRequired[list[str]]
    id: NotRequired[str]
    cooldown: NotRequired[JsonValue]
    suffix: NotRequired[str | None]
    ttl: NotRequired[float]
    window: NotRequired[_UsageWindow]
    tokens: NotRequired[int | float]
    requests: NotRequired[int | float]

# One lookup of a model-info query (router-core `ModelInfoLookup`): its `lookup` name and that
# lookup's arguments, answered as the lookup documents.
class _ModelInfoStep(_Step):
    kind: Literal["model_info"]
    lookup: Literal[
        "model_info",
        "cost_entry",
        "provider",
        "pattern_models",
        "bedrock_base",
        "discovered",
        "user_model_info",
        "log",
        "group_params",
        "group_deployment",
        "group_capabilities",
        "group_seed",
        "reasoning_intersect",
    ]
    model: NotRequired[JsonValue]
    tolerate: NotRequired[bool]
    exact: NotRequired[bool]
    key: NotRequired[str]
    process: NotRequired[bool]
    name: NotRequired[str]
    model_id: NotRequired[str]
    id: NotRequired[JsonValue]
    level: NotRequired[Literal["debug", "error"]]
    message: NotRequired[str]
    index: NotRequired[int]
    model_info: NotRequired[JsonValue]
    litellm_model: NotRequired[JsonValue]
    llm_provider: NotRequired[JsonValue]
    mapped: NotRequired[bool]
    info: NotRequired[dict[str, JsonValue]]
    current: NotRequired[JsonValue]
    resolved: NotRequired[JsonValue]

class _FailedStep(_Step):
    kind: Literal["failed"]
    error: str
    error_kind: Literal[
        "config",
        "unknown_model",
        "no_healthy_deployments",
        "attempt",
        "cache",
        "retired",
        "paused",
        "not_pass_through",
        "no_pass_through_deployments",
        "only_markers",
        "ptu_reserved",
        "context_window_exceeded",
        "unsupported_params",
        "routing_plugins_left_nothing",
        "routing_plugins_need_async",
        "multiple_teams",
        "reserved",
        "invalid_router_weights",
        "non_numeric_weight",
        "missing_routing_group",
        "default_deployment_raises",
        "index_out_of_range",
        "model_info",
    ]
    attempt_failed: bool
    status: int | None
    failure_kind: _FailureKind | None
    error_model: NotRequired[str]
    error_detail: NotRequired[str]
    error_params: NotRequired[list[str]]
    # `model_info` only: the builtin exception the lookup raised.
    error_exception: NotRequired[str]

# The end of a `RouterCoreHandle.start` (and the stream recovery calls).
class _DoneStep(_Step):
    kind: Literal["done"]
    deployment_id: str
    model_group: str
    attempted_retries: int
    max_retries: int
    attempted_fallbacks: int
    total_tokens: int

# The end of a `RouterCoreHandle.select`.
class _SelectedStep(_Step):
    kind: Literal["done"]
    deployment_id: str
    model_group: str
    deployment: dict[str, JsonValue]

# The end of a `RouterCoreHandle.healthy`.
class _HealthyStep(_Step):
    kind: Literal["done"]
    # The name was a deployment id.
    single: bool
    deployment_ids: list[str]
    deployments: list[dict[str, JsonValue]]

# The end of a `RouterCoreHandle.report_failure`.
class _ReportedStep(_Step):
    kind: Literal["done"]
    cooled: bool
    # The cooldown applied, in seconds; None when the deployment was not cooled down.
    cooldown_time: float | None

# The end of a `RouterCoreHandle.report_success`.
class _CountedStep(_Step):
    kind: Literal["done"]
    key: str

# The end of a `RouterCoreHandle.count_success`.
class _SuccessCountedStep(_Step):
    kind: Literal["done"]
    counted: bool

# One `os.environ/` reference a `secrets` step asks for: a top-level param (`key`), or an
# env-only `dataSources[source].parameters[key]` (`env_only`).
class _SecretRef(TypedDict):
    name: str
    key: str
    env_only: bool
    source: NotRequired[int]

# The deployment build's `os.environ/` references: answered with one value per ref, in order.
class _SecretsStep(_Step):
    kind: Literal["secrets"]
    index: int
    refs: list[_SecretRef]

# The deployment build's steps for one deployment. `validate` and `added` are answered with
# `{"entry": <model_list entry>}`, `{"skip": True}` or `{"invalid": True}`; `provider` with
# `{"provider": str | None}` or `{"invalid": True}`.
class _BuildStep(_Step):
    kind: Literal["validate", "provider", "added"]
    path: Literal["create", "add"]
    index: int
    entry: dict[str, JsonValue]
    declared_id: str | None
    duplicate_id: bool
    provider: str | None

class _BuiltDeployment(TypedDict):
    index: int
    id: str | None
    provider: str | None

# The end of a `RouterCoreHandle.build`.
class _BuiltStep(_Step):
    kind: Literal["done"]
    built: list[_BuiltDeployment]

# The end of a model-info query: `router_model_info` sets `info` (the merged map or None) and
# `user_keys`; `deployment_model_info` sets `info`; the others set `value`.
class _ModelInfoDoneStep(_Step):
    kind: Literal["done"]
    info: NotRequired[dict[str, JsonValue] | None]
    user_keys: NotRequired[list[str]]
    value: NotRequired[JsonValue]

# `RouterCoreHandle.resolve`: what a name resolves to.
class _Resolution(TypedDict):
    model_group: str
    kind: str
    ids: list[str]

# The steps each kind of `RouterCall` returns, by the handle method that starts it.
_CallSteps = (
    _AttemptStep
    | _SleepStep
    | _AllowFallbackStep
    | _RefuseFallbacksStep
    | _SelectStep
    | _PreRouteStep
    | _CacheStep
    | _DoneStep
    | _FailedStep
)
_SelectSteps = _SelectStep | _PreRouteStep | _CacheStep | _SelectedStep | _FailedStep
_HealthySteps = _SelectStep | _PreRouteStep | _CacheStep | _HealthyStep | _FailedStep
_BuildSteps = _SecretsStep | _BuildStep | _BuiltStep | _FailedStep
_ModelInfoSteps = _ModelInfoStep | _ModelInfoDoneStep | _FailedStep
_FailureReportSteps = _CacheStep | _ReportedStep | _FailedStep
_SuccessReportSteps = _CacheStep | _CountedStep | _SuccessCountedStep | _FailedStep

class _AttemptSucceeded(TypedDict):
    ok: Literal[True]
    total_tokens: NotRequired[int]
    blocked: NotRequired[Literal["content_filter", "refusal"] | None]
    # A stream whose success or failure the host reports when it ends (`report_success`,
    # `report_failure`): the attempt counts its request and `total_tokens` only.
    deferred: NotRequired[bool]
    # Whether the attempt counts against its deployment's usage; unset, its own limits decide.
    tracks_usage: NotRequired[bool | None]

class _AttemptFailed(TypedDict):
    ok: Literal[False]
    kind: _FailureKind
    status: NotRequired[int | None]
    retry_after: NotRequired[float | None]
    num_retries: NotRequired[int | None]
    # The deployment the exception names, which a retry skips over the attempt's own.
    failed_deployment_id: NotRequired[str | None]
    detail: NotRequired[str | None]
    agentic_followup: NotRequired[bool]
    # The exception has no `status_code`, which no cooldown rule reads.
    status_absent: NotRequired[bool]
    # The failure counts against no per-minute requests.
    uncounted_request: NotRequired[bool]

# Each document is a mapping or its JSON text.
_Document = Mapping[str, object] | str

@final
class RouterCoreHandle:
    def __new__(cls, config: _Document, dynamic: _Document) -> RouterCoreHandle: ...
    def start(self, op: str, request: _Document) -> RouterCall[_CallSteps]: ...
    # Driven like `start`; its `done` step is a `_SelectedStep`.
    def select(self, model: str, request: _Document) -> RouterCall[_SelectSteps]: ...
    # Driven like `select`; its `done` step is a `_HealthyStep`.
    def healthy(self, model: str, request: _Document) -> RouterCall[_HealthySteps]: ...
    # `{"replace" | "append" | "upsert": [deployments], "delete": [ids], "settings": {dynamic document}, "index": {...}}`.
    def apply(self, changes: _Document) -> int: ...
    # Build model_list entries through the host, driven like `start`: `_SecretsStep` and
    # `_BuildStep` steps, then a `_BuiltStep`. `path`: "create" (`set_model_list`) or "add";
    # `install`: "replace", "append" or "upsert".
    def build(
        self, entries: Sequence[object], path: str, resolve_secrets: bool, install: str
    ) -> RouterCall[_BuildSteps]: ...
    # Keep the providers `other`'s builds resolved for the same deployments.
    def adopt_providers(self, other: RouterCoreHandle) -> None: ...
    # `get_router_model_info`, driven like `start`: `model_info` lookup steps (a `lookup` field
    # and its arguments), then a done step with `info` (the merged map or None) and `user_keys`.
    def router_model_info(self, query: _Document) -> RouterCall[_ModelInfoSteps]: ...
    # `get_deployment_model_info`, driven like `router_model_info`; its done step has `info`.
    def deployment_model_info(self, model_id: str, model_name: object) -> RouterCall[_ModelInfoSteps]: ...
    # `_set_model_group_info` for a `GroupInfoQuery` document, driven like `router_model_info`; its
    # done step has `value`: None, or `{"assignments": {field: value}}`.
    def model_group_info(self, query: _Document) -> RouterCall[_ModelInfoSteps]: ...
    # `get_configured_token_limits`, driven like `router_model_info`; `value` is `[input, output]`.
    def configured_token_limits(self, id: object, configured: _Document) -> RouterCall[_ModelInfoSteps]: ...
    # `get_model_listing_info`, driven like `router_model_info`; `value` has `cost_map_keys`,
    # `max_input_tokens` and `max_output_tokens`.
    def model_listing_info(self, deployments: list[object]) -> RouterCall[_ModelInfoSteps]: ...
    # A configured string that is not blank (`get_configured_mode`, `get_configured_display_name`).
    @staticmethod
    def configured_text(value: object) -> str | None: ...
    # Each model_info's `service_tiers`, unvalidated.
    @staticmethod
    def configured_service_tiers(model_infos: list[object]) -> list[object]: ...
    # The routing-cache key a deployment counts `metric` (tpm, rpm, itpm, otpm) under in `minute`.
    def registry_version(self) -> int: ...
    # The version the deployments before the newest appends last changed at.
    def registry_rewritten(self) -> int: ...
    # A failure the router did not see itself, driven like `start`: cache steps, then a
    # `_ReportedStep`.
    def report_failure(
        self,
        id: str,
        failure: Mapping[str, object],
        deployment_cooldown: bool = False,
        count_only: bool = False,
        requested_group: str | None = None,
        cooldown_time: float | None = None,
        uncounted: bool = False,
        counts_request: bool = False,
        usage_model: str | None = None,
        callbacks: Mapping[str, bool] | None = None,
    ) -> RouterCall[_FailureReportSteps]: ...
    # A success a logging callback saw, driven like `report_failure`; `done` carries `key`.
    def report_success(
        self,
        id: str,
        total_tokens: float,
        requests: float,
        usage_model: str | None = None,
        tracks_usage: bool | None = None,
        callbacks: Mapping[str, bool] | None = None,
    ) -> RouterCall[_SuccessReportSteps]: ...
    # Count a success against any id, driven like `report_failure`; `done` carries `counted`.
    def count_success(
        self, id: str, callbacks: Mapping[str, bool] | None = None
    ) -> RouterCall[_SuccessReportSteps]: ...
    # Every deployment as a `model_list` entry, in config order.
    def deployments(self) -> list[dict[str, JsonValue]]: ...
    def deployments_from(self, start: int) -> list[dict[str, JsonValue]]: ...
    def deployment(self, id: str) -> dict[str, JsonValue] | None: ...
    # Lookups over the registry indexes, answering what each index holds as written.
    # Registry lookups (router-core `lookup.rs`): rows the host reads its stored entries at
    # (`{"at": position, "name"?, "group"?, "member"?}`, `{"pattern": entry, "name"}`, or a
    # `{"raw": value}` / `{"raw_positions": value}` its own read raises at).
    def lookup_all(
        self, model_name: str, team_id: str | None = None, rename: str | None = None
    ) -> list[dict[str, JsonValue]]: ...
    def lookup_across_teams(self, model: str) -> tuple[list[dict[str, JsonValue]], int]: ...
    def lookup_config(self) -> list[dict[str, JsonValue]]: ...
    def lookup_model_ids(
        self, model_name: str | None = None, exclude_team_models: bool = False
    ) -> tuple[list[str], dict[str, JsonValue] | None]: ...
    def lookup_model_list(
        self, model_name: str | None, team_id: str | None, scope: Mapping[str, object]
    ) -> list[dict[str, JsonValue]]: ...
    def lookup_model_names(
        self, team_id: str | None, scope: Mapping[str, object]
    ) -> tuple[list[str], dict[str, JsonValue] | None]: ...
    def lookup_access_groups(
        self, model_name: str | None, access_group: str | None, team_id: str | None, scope: Mapping[str, object]
    ) -> tuple[list[tuple[str, list[str]]], dict[str, JsonValue] | None]: ...
    def lookup_name_serving(
        self, model_id: str, scope: Mapping[str, object]
    ) -> tuple[bool, str | None, dict[str, JsonValue] | None]: ...
    def has_model_name(self, name: str) -> bool: ...
    def lookup_team_model_has_alternatives(self, id: str) -> tuple[bool, dict[str, JsonValue] | None]: ...
    def lookup_group_usable_by_team(
        self, model_group: str, team_id: str | None
    ) -> tuple[int | None, dict[str, JsonValue] | None]: ...
    def lookup_team_public_name(
        self, model_id: str, team_id: str
    ) -> tuple[int | None, dict[str, JsonValue] | None]: ...
    def lookup_routable(self, model_group: str, team_id: str | None) -> list[dict[str, JsonValue]]: ...
    def lookup_wildcard_route(
        self, model: str, team_id: str | None, requests: list[str]
    ) -> list[dict[str, JsonValue]]: ...
    def lookup_candidate_ids(
        self, resolved: str, team_id: str | None, members: list[str] | None, requests: list[str]
    ) -> tuple[list[str], dict[str, JsonValue] | None]: ...
    def has_model_id(self, id: str) -> bool: ...
    def id_position(self, id: str) -> JsonValue | None: ...
    def id_positions(self) -> list[tuple[str, JsonValue]]: ...
    def named_positions(self, name: str) -> JsonValue | None: ...
    def team_positions(self, team: str, public: str) -> JsonValue | None: ...
    def team_public_names_of(self, team: str) -> list[str]: ...
    def litellm_model_positions(self, model: str) -> list[int]: ...
    # The registry indexes the host hands out (`HostIndexes`), as JSON.
    def host_indexes(self) -> dict[str, JsonValue]: ...
    def provider_default_ids(self) -> list[str]: ...
    # `PatternMatchRouter.route` over the wildcard router, or `team`'s.
    def pattern_route(
        self, team: str | None, request: str, filtered: list[str] | None = None
    ) -> list[dict[str, JsonValue]] | None: ...
    # `get_models_by_pattern` over the wildcard router, or `team`'s.
    def models_by_pattern(self, team: str | None, requests: list[str]) -> list[str] | None: ...
    # The provider the host resolved when it built deployment `id` (its `provider` build step).
    def provider(self, id: str) -> str | None: ...
    def resolve(self, name: str, specific_deployment: bool = False) -> _Resolution | None: ...
    def routable(self, model: str, request: _Document) -> list[dict[str, JsonValue]]: ...
    # `_get_routing_context` for a pick of `model`: (strategy, scope, group).
    def routing_context(self, model: str, routing_strategy: str | None = None) -> tuple[str, str, str | None]: ...
    def retire(self) -> None: ...
    # A stream that failed before content: what may still take over it, and the retry budget for
    # a failure of `kind`: {"can_retry", "can_fall_back", "refusal_fallback": bool,
    # "retry_budget": int | None, "policy_applies": bool}.
    def stream_policy(
        self, request: _Document, origin: _Document, kind: str | None = None
    ) -> dict[str, bool | int | None]: ...
    # The same-group retry and the fallback walk for such a stream, driven like `start`.
    def retry_stream(
        self, op: str, request: _Document, origin: _Document, failure: Mapping[str, object]
    ) -> RouterCall[_CallSteps]: ...
    def fallback_stream(
        self, op: str, request: _Document, origin: _Document, failure: Mapping[str, object]
    ) -> RouterCall[_CallSteps]: ...
    def retry_policy_retries(self, failure: Mapping[str, object], model_group: str | None = None) -> int | None: ...
    def allowed_fails_for(self, failure: Mapping[str, object]) -> int | None: ...
    def should_retry(
        self, failure: Mapping[str, object], counts: tuple[int, int], fallbacks: tuple[bool, bool, bool]
    ) -> bool: ...

_S = TypeVar("_S", covariant=True)

@final
class RouterCall(Generic[_S]):
    # The answer to the last step, which Rust validates against it: attempt: _AttemptSucceeded |
    # _AttemptFailed; sleep: None; allow_fallback: bool; refuse_fallbacks: bool; select: the remaining
    # candidate ids, pick first; pre_route: see _PreRouteStep; cache: see _CacheStep; secrets, build and model_info:
    # see those steps; first call: None.
    def next(self, outcome: object) -> _S: ...
