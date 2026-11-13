from asyncio import Future
from collections.abc import Callable, Mapping, Sequence
from typing import ClassVar, Final, Generic, Literal, Never, NotRequired, Protocol, TypedDict, TypeVar, final

import httpx
from pydantic import JsonValue
from typing_extensions import disjoint_base

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.team import LiteLLM_TeamTable, LiteLLM_TeamTableCachedObj
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_OrganizationTable, LiteLLM_ProjectTableCachedObj
from litellm.proxy.auth.host.facts import EndUserRow
from litellm.redis.store_settings import RedisBreakerSettings, RedisStoreSettings
from litellm.redis.stores.router_session_pins.keys import PinKey
from litellm.rust_bridge.execution import Execution
from litellm.rust_bridge.public_call import NativeCall
from litellm.types.proxy.agent_identity import ManagedAgentContext
from litellm.types.proxy.auth.user_api_key_auth import UserAPIKeyAuth

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
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]  # PyO3 `eq` without `hash` sets it to None
    def __new__(cls, window_start: str, count: int) -> WindowCount: ...
    @property
    def window_start(self) -> str: ...
    @property
    def count(self) -> int: ...

@final
class AtomicApplied:
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]  # PyO3 `eq` without `hash` sets it to None
    def __new__(cls, counters: Sequence[tuple[int, int]]) -> AtomicApplied: ...
    @property
    def counters(self) -> list[tuple[int, int]]: ...

@final
class AtomicOverLimit:
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]  # PyO3 `eq` without `hash` sets it to None
    def __new__(cls, index: int, current: int, limit: int) -> AtomicOverLimit: ...
    @property
    def index(self) -> int: ...
    @property
    def current(self) -> int: ...
    @property
    def limit(self) -> int: ...

@final
class GuardedReply:
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]  # PyO3 `eq` without `hash` sets it to None
    def __new__(cls, applied: bool, count: int) -> GuardedReply: ...
    @property
    def applied(self) -> bool: ...
    @property
    def count(self) -> int: ...

@final
class GaugesAcquired:
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]  # PyO3 `eq` without `hash` sets it to None
    def __new__(cls, in_flight: Sequence[int]) -> GaugesAcquired: ...
    @property
    def in_flight(self) -> list[int]: ...

@final
class GaugeOverLimit:
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]  # PyO3 `eq` without `hash` sets it to None
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
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]  # PyO3 `eq` without `hash` sets it to None
    def __new__(cls, objects: int, key_objects: int) -> LocalSizes: ...
    @property
    def objects(self) -> int: ...
    @property
    def key_objects(self) -> int: ...

class _AuthRequest(Protocol):
    @property
    def route(self) -> str: ...
    @property
    def method(self) -> str | None: ...
    @property
    def token(self) -> str | None: ...
    @property
    def model(self) -> str | None: ...
    @property
    def end_user_id(self) -> str | None: ...
    @property
    def team_header(self) -> str | None: ...
    @property
    def is_llm_api_route(self) -> bool: ...
    @property
    def auth_enforced_pass_through(self) -> bool: ...
    @property
    def forwarded_routes(self) -> Sequence[str]: ...
    @property
    def is_info_route(self) -> bool: ...
    @property
    def pass_through_open(self) -> bool: ...
    @property
    def from_custom_auth(self) -> bool: ...
    @property
    def now(self) -> float: ...

class _BudgetExceeded(Protocol):
    @property
    def current_cost(self) -> float: ...
    @property
    def max_budget(self) -> float: ...
    @property
    def message(self) -> str: ...
    @property
    def entity_type(self) -> str | None: ...
    @property
    def entity_id(self) -> str | None: ...

class _KeyFacts(Protocol):
    @property
    def token(self) -> str | None: ...
    @property
    def blocked(self) -> bool: ...
    @property
    def expires(self) -> float | None: ...
    @property
    def is_proxy_admin(self) -> bool: ...
    @property
    def user_id(self) -> str | None: ...
    @property
    def team_id(self) -> str | None: ...
    @property
    def team_object_permission_id(self) -> str | None: ...
    @property
    def project_id(self) -> str | None: ...
    @property
    def end_user_id(self) -> str | None: ...
    @property
    def last_refreshed_at(self) -> float | None: ...

class _TeamFacts(Protocol):
    @property
    def team_id(self) -> str: ...
    @property
    def models(self) -> Sequence[str] | None: ...
    @property
    def allowed_passthrough_routes(self) -> Sequence[str] | None: ...
    @property
    def denied_passthrough_routes(self) -> Sequence[str]: ...
    @property
    def members(self) -> Sequence[str]: ...

class _UserFacts(Protocol):
    @property
    def user_id(self) -> str: ...
    @property
    def user_email(self) -> str | None: ...
    @property
    def user_role(self) -> str | None: ...
    @property
    def teams(self) -> Sequence[str]: ...
    @property
    def deactivated(self) -> bool: ...

class _ManagedAgent(Protocol):
    @property
    def mode(self) -> str: ...
    @property
    def agent_id(self) -> str: ...
    @property
    def user_id(self) -> str | None: ...

class _AgentFailure(Protocol):
    @property
    def code(self) -> str: ...
    @property
    def message(self) -> str: ...

class _RegisteredAgent(Protocol):
    @property
    def agent_id(self) -> str: ...
    @property
    def identity_managed(self) -> bool: ...
    @property
    def has_identity(self) -> bool: ...
    @property
    def has_legacy_identity(self) -> bool: ...

class _HostFacts(Protocol):
    @property
    def master_key(self) -> str | None: ...
    @property
    def premium(self) -> bool: ...
    @property
    def has_db(self) -> bool: ...
    @property
    def ui_login(self) -> bool: ...
    @property
    def proxy_budget(self) -> bool: ...
    @property
    def metrics_require_auth(self) -> bool: ...

@final
class RustAuthSettings:
    def __new__(
        cls, general_settings: Mapping[str, object], litellm_jwtauth: LiteLLM_JWTAuth | None, facts: _HostFacts
    ) -> RustAuthSettings: ...
    @property
    def jwt_leeway_seconds(self) -> int: ...

@final
class RustPlan:
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]  # PyO3 `eq` without `hash` sets it to None
    Public: ClassVar[RustPlan]
    OAuth2Introspection: ClassVar[RustPlan]
    OAuth2Proxy: ClassVar[RustPlan]
    JwtWithOperatorCode: ClassVar[RustPlan]
    Jwt: ClassVar[RustPlan]
    PassThroughOpen: ClassVar[RustPlan]
    Key: ClassVar[RustPlan]

@final
class RustJwtMapping:
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]  # PyO3 `eq` without `hash` sets it to None
    Key: ClassVar[RustJwtMapping]
    Pending: ClassVar[RustJwtMapping]
    Unmapped: ClassVar[RustJwtMapping]

@final
class RustClaimIdentity:
    @property
    def user_id(self) -> str | None: ...
    @property
    def agent_id(self) -> str | None: ...
    @property
    def user(self) -> LiteLLM_UserTable | None: ...

@final
class RustMappedClaims:
    @property
    def user_id(self) -> str | None: ...
    @property
    def claim_user_id(self) -> str | None: ...
    @property
    def user_email(self) -> str | None: ...
    @property
    def valid_user_email(self) -> bool | None: ...
    @property
    def team_id(self) -> str | None: ...
    @property
    def team_ids(self) -> list[str]: ...
    @property
    def all_team_ids(self) -> list[str]: ...
    @property
    def team_alias(self) -> str | None: ...
    @property
    def org_id(self) -> str | None: ...
    @property
    def org_alias(self) -> str | None: ...
    @property
    def end_user_id(self) -> str | None: ...
    @property
    def object_id(self) -> str | None: ...
    @property
    def agent_claim(self) -> str | None: ...
    @property
    def litellm_role(self) -> str | None: ...

@disjoint_base
class RustIdentity:
    @final
    class NoMasterKey(RustIdentity):
        __match_args__: tuple[Literal["token"]] = ("token",)
        def __new__(cls, token: str | None) -> RustIdentity.NoMasterKey: ...
        @property
        def token(self) -> str | None: ...

    @final
    class UserAuthRoute(RustIdentity):
        __match_args__: tuple[()] = ()

    @final
    class CachedAdmin(RustIdentity):
        __match_args__: tuple[Literal["key"], Literal["end_user"]] = ("key", "end_user")
        def __new__(cls, key: UserAPIKeyAuth, end_user: EndUserRow) -> RustIdentity.CachedAdmin: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def end_user(self) -> EndUserRow: ...

    @final
    class MasterKey(RustIdentity):
        __match_args__: tuple[Literal["hash"], Literal["end_user"]] = ("hash", "end_user")
        def __new__(cls, hash: str, end_user: EndUserRow) -> RustIdentity.MasterKey: ...
        @property
        def hash(self) -> str: ...
        @property
        def end_user(self) -> EndUserRow: ...

    @final
    class VirtualKey(RustIdentity):
        __match_args__: tuple[Literal["context"]] = ("context",)
        def __new__(cls, context: RustKeyContext) -> RustIdentity.VirtualKey: ...
        @property
        def context(self) -> RustKeyContext: ...

    @final
    class MappedJwtKey(RustIdentity):
        __match_args__: tuple[
            Literal["claims"], Literal["end_user_id"], Literal["user_email"], Literal["context"]
        ] = ("claims", "end_user_id", "user_email", "context")
        def __new__(
            cls,
            claims: dict[str, object],
            end_user_id: str | None,
            user_email: str | None,
            context: RustKeyContext,
        ) -> RustIdentity.MappedJwtKey: ...
        @property
        def claims(self) -> dict[str, object]: ...
        @property
        def end_user_id(self) -> str | None: ...
        @property
        def user_email(self) -> str | None: ...
        @property
        def context(self) -> RustKeyContext: ...

    @final
    class AutoRegisteredJwtKey(RustIdentity):
        """`org_id` is the JWT's and is set only for a key created for it."""

        __match_args__: tuple[
            Literal["claims"],
            Literal["end_user_id"],
            Literal["user_email"],
            Literal["org_id"],
            Literal["user"],
            Literal["context"],
        ] = ("claims", "end_user_id", "user_email", "org_id", "user", "context")
        def __new__(
            cls,
            claims: dict[str, object],
            end_user_id: str | None,
            user_email: str | None,
            org_id: str | None,
            user: LiteLLM_UserTable | None,
            context: RustKeyContext,
        ) -> RustIdentity.AutoRegisteredJwtKey: ...
        @property
        def claims(self) -> dict[str, object]: ...
        @property
        def end_user_id(self) -> str | None: ...
        @property
        def user_email(self) -> str | None: ...
        @property
        def org_id(self) -> str | None: ...
        @property
        def user(self) -> LiteLLM_UserTable | None: ...
        @property
        def context(self) -> RustKeyContext: ...

    @final
    class FinishedJwt(RustIdentity):
        __match_args__: tuple[Literal["token"]] = ("token",)
        def __new__(cls, token: UserAPIKeyAuth) -> RustIdentity.FinishedJwt: ...
        @property
        def token(self) -> UserAPIKeyAuth: ...

    @final
    class Jwt(RustIdentity):
        __match_args__: tuple[
            Literal["claims"],
            Literal["is_proxy_admin"],
            Literal["user_id"],
            Literal["user_email"],
            Literal["team_id"],
            Literal["org_id"],
            Literal["end_user_id"],
            Literal["agent_id"],
            Literal["managed_agent"],
            Literal["user"],
            Literal["team"],
            Literal["membership"],
            Literal["org"],
            Literal["end_user"],
        ] = (
            "claims",
            "is_proxy_admin",
            "user_id",
            "user_email",
            "team_id",
            "org_id",
            "end_user_id",
            "agent_id",
            "managed_agent",
            "user",
            "team",
            "membership",
            "org",
            "end_user",
        )
        def __new__(
            cls,
            claims: dict[str, object],
            is_proxy_admin: bool,
            user_id: str | None,
            user_email: str | None,
            team_id: str | None,
            org_id: str | None,
            end_user_id: str | None,
            agent_id: str | None,
            managed_agent: ManagedAgentContext | None,
            user: LiteLLM_UserTable | None,
            team: LiteLLM_TeamTable | None,
            membership: LiteLLM_TeamMembership | None,
            org: LiteLLM_OrganizationTable | None,
            end_user: EndUserRow | None,
        ) -> RustIdentity.Jwt: ...
        @property
        def claims(self) -> dict[str, object]: ...
        @property
        def is_proxy_admin(self) -> bool: ...
        @property
        def user_id(self) -> str | None: ...
        @property
        def user_email(self) -> str | None: ...
        @property
        def team_id(self) -> str | None: ...
        @property
        def org_id(self) -> str | None: ...
        @property
        def end_user_id(self) -> str | None: ...
        @property
        def agent_id(self) -> str | None: ...
        @property
        def managed_agent(self) -> ManagedAgentContext | None: ...
        @property
        def user(self) -> LiteLLM_UserTable | None: ...
        @property
        def team(self) -> LiteLLM_TeamTable | None: ...
        @property
        def membership(self) -> LiteLLM_TeamMembership | None: ...
        @property
        def org(self) -> LiteLLM_OrganizationTable | None: ...
        @property
        def end_user(self) -> EndUserRow | None: ...

_AuthenticatedIdentity = (
    RustIdentity.NoMasterKey
    | RustIdentity.UserAuthRoute
    | RustIdentity.CachedAdmin
    | RustIdentity.MasterKey
    | RustIdentity.VirtualKey
    | RustIdentity.MappedJwtKey
    | RustIdentity.AutoRegisteredJwtKey
    | RustIdentity.FinishedJwt
    | RustIdentity.Jwt
)

@disjoint_base
class RustTeamGrant:
    @final
    class NoTeam(RustTeamGrant):
        __match_args__: tuple[()] = ()

    @final
    class Loaded(RustTeamGrant):
        __match_args__: tuple[Literal["team"]] = ("team",)
        def __new__(cls, team: LiteLLM_TeamTableCachedObj) -> RustTeamGrant.Loaded: ...
        @property
        def team(self) -> LiteLLM_TeamTableCachedObj: ...

    @final
    class KeyCopy(RustTeamGrant):
        __match_args__: tuple[Literal["object_permission"]] = ("object_permission",)
        def __new__(cls, object_permission: LiteLLM_ObjectPermissionTable | None) -> RustTeamGrant.KeyCopy: ...
        @property
        def object_permission(self) -> LiteLLM_ObjectPermissionTable | None: ...

@final
class RustKeyContext:
    def __new__(
        cls,
        key: UserAPIKeyAuth,
        end_user: EndUserRow,
        *,
        owner: LiteLLM_UserTable | None = None,
        team: RustTeamGrant,
        project: LiteLLM_ProjectTableCachedObj | None = None,
    ) -> RustKeyContext: ...
    @property
    def key(self) -> UserAPIKeyAuth: ...
    @property
    def end_user(self) -> EndUserRow: ...
    @property
    def owner(self) -> LiteLLM_UserTable | None: ...
    @property
    def team(self) -> RustTeamGrant: ...
    @property
    def project(self) -> LiteLLM_ProjectTableCachedObj | None: ...

@disjoint_base
class RustDenial:
    @final
    class Proxy(RustDenial):
        __match_args__: tuple[Literal["status"], Literal["message"], Literal["error_type"], Literal["param"]] = (
            "status",
            "message",
            "error_type",
            "param",
        )
        def __new__(cls, status: int, message: str, error_type: str, param: str | None) -> RustDenial.Proxy: ...
        @property
        def status(self) -> int: ...
        @property
        def message(self) -> str: ...
        @property
        def error_type(self) -> str: ...
        @property
        def param(self) -> str | None: ...

    @final
    class Http(RustDenial):
        __match_args__: tuple[Literal["status"], Literal["detail"]] = ("status", "detail")
        def __new__(cls, status: int, detail: str) -> RustDenial.Http: ...
        @property
        def status(self) -> int: ...
        @property
        def detail(self) -> str: ...

    @final
    class InvalidVirtualKey(RustDenial):
        __match_args__: tuple[Literal["status"], Literal["detail"]] = ("status", "detail")
        def __new__(cls, status: int, detail: str) -> RustDenial.InvalidVirtualKey: ...
        @property
        def status(self) -> int: ...
        @property
        def detail(self) -> str: ...

    @final
    class ModelAccess(RustDenial):
        __match_args__: tuple[Literal["status"], Literal["detail"], Literal["internal_message"]] = (
            "status",
            "detail",
            "internal_message",
        )
        def __new__(cls, status: int, detail: str, internal_message: str) -> RustDenial.ModelAccess: ...
        @property
        def status(self) -> int: ...
        @property
        def detail(self) -> str: ...
        @property
        def internal_message(self) -> str: ...

    @final
    class Plain(RustDenial):
        __match_args__: tuple[Literal["message"]] = ("message",)
        def __new__(cls, message: str) -> RustDenial.Plain: ...
        @property
        def message(self) -> str: ...

    @final
    class BudgetExceeded(RustDenial):
        __match_args__: tuple[
            Literal["current_cost"],
            Literal["max_budget"],
            Literal["message"],
            Literal["entity_type"],
            Literal["entity_id"],
        ] = ("current_cost", "max_budget", "message", "entity_type", "entity_id")
        def __new__(
            cls,
            current_cost: float,
            max_budget: float,
            message: str,
            entity_type: str | None,
            entity_id: str | None,
        ) -> RustDenial.BudgetExceeded: ...
        @property
        def current_cost(self) -> float: ...
        @property
        def max_budget(self) -> float: ...
        @property
        def message(self) -> str: ...
        @property
        def entity_type(self) -> str | None: ...
        @property
        def entity_id(self) -> str | None: ...

    @final
    class TeamNotFound(RustDenial):
        __match_args__: tuple[Literal["team_id"]] = ("team_id",)
        def __new__(cls, team_id: str) -> RustDenial.TeamNotFound: ...
        @property
        def team_id(self) -> str: ...

@final
class RustAuthOutcome:
    @property
    def denied(self) -> bool: ...
    @property
    def identity(self) -> _AuthenticatedIdentity | None: ...
    @property
    def denial(self) -> RustDenial | None: ...
    @property
    def claims(self) -> dict[str, object] | None: ...
    @property
    def assertion_rejected(self) -> bool: ...
    @property
    def claim_identity(self) -> RustClaimIdentity | None: ...
    @property
    def mapping(self) -> RustJwtMapping | None: ...
    @property
    def mapped_key(self) -> UserAPIKeyAuth | None: ...
    @property
    def denied_key(self) -> UserAPIKeyAuth | None: ...

@disjoint_base
class RustAuthStep:
    @final
    class ValidatedEndUserId(RustAuthStep):
        __match_args__: tuple[Literal["end_user_id"]] = ("end_user_id",)
        def __new__(cls, end_user_id: str) -> RustAuthStep.ValidatedEndUserId: ...
        @property
        def end_user_id(self) -> str: ...

    @final
    class PrefetchIdentity(RustAuthStep):
        __match_args__: tuple[Literal["token"], Literal["end_user_id"]] = ("token", "end_user_id")
        def __new__(cls, token: str | None, end_user_id: str | None) -> RustAuthStep.PrefetchIdentity: ...
        @property
        def token(self) -> str | None: ...
        @property
        def end_user_id(self) -> str | None: ...

    @final
    class EndUser(RustAuthStep):
        __match_args__: tuple[Literal["end_user_id"]] = ("end_user_id",)
        def __new__(cls, end_user_id: str | None) -> RustAuthStep.EndUser: ...
        @property
        def end_user_id(self) -> str | None: ...

    @final
    class CachedKey(RustAuthStep):
        __match_args__: tuple[Literal["hash"]] = ("hash",)
        def __new__(cls, hash: str) -> RustAuthStep.CachedKey: ...
        @property
        def hash(self) -> str: ...

    @final
    class UiKey(RustAuthStep):
        __match_args__: tuple[Literal["token"]] = ("token",)
        def __new__(cls, token: str) -> RustAuthStep.UiKey: ...
        @property
        def token(self) -> str: ...

    @final
    class EvictKey(RustAuthStep):
        __match_args__: tuple[Literal["hash"]] = ("hash",)
        def __new__(cls, hash: str) -> RustAuthStep.EvictKey: ...
        @property
        def hash(self) -> str: ...

    @final
    class CachedTeam(RustAuthStep):
        __match_args__: tuple[Literal["team_id"]] = ("team_id",)
        def __new__(cls, team_id: str) -> RustAuthStep.CachedTeam: ...
        @property
        def team_id(self) -> str: ...

    @final
    class WithTeam(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["team"]] = ("key", "team")
        def __new__(cls, key: UserAPIKeyAuth, team: LiteLLM_TeamTableCachedObj) -> RustAuthStep.WithTeam: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def team(self) -> LiteLLM_TeamTableCachedObj: ...

    @final
    class DbKey(RustAuthStep):
        __match_args__: tuple[Literal["hash"], Literal["end_user"]] = ("hash", "end_user")
        def __new__(cls, hash: str, end_user: EndUserRow) -> RustAuthStep.DbKey: ...
        @property
        def hash(self) -> str: ...
        @property
        def end_user(self) -> EndUserRow: ...

    @final
    class WithTempBudget(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        def __new__(cls, key: UserAPIKeyAuth) -> RustAuthStep.WithTempBudget: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...

    @final
    class NormalizeModel(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        def __new__(cls, key: UserAPIKeyAuth) -> RustAuthStep.NormalizeModel: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...

    @final
    class PrefetchAuthObjects(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["end_user"]] = ("key", "end_user")
        def __new__(cls, key: UserAPIKeyAuth, end_user: EndUserRow) -> RustAuthStep.PrefetchAuthObjects: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def end_user(self) -> EndUserRow: ...

    @final
    class KeyOwner(RustAuthStep):
        __match_args__: tuple[Literal["user_id"]] = ("user_id",)
        def __new__(cls, user_id: str) -> RustAuthStep.KeyOwner: ...
        @property
        def user_id(self) -> str: ...

    @final
    class KeySkipsBudgetChecks(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        def __new__(cls, key: UserAPIKeyAuth) -> RustAuthStep.KeySkipsBudgetChecks: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...

    @final
    class TeamMemberBudget(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["owner"]] = ("key", "owner")
        def __new__(cls, key: UserAPIKeyAuth, owner: LiteLLM_UserTable | None) -> RustAuthStep.TeamMemberBudget: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def owner(self) -> LiteLLM_UserTable | None: ...

    @final
    class KeyBudgets(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["owner"], Literal["is_llm_api_route"]] = (
            "key",
            "owner",
            "is_llm_api_route",
        )
        def __new__(
            cls,
            key: UserAPIKeyAuth,
            owner: LiteLLM_UserTable | None,
            is_llm_api_route: bool,
        ) -> RustAuthStep.KeyBudgets: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def owner(self) -> LiteLLM_UserTable | None: ...
        @property
        def is_llm_api_route(self) -> bool: ...

    @final
    class KeyTeam(RustAuthStep):
        __match_args__: tuple[Literal["team_id"]] = ("team_id",)
        def __new__(cls, team_id: str) -> RustAuthStep.KeyTeam: ...
        @property
        def team_id(self) -> str: ...

    @final
    class ObjectPermission(RustAuthStep):
        __match_args__: tuple[Literal["id"]] = ("id",)
        def __new__(cls, id: str) -> RustAuthStep.ObjectPermission: ...
        @property
        def id(self) -> str: ...

    @final
    class Project(RustAuthStep):
        __match_args__: tuple[Literal["project_id"]] = ("project_id",)
        def __new__(cls, project_id: str) -> RustAuthStep.Project: ...
        @property
        def project_id(self) -> str: ...

    @final
    class ProxyBudgetAlert(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        def __new__(cls, key: UserAPIKeyAuth) -> RustAuthStep.ProxyBudgetAlert: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...

    @final
    class FetchJson(RustAuthStep):
        __match_args__: tuple[Literal["url"]] = ("url",)
        def __new__(cls, url: str) -> RustAuthStep.FetchJson: ...
        @property
        def url(self) -> str: ...

    @final
    class FetchUserinfo(RustAuthStep):
        __match_args__: tuple[Literal["endpoint"], Literal["token"]] = ("endpoint", "token")
        def __new__(cls, endpoint: str, token: str) -> RustAuthStep.FetchUserinfo: ...
        @property
        def endpoint(self) -> str: ...
        @property
        def token(self) -> str: ...

    @final
    class CachedJson(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        def __new__(cls, key: str) -> RustAuthStep.CachedJson: ...
        @property
        def key(self) -> str: ...

    @final
    class CacheJson(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["value"], Literal["ttl_seconds"]] = (
            "key",
            "value",
            "ttl_seconds",
        )
        def __new__(cls, key: str, value: object, ttl_seconds: float) -> RustAuthStep.CacheJson: ...
        @property
        def key(self) -> str: ...
        @property
        def value(self) -> object: ...
        @property
        def ttl_seconds(self) -> float: ...

    @final
    class ForgetJson(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        def __new__(cls, key: str) -> RustAuthStep.ForgetJson: ...
        @property
        def key(self) -> str: ...

    @final
    class Team(RustAuthStep):
        __match_args__: tuple[Literal["team_id"], Literal["upsert"]] = ("team_id", "upsert")
        def __new__(cls, team_id: str, upsert: bool) -> RustAuthStep.Team: ...
        @property
        def team_id(self) -> str: ...
        @property
        def upsert(self) -> bool: ...

    @final
    class TeamByAlias(RustAuthStep):
        __match_args__: tuple[Literal["alias"]] = ("alias",)
        def __new__(cls, alias: str) -> RustAuthStep.TeamByAlias: ...
        @property
        def alias(self) -> str: ...

    @final
    class GrantingTeam(RustAuthStep):
        __match_args__: tuple[Literal["team_id"]] = ("team_id",)
        def __new__(cls, team_id: str) -> RustAuthStep.GrantingTeam: ...
        @property
        def team_id(self) -> str: ...

    @final
    class LoadTeamMembership(RustAuthStep):
        __match_args__: tuple[Literal["user_id"], Literal["team_id"]] = ("user_id", "team_id")
        def __new__(cls, user_id: str, team_id: str) -> RustAuthStep.LoadTeamMembership: ...
        @property
        def user_id(self) -> str: ...
        @property
        def team_id(self) -> str: ...

    @final
    class TeamModelAccess(RustAuthStep):
        __match_args__: tuple[Literal["team"], Literal["model"]] = ("team", "model")
        def __new__(cls, team: LiteLLM_TeamTable, model: str) -> RustAuthStep.TeamModelAccess: ...
        @property
        def team(self) -> LiteLLM_TeamTable: ...
        @property
        def model(self) -> str: ...

    @final
    class User(RustAuthStep):
        __match_args__: tuple[Literal["user_id"], Literal["user_email"], Literal["sso_user_id"], Literal["upsert"]] = (
            "user_id",
            "user_email",
            "sso_user_id",
            "upsert",
        )
        def __new__(
            cls,
            user_id: str,
            user_email: str | None,
            sso_user_id: str | None,
            upsert: bool,
        ) -> RustAuthStep.User: ...
        @property
        def user_id(self) -> str: ...
        @property
        def user_email(self) -> str | None: ...
        @property
        def sso_user_id(self) -> str | None: ...
        @property
        def upsert(self) -> bool: ...

    @final
    class ResolveUserIdentity(RustAuthStep):
        __match_args__: tuple[
            Literal["user_id"], Literal["user_email"], Literal["sso_user_id"], Literal["upsert"], Literal["team_id"]
        ] = ("user_id", "user_email", "sso_user_id", "upsert", "team_id")
        def __new__(
            cls,
            user_id: str,
            user_email: str | None,
            sso_user_id: str | None,
            upsert: bool,
            team_id: str | None,
        ) -> RustAuthStep.ResolveUserIdentity: ...
        @property
        def user_id(self) -> str: ...
        @property
        def user_email(self) -> str | None: ...
        @property
        def sso_user_id(self) -> str | None: ...
        @property
        def upsert(self) -> bool: ...
        @property
        def team_id(self) -> str | None: ...

    @final
    class SetUserRole(RustAuthStep):
        __match_args__: tuple[Literal["user"], Literal["role"]] = ("user", "role")
        def __new__(cls, user: LiteLLM_UserTable, role: str) -> RustAuthStep.SetUserRole: ...
        @property
        def user(self) -> LiteLLM_UserTable: ...
        @property
        def role(self) -> str: ...

    @final
    class PatchTeamMembership(RustAuthStep):
        __match_args__: tuple[Literal["user"], Literal["add"], Literal["remove"]] = ("user", "add", "remove")
        def __new__(
            cls,
            user: LiteLLM_UserTable,
            add: list[str],
            remove: list[str],
        ) -> RustAuthStep.PatchTeamMembership: ...
        @property
        def user(self) -> LiteLLM_UserTable: ...
        @property
        def add(self) -> list[str]: ...
        @property
        def remove(self) -> list[str]: ...

    @final
    class AddTeamMember(RustAuthStep):
        __match_args__: tuple[Literal["user_id"], Literal["team_id"]] = ("user_id", "team_id")
        def __new__(cls, user_id: str, team_id: str) -> RustAuthStep.AddTeamMember: ...
        @property
        def user_id(self) -> str: ...
        @property
        def team_id(self) -> str: ...

    @final
    class Org(RustAuthStep):
        __match_args__: tuple[Literal["org_id"]] = ("org_id",)
        def __new__(cls, org_id: str) -> RustAuthStep.Org: ...
        @property
        def org_id(self) -> str: ...

    @final
    class OrgByAlias(RustAuthStep):
        __match_args__: tuple[Literal["alias"]] = ("alias",)
        def __new__(cls, alias: str) -> RustAuthStep.OrgByAlias: ...
        @property
        def alias(self) -> str: ...

    @final
    class JwtEndUser(RustAuthStep):
        __match_args__: tuple[Literal["end_user_id"]] = ("end_user_id",)
        def __new__(cls, end_user_id: str) -> RustAuthStep.JwtEndUser: ...
        @property
        def end_user_id(self) -> str: ...

    @final
    class FinishJwt(RustAuthStep):
        __match_args__: tuple[Literal["identity"]] = ("identity",)
        def __new__(cls, identity: RustIdentity.Jwt) -> RustAuthStep.FinishJwt: ...
        @property
        def identity(self) -> RustIdentity.Jwt: ...

    @final
    class FinishAutoRegisteredJwt(RustAuthStep):
        __match_args__: tuple[
            Literal["claims"],
            Literal["end_user_id"],
            Literal["user_email"],
            Literal["org_id"],
            Literal["user"],
            Literal["key"],
        ] = ("claims", "end_user_id", "user_email", "org_id", "user", "key")
        def __new__(
            cls,
            claims: dict[str, object],
            end_user_id: str | None,
            user_email: str | None,
            org_id: str | None,
            user: LiteLLM_UserTable | None,
            key: UserAPIKeyAuth,
        ) -> RustAuthStep.FinishAutoRegisteredJwt: ...
        @property
        def claims(self) -> dict[str, object]: ...
        @property
        def end_user_id(self) -> str | None: ...
        @property
        def user_email(self) -> str | None: ...
        @property
        def org_id(self) -> str | None: ...
        @property
        def user(self) -> LiteLLM_UserTable | None: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...

    @final
    class KeyMappingHash(RustAuthStep):
        __match_args__: tuple[Literal["claim_field"], Literal["claim_value"], Literal["issuer"]] = (
            "claim_field",
            "claim_value",
            "issuer",
        )
        def __new__(cls, claim_field: str, claim_value: str, issuer: str | None) -> RustAuthStep.KeyMappingHash: ...
        @property
        def claim_field(self) -> str: ...
        @property
        def claim_value(self) -> str: ...
        @property
        def issuer(self) -> str | None: ...

    @final
    class KeyByHash(RustAuthStep):
        __match_args__: tuple[Literal["hash"]] = ("hash",)
        def __new__(cls, hash: str) -> RustAuthStep.KeyByHash: ...
        @property
        def hash(self) -> str: ...

    @final
    class ReusableKeyHash(RustAuthStep):
        __match_args__: tuple[Literal["user_id"], Literal["team_id"]] = ("user_id", "team_id")
        def __new__(cls, user_id: str, team_id: str | None) -> RustAuthStep.ReusableKeyHash: ...
        @property
        def user_id(self) -> str: ...
        @property
        def team_id(self) -> str | None: ...

    @final
    class CreateKeyMapping(RustAuthStep):
        __match_args__: tuple[
            Literal["claim_field"],
            Literal["claim_value"],
            Literal["issuer"],
            Literal["behavior"],
            Literal["existing_hash"],
            Literal["team_id"],
            Literal["user_id"],
            Literal["org_id"],
            Literal["agent_id"],
        ] = (
            "claim_field",
            "claim_value",
            "issuer",
            "behavior",
            "existing_hash",
            "team_id",
            "user_id",
            "org_id",
            "agent_id",
        )
        def __new__(
            cls,
            claim_field: str,
            claim_value: str,
            issuer: str | None,
            behavior: str,
            existing_hash: str | None,
            team_id: str | None,
            user_id: str | None,
            org_id: str | None,
            agent_id: str | None,
        ) -> RustAuthStep.CreateKeyMapping: ...
        @property
        def claim_field(self) -> str: ...
        @property
        def claim_value(self) -> str: ...
        @property
        def issuer(self) -> str | None: ...
        @property
        def behavior(self) -> str: ...
        @property
        def existing_hash(self) -> str | None: ...
        @property
        def team_id(self) -> str | None: ...
        @property
        def user_id(self) -> str | None: ...
        @property
        def org_id(self) -> str | None: ...
        @property
        def agent_id(self) -> str | None: ...

    @final
    class ManagedAgent(RustAuthStep):
        __match_args__: tuple[Literal["claims"]] = ("claims",)
        def __new__(cls, claims: dict[str, object]) -> RustAuthStep.ManagedAgent: ...
        @property
        def claims(self) -> dict[str, object]: ...

    @final
    class RecordAgentAuthentication(RustAuthStep):
        __match_args__: tuple[Literal["agent"]] = ("agent",)
        def __new__(cls, agent: ManagedAgentContext) -> RustAuthStep.RecordAgentAuthentication: ...
        @property
        def agent(self) -> ManagedAgentContext: ...

    @final
    class RegisteredAgent(RustAuthStep):
        __match_args__: tuple[Literal["claim"]] = ("claim",)
        def __new__(cls, claim: str) -> RustAuthStep.RegisteredAgent: ...
        @property
        def claim(self) -> str: ...

    @final
    class DelegatedAgentTeam(RustAuthStep):
        __match_args__: tuple[
            Literal["user_id"],
            Literal["agent_id"],
            Literal["team_id"],
            Literal["explicit_team"],
            Literal["allowed_team_ids"],
        ] = ("user_id", "agent_id", "team_id", "explicit_team", "allowed_team_ids")
        def __new__(
            cls,
            user_id: str | None,
            agent_id: str,
            team_id: str | None,
            explicit_team: bool,
            allowed_team_ids: list[str] | None,
        ) -> RustAuthStep.DelegatedAgentTeam: ...
        @property
        def user_id(self) -> str | None: ...
        @property
        def agent_id(self) -> str: ...
        @property
        def team_id(self) -> str | None: ...
        @property
        def explicit_team(self) -> bool: ...
        @property
        def allowed_team_ids(self) -> list[str] | None: ...

    @final
    class CustomValidate(RustAuthStep):
        __match_args__: tuple[Literal["claims"]] = ("claims",)
        def __new__(cls, claims: dict[str, object]) -> RustAuthStep.CustomValidate: ...
        @property
        def claims(self) -> dict[str, object]: ...

    @final
    class Done(RustAuthStep):
        __match_args__: tuple[Literal["outcome"]] = ("outcome",)
        def __new__(cls, outcome: RustAuthOutcome) -> RustAuthStep.Done: ...
        @property
        def outcome(self) -> RustAuthOutcome: ...

@final
class RustAuth:
    @staticmethod
    def plan(request: _AuthRequest, settings: RustAuthSettings) -> RustPlan | RustDenial: ...
    @staticmethod
    def authenticate_key(request: _AuthRequest, key: str | None, settings: RustAuthSettings) -> RustAuth: ...
    @staticmethod
    def oauth2_introspection_applies(request: _AuthRequest, settings: RustAuthSettings) -> bool: ...
    @staticmethod
    def authenticate_jwt(request: _AuthRequest, settings: RustAuthSettings) -> RustAuth: ...
    @staticmethod
    def authorize_jwt(request: _AuthRequest, settings: RustAuthSettings, *, provisioning: bool) -> RustAuth: ...
    @staticmethod
    def resolve_jwt_identity(request: _AuthRequest, settings: RustAuthSettings) -> RustAuth: ...
    @staticmethod
    def resolve_jwt_mapping(request: _AuthRequest, settings: RustAuthSettings) -> RustAuth: ...
    @staticmethod
    def verify_jwt(token: str, now: float, settings: RustAuthSettings) -> RustAuth: ...
    @staticmethod
    def verify_jwt_assertion(
        token: str, now: float, settings: RustAuthSettings, *, audience: list[str], required_claims: list[str]
    ) -> RustAuth: ...
    def resume(self) -> RustAuthStep: ...
    def fail(self) -> None: ...
    def done(self) -> None: ...
    def budget_exceeded(self, error: _BudgetExceeded) -> None: ...
    def found(self, found: bool) -> None: ...
    def row(self, row: UserAPIKeyAuth | EndUserRow | LiteLLM_UserTable) -> None: ...
    def found_row(
        self,
        row: LiteLLM_TeamTableCachedObj
        | LiteLLM_ObjectPermissionTable
        | LiteLLM_ProjectTableCachedObj
        | LiteLLM_TeamMembership
        | None = None,
    ) -> None: ...
    def key(self, facts: _KeyFacts | None = None, row: UserAPIKeyAuth | None = None) -> None: ...
    def key_owner(self, row: LiteLLM_UserTable | None = None, *, deactivated: bool = False) -> None: ...
    def cached_team(
        self, row: LiteLLM_TeamTableCachedObj | None = None, *, refreshed_at: float | None = None
    ) -> None: ...
    def json(self, value: object = None) -> None: ...
    def team(self, read: _TeamFacts | str, row: LiteLLM_TeamTable | None = None) -> None: ...
    def team_alias(self, read: _TeamFacts | str, row: LiteLLM_TeamTable | None = None) -> None: ...
    def model_access(self, *, allowed: bool) -> None: ...
    def user(
        self,
        user: _UserFacts | None = None,
        row: LiteLLM_UserTable | None = None,
        *,
        membership: LiteLLM_TeamMembership | None = None,
    ) -> None: ...
    def org(self, org_id: str | None = None, row: LiteLLM_OrganizationTable | None = None) -> None: ...
    def text(self, value: str | None = None) -> None: ...
    def mapped_key(
        self, facts: _KeyFacts | None = None, row: UserAPIKeyAuth | None = None, *, user_email: str | None = None
    ) -> None: ...
    def key_mapping(self, *, token_hash: str | None, conflict: bool) -> None: ...
    def managed_agent(self, agent: _ManagedAgent | None = None) -> None: ...
    def agent_failure(self, failure: _AgentFailure | None = None) -> None: ...
    def registered_agent(self, agent: _RegisteredAgent | None = None) -> None: ...

JWT_ID_JAG_TYP: Final[str]

def jwt_is_jwt(token: str | None) -> bool: ...
def jwt_unverified_claims(token: str) -> dict[str, object] | None: ...
def jwt_presents_claims(token: str, claims: dict[str, object]) -> bool: ...
def jwt_claims(claims: dict[str, object], litellm_jwtauth: LiteLLM_JWTAuth) -> RustMappedClaims: ...
def jwt_is_allowed_domain(user_email: str, litellm_jwtauth: LiteLLM_JWTAuth) -> bool: ...
def jwt_key_mapping_cache_key(claim_field: str, claim_value: str, issuer: str | None = None) -> str: ...
def auth_route_matches_wildcard(route: str, pattern: str) -> bool: ...
def auth_managed_route(route: str) -> Literal["discovery", "realtime", "live", "model", "tool", "other"]: ...
def auth_managed_agent_route_allowed(route: str, method: str | None = None) -> bool: ...
def auth_route_matches_prefix(route: str, allowed_route: str) -> bool: ...
def auth_route_matches_denied(route: str, denied_route: str) -> bool: ...
def auth_first_denied_route(forwarded_routes: Sequence[str], denied_routes: Sequence[str]) -> str | None: ...
def auth_jwt_team_routes_grant(route: str, team_allowed_routes: Sequence[str]) -> bool: ...
def auth_allowed_routes_grant(route: str, *, key: Sequence[str] | None, team: Sequence[str] | None) -> bool: ...
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
    __hash__: ClassVar[None]  # type: ignore[assignment]  # pyright: ignore[reportIncompatibleMethodOverride]  # PyO3 `eq` without `hash` sets it to None
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
    "JWT_ID_JAG_TYP",
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
    "NativeUsageRead",
    "PostCallSettlement",
    "ProcessReservedForForking",
    "ResponsesWebSocketConnection",
    "RouterCall",
    "RouterCoreHandle",
    "RustAuth",
    "RustAuthOutcome",
    "RustAuthSettings",
    "RustAuthStep",
    "RustBridgeDeclined",
    "RustClaimIdentity",
    "RustDenial",
    "RustIdentity",
    "RustJwtMapping",
    "RustKeyContext",
    "RustMappedClaims",
    "RustPlan",
    "RustTeamGrant",
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
    "auth_allowed_routes_grant",
    "auth_first_denied_route",
    "auth_jwt_team_routes_grant",
    "auth_managed_agent_route_allowed",
    "auth_managed_route",
    "auth_route_matches_denied",
    "auth_route_matches_prefix",
    "auth_route_matches_wildcard",
    "bundled_model_cost_map",
    "completion",
    "embedding",
    "gil_stats",
    "jwt_claims",
    "jwt_is_allowed_domain",
    "jwt_is_jwt",
    "jwt_key_mapping_cache_key",
    "jwt_presents_claims",
    "jwt_unverified_claims",
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
