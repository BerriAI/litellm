from asyncio import Future
from collections.abc import Mapping, Sequence
from enum import Enum
from typing import ClassVar, Generic, Literal, Never, NotRequired, Protocol, TypedDict, TypeVar, final

import httpx
from pydantic import JsonValue
from typing_extensions import disjoint_base

from litellm.models.object_permission import LiteLLM_ObjectPermissionTable
from litellm.models.team import LiteLLM_TeamTable, LiteLLM_TeamTableCachedObj
from litellm.models.team_membership import LiteLLM_TeamMembership
from litellm.models.user import LiteLLM_UserTable
from litellm.proxy._types import LiteLLM_JWTAuth, LiteLLM_OrganizationTable, LiteLLM_ProjectTableCachedObj
from litellm.proxy.auth.host.facts import EndUserRow
from litellm.proxy.auth.host.steps.authorization import BudgetFacts
from litellm.proxy.auth.route_checks import RouteFacts
from litellm.redis.store_settings import RedisBreakerSettings, RedisStoreSettings
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
class NativeSensitiveRoutePins:
    def __new__(cls, client: NativeRedisClient | None) -> NativeSensitiveRoutePins: ...
    def get(self, tenant: str, session_id: str, default_ttl_seconds: float) -> Future[str | None]: ...
    def set(self, tenant: str, session_id: str, model: str, ttl_seconds: float) -> Future[None]: ...

class _UsageCounter(Protocol):
    @property
    def deployment_id(self) -> str: ...
    @property
    def model(self) -> str | None: ...
    @property
    def minute(self) -> str: ...
    @property
    def kind(self) -> Literal["tpm", "rpm"]: ...

@final
class NativeRouterUsage:
    def __new__(cls, client: NativeRedisClient | None) -> NativeRouterUsage: ...
    def attach(self, client: NativeRedisClient) -> bool: ...
    @property
    def has_redis(self) -> bool: ...
    def increment(self, counter: _UsageCounter, value: float, ttl_seconds: int) -> Future[float]: ...
    def increment_deployment(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> Future[None]: ...
    def get_many(self, counters: Sequence[_UsageCounter]) -> Future[list[float | None]]: ...
    def increment_sync(self, counter: _UsageCounter, value: float, ttl_seconds: int) -> float: ...
    def increment_deployment_sync(
        self,
        deployment_id: str,
        model: str | None,
        minute: str,
        tokens: float,
        requests: int,
    ) -> None: ...
    def get_many_sync(self, counters: Sequence[_UsageCounter]) -> list[float | None]: ...
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
    def is_session_token(self) -> bool: ...
    @property
    def user_id(self) -> str | None: ...
    @property
    def team_id(self) -> str | None: ...
    @property
    def team_object_permission_id(self) -> str | None: ...
    @property
    def project_id(self) -> str | None: ...
    @property
    def last_refreshed_at(self) -> float | None: ...
    @property
    def team_member_spend(self) -> float | None: ...
    @property
    def max_budget(self) -> float | None: ...
    @property
    def spend(self) -> float | None: ...
    @property
    def soft_budget(self) -> float | None: ...
    @property
    def key_alias(self) -> str | None: ...
    @property
    def key_name(self) -> str | None: ...
    @property
    def alert_email_lowest_percent(self) -> float | None: ...
    @property
    def alert_emails_configured(self) -> bool: ...
    @property
    def default_warning_share(self) -> float: ...
    @property
    def throttle_share(self) -> float | None: ...

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

class _AuthorizeFacts(Protocol):
    @property
    def route(self) -> str: ...
    @property
    def method(self) -> str | None: ...
    @property
    def is_llm_api_route(self) -> bool: ...
    @property
    def dispatched_to_pass_through(self) -> bool: ...
    @property
    def has_model(self) -> bool: ...
    @property
    def skip_budget_checks(self) -> bool: ...
    @property
    def user_id(self) -> str | None: ...
    @property
    def agent_id(self) -> str | None: ...
    @property
    def team_id(self) -> str | None: ...
    @property
    def team_blocked(self) -> bool: ...
    @property
    def team_requires_trace_id(self) -> bool: ...
    @property
    def enforce_user_param(self) -> bool: ...
    @property
    def has_user_param(self) -> bool: ...
    @property
    def reject_clientside_metadata_tags(self) -> bool: ...
    @property
    def has_client_tags(self) -> bool: ...
    @property
    def block_unpriced_models(self) -> bool: ...
    @property
    def max_budget(self) -> float: ...
    @property
    def global_proxy_spend(self) -> float | None: ...
    @property
    def route_access(self) -> RouteFacts: ...
    @property
    def budgets(self) -> BudgetFacts: ...

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

@final
class RustAuthSettings:
    @staticmethod
    def from_general_settings(
        general_settings: Mapping[str, object],
        litellm_jwtauth: LiteLLM_JWTAuth | None,
        *,
        master_key: str | None,
        premium: bool,
        has_db: bool,
        ui_login: bool,
        proxy_budget: bool,
        metrics_require_auth: bool,
    ) -> RustAuthSettings: ...

@final
class RustPlan(Enum):
    Public = ...
    OAuth2Introspection = ...
    OAuth2Proxy = ...
    Jwt = ...
    PassThroughOpen = ...
    Key = ...

@final
class RustJwtMapping:
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
        @property
        def token(self) -> str | None: ...

    @final
    class UserAuthRoute(RustIdentity):
        __match_args__: tuple[()] = ()

    @final
    class CachedAdmin(RustIdentity):
        __match_args__: tuple[Literal["key"], Literal["end_user"]] = ("key", "end_user")
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def end_user(self) -> EndUserRow: ...

    @final
    class MasterKey(RustIdentity):
        __match_args__: tuple[Literal["hash"], Literal["end_user"]] = ("hash", "end_user")
        @property
        def hash(self) -> str: ...
        @property
        def end_user(self) -> EndUserRow: ...

    @final
    class VirtualKey(RustIdentity):
        __match_args__: tuple[Literal["context"]] = ("context",)
        @property
        def context(self) -> RustKeyContext: ...

    @final
    class MappedJwtKey(RustIdentity):
        __match_args__: tuple[
            Literal["claims"],
            Literal["end_user_id"],
            Literal["user_email"],
            Literal["auto_registered"],
            Literal["minted"],
            Literal["org_id"],
            Literal["user"],
            Literal["context"],
        ] = ("claims", "end_user_id", "user_email", "auto_registered", "minted", "org_id", "user", "context")
        @property
        def claims(self) -> dict[str, object]: ...
        @property
        def end_user_id(self) -> str | None: ...
        @property
        def user_email(self) -> str | None: ...
        @property
        def auto_registered(self) -> bool: ...
        @property
        def minted(self) -> bool: ...
        @property
        def org_id(self) -> str | None: ...
        @property
        def user(self) -> LiteLLM_UserTable | None: ...
        @property
        def context(self) -> RustKeyContext: ...

    @final
    class AutoRegisteredJwtKey(RustIdentity):
        """Only in `RustAuthStep.FinishJwt`."""

        __match_args__: tuple[
            Literal["claims"],
            Literal["end_user_id"],
            Literal["user_email"],
            Literal["minted"],
            Literal["org_id"],
            Literal["user"],
            Literal["key"],
        ] = ("claims", "end_user_id", "user_email", "minted", "org_id", "user", "key")
        @property
        def claims(self) -> dict[str, object]: ...
        @property
        def end_user_id(self) -> str | None: ...
        @property
        def user_email(self) -> str | None: ...
        @property
        def minted(self) -> bool: ...
        @property
        def org_id(self) -> str | None: ...
        @property
        def user(self) -> LiteLLM_UserTable | None: ...
        @property
        def key(self) -> UserAPIKeyAuth: ...

    @final
    class FinishedJwt(RustIdentity):
        __match_args__: tuple[Literal["token"]] = ("token",)
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

    @final
    class DbUnavailable(RustIdentity):
        __match_args__: tuple[Literal["message"]] = ("message",)
        @property
        def message(self) -> str: ...

@disjoint_base
class RustTeamGrant:
    @final
    class NoTeam(RustTeamGrant):
        __match_args__: tuple[()] = ()

    @final
    class Loaded(RustTeamGrant):
        __match_args__: tuple[Literal["team"]] = ("team",)
        @property
        def team(self) -> LiteLLM_TeamTableCachedObj: ...

    @final
    class KeyCopy(RustTeamGrant):
        __match_args__: tuple[Literal["object_permission"]] = ("object_permission",)
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
        @property
        def status(self) -> int: ...
        @property
        def detail(self) -> str: ...

    @final
    class InvalidVirtualKey(RustDenial):
        __match_args__: tuple[Literal["status"], Literal["detail"]] = ("status", "detail")
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
        @property
        def status(self) -> int: ...
        @property
        def detail(self) -> str: ...
        @property
        def internal_message(self) -> str: ...

    @final
    class Plain(RustDenial):
        __match_args__: tuple[Literal["message"]] = ("message",)
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
        @property
        def team_id(self) -> str: ...

@final
class RustAuthOutcome:
    @property
    def denied(self) -> bool: ...
    @property
    def identity(self) -> RustIdentity | None: ...
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
    class EndUser(RustAuthStep):
        __match_args__: tuple[Literal["token"], Literal["resolved"]] = ("token", "resolved")
        @property
        def token(self) -> str: ...
        @property
        def resolved(self) -> UserAPIKeyAuth | None: ...

    @final
    class CachedKey(RustAuthStep):
        __match_args__: tuple[Literal["hash"]] = ("hash",)
        @property
        def hash(self) -> str: ...

    @final
    class UiKey(RustAuthStep):
        __match_args__: tuple[Literal["token"]] = ("token",)
        @property
        def token(self) -> str: ...

    @final
    class WithSessionGrants(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["user"], Literal["team"], Literal["membership"]] = (
            "key",
            "user",
            "team",
            "membership",
        )
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def user(self) -> LiteLLM_UserTable: ...
        @property
        def team(self) -> LiteLLM_TeamTable | None: ...
        @property
        def membership(self) -> LiteLLM_TeamMembership | None: ...

    @final
    class EvictKey(RustAuthStep):
        __match_args__: tuple[Literal["hash"]] = ("hash",)
        @property
        def hash(self) -> str: ...

    @final
    class CachedTeam(RustAuthStep):
        __match_args__: tuple[Literal["team_id"]] = ("team_id",)
        @property
        def team_id(self) -> str: ...

    @final
    class WithTeam(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["team"]] = ("key", "team")
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def team(self) -> LiteLLM_TeamTableCachedObj: ...

    @final
    class DbKey(RustAuthStep):
        __match_args__: tuple[Literal["hash"], Literal["end_user"]] = ("hash", "end_user")
        @property
        def hash(self) -> str: ...
        @property
        def end_user(self) -> EndUserRow: ...

    @final
    class WithTempBudget(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        @property
        def key(self) -> UserAPIKeyAuth: ...

    @final
    class NormalizeModel(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        @property
        def key(self) -> UserAPIKeyAuth: ...

    @final
    class PrefetchAuthObjects(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["end_user"]] = ("key", "end_user")
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def end_user(self) -> EndUserRow: ...

    @final
    class KeyOwner(RustAuthStep):
        __match_args__: tuple[Literal["user_id"]] = ("user_id",)
        @property
        def user_id(self) -> str: ...

    @final
    class KeySkipsBudgetChecks(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        @property
        def key(self) -> UserAPIKeyAuth: ...

    @final
    class TeamMemberMaxBudget(RustAuthStep):
        __match_args__: tuple[Literal["user_id"], Literal["team_id"]] = ("user_id", "team_id")
        @property
        def user_id(self) -> str: ...
        @property
        def team_id(self) -> str: ...

    @final
    class KeyModelBudgets(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["owner"]] = ("key", "owner")
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def owner(self) -> LiteLLM_UserTable | None: ...

    @final
    class ThrottleKey(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["share"]] = ("key", "share")
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def share(self) -> float: ...

    @final
    class KeyTeam(RustAuthStep):
        __match_args__: tuple[Literal["team_id"]] = ("team_id",)
        @property
        def team_id(self) -> str: ...

    @final
    class ObjectPermission(RustAuthStep):
        __match_args__: tuple[Literal["id"]] = ("id",)
        @property
        def id(self) -> str: ...

    @final
    class Project(RustAuthStep):
        __match_args__: tuple[Literal["project_id"]] = ("project_id",)
        @property
        def project_id(self) -> str: ...

    @final
    class GlobalProxySpend(RustAuthStep):
        __match_args__: tuple[()] = ()

    @final
    class KeyBudgetAlert(RustAuthStep):
        __match_args__: tuple[
            Literal["key"], Literal["owner"], Literal["kind"], Literal["spend"], Literal["max_budget"]
        ] = ("key", "owner", "kind", "spend", "max_budget")
        @property
        def key(self) -> UserAPIKeyAuth: ...
        @property
        def owner(self) -> LiteLLM_UserTable | None: ...
        @property
        def kind(self) -> Literal["team_member", "proxy", "max_budget_warning", "token_budget", "soft_budget"]: ...
        @property
        def spend(self) -> float | None: ...
        @property
        def max_budget(self) -> float | None: ...

    @final
    class FetchJson(RustAuthStep):
        __match_args__: tuple[Literal["url"]] = ("url",)
        @property
        def url(self) -> str: ...

    @final
    class FetchUserinfo(RustAuthStep):
        __match_args__: tuple[Literal["endpoint"], Literal["token"]] = ("endpoint", "token")
        @property
        def endpoint(self) -> str: ...
        @property
        def token(self) -> str: ...

    @final
    class CachedJson(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        @property
        def key(self) -> str: ...

    @final
    class CacheJson(RustAuthStep):
        __match_args__: tuple[Literal["key"], Literal["value"], Literal["ttl_seconds"]] = (
            "key",
            "value",
            "ttl_seconds",
        )
        @property
        def key(self) -> str: ...
        @property
        def value(self) -> object: ...
        @property
        def ttl_seconds(self) -> float: ...

    @final
    class ForgetJson(RustAuthStep):
        __match_args__: tuple[Literal["key"]] = ("key",)
        @property
        def key(self) -> str: ...

    @final
    class Team(RustAuthStep):
        __match_args__: tuple[Literal["team_id"], Literal["upsert"]] = ("team_id", "upsert")
        @property
        def team_id(self) -> str: ...
        @property
        def upsert(self) -> bool: ...

    @final
    class TeamByAlias(RustAuthStep):
        __match_args__: tuple[Literal["alias"]] = ("alias",)
        @property
        def alias(self) -> str: ...

    @final
    class GrantingTeam(RustAuthStep):
        __match_args__: tuple[Literal["team_id"]] = ("team_id",)
        @property
        def team_id(self) -> str: ...

    @final
    class LoadTeamMembership(RustAuthStep):
        __match_args__: tuple[Literal["user_id"], Literal["team_id"]] = ("user_id", "team_id")
        @property
        def user_id(self) -> str: ...
        @property
        def team_id(self) -> str: ...

    @final
    class TeamModelAccess(RustAuthStep):
        __match_args__: tuple[Literal["team"], Literal["model"]] = ("team", "model")
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
        @property
        def user(self) -> LiteLLM_UserTable: ...
        @property
        def role(self) -> str: ...

    @final
    class PatchTeamMembership(RustAuthStep):
        __match_args__: tuple[Literal["user"], Literal["add"], Literal["remove"]] = ("user", "add", "remove")
        @property
        def user(self) -> LiteLLM_UserTable: ...
        @property
        def add(self) -> list[str]: ...
        @property
        def remove(self) -> list[str]: ...

    @final
    class AddTeamMember(RustAuthStep):
        __match_args__: tuple[Literal["user_id"], Literal["team_id"]] = ("user_id", "team_id")
        @property
        def user_id(self) -> str: ...
        @property
        def team_id(self) -> str: ...

    @final
    class Org(RustAuthStep):
        __match_args__: tuple[Literal["org_id"]] = ("org_id",)
        @property
        def org_id(self) -> str: ...

    @final
    class OrgByAlias(RustAuthStep):
        __match_args__: tuple[Literal["alias"]] = ("alias",)
        @property
        def alias(self) -> str: ...

    @final
    class JwtEndUser(RustAuthStep):
        __match_args__: tuple[Literal["end_user_id"]] = ("end_user_id",)
        @property
        def end_user_id(self) -> str: ...

    @final
    class FinishJwt(RustAuthStep):
        __match_args__: tuple[Literal["identity"]] = ("identity",)
        @property
        def identity(self) -> RustIdentity: ...

    @final
    class KeyMappingHash(RustAuthStep):
        __match_args__: tuple[Literal["claim_field"], Literal["claim_value"], Literal["issuer"]] = (
            "claim_field",
            "claim_value",
            "issuer",
        )
        @property
        def claim_field(self) -> str: ...
        @property
        def claim_value(self) -> str: ...
        @property
        def issuer(self) -> str | None: ...

    @final
    class KeyByHash(RustAuthStep):
        __match_args__: tuple[Literal["hash"]] = ("hash",)
        @property
        def hash(self) -> str: ...

    @final
    class ReusableKeyHash(RustAuthStep):
        __match_args__: tuple[Literal["user_id"], Literal["team_id"]] = ("user_id", "team_id")
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
        @property
        def claims(self) -> dict[str, object]: ...

    @final
    class RecordAgentAuthentication(RustAuthStep):
        __match_args__: tuple[Literal["agent"]] = ("agent",)
        @property
        def agent(self) -> ManagedAgentContext: ...

    @final
    class RegisteredAgent(RustAuthStep):
        __match_args__: tuple[Literal["claim"]] = ("claim",)
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
        @property
        def claims(self) -> dict[str, object]: ...

    @final
    class UnpricedModels(RustAuthStep):
        __match_args__: tuple[()] = ()

    @final
    class AgentRequiresTraceId(RustAuthStep):
        __match_args__: tuple[Literal["agent_id"]] = ("agent_id",)
        @property
        def agent_id(self) -> str: ...

    @final
    class HasTraceId(RustAuthStep):
        __match_args__: tuple[Literal["headers_only"]] = ("headers_only",)
        @property
        def headers_only(self) -> bool: ...

    @final
    class MergeBudgetTags(RustAuthStep):
        __match_args__: tuple[()] = ()

    @final
    class CheckTeamModelAccess(RustAuthStep):
        __match_args__: tuple[Literal["membership_read"], Literal["membership"]] = ("membership_read", "membership")
        @property
        def membership_read(self) -> bool: ...
        @property
        def membership(self) -> LiteLLM_TeamMembership | None: ...

    @final
    class CheckModelGrants(RustAuthStep):
        __match_args__: tuple[()] = ()

    @final
    class CheckProject(RustAuthStep):
        __match_args__: tuple[()] = ()

    @final
    class StampAccessGroups(RustAuthStep):
        __match_args__: tuple[Literal["membership_read"], Literal["membership"]] = ("membership_read", "membership")
        @property
        def membership_read(self) -> bool: ...
        @property
        def membership(self) -> LiteLLM_TeamMembership | None: ...

    @final
    class Spend(RustAuthStep):
        __match_args__: tuple[
            Literal["scope"],
            Literal["entity_id"],
            Literal["team_id"],
            Literal["window"],
            Literal["recorded"],
            Literal["max_budget"],
        ] = ("scope", "entity_id", "team_id", "window", "recorded", "max_budget")
        @property
        def scope(
            self,
        ) -> Literal[
            "team", "team_window", "key_window", "organization", "tag", "model_access_group", "user", "team_member", "end_user"
        ]: ...
        @property
        def entity_id(self) -> str: ...
        @property
        def team_id(self) -> str | None: ...
        @property
        def window(self) -> str | None: ...
        @property
        def recorded(self) -> float: ...
        @property
        def max_budget(self) -> float: ...

    @final
    class OrganizationBudget(RustAuthStep):
        __match_args__: tuple[Literal["org_id"]] = ("org_id",)
        @property
        def org_id(self) -> str: ...

    @final
    class TagBudgets(RustAuthStep):
        __match_args__: tuple[()] = ()

    @final
    class ModelAccessGroupBudgets(RustAuthStep):
        __match_args__: tuple[Literal["groups"]] = ("groups",)
        @property
        def groups(self) -> list[str]: ...

    @final
    class TeamMemberBudgetState(RustAuthStep):
        __match_args__: tuple[Literal["membership"]] = ("membership",)
        @property
        def membership(self) -> LiteLLM_TeamMembership | None: ...

    @final
    class BudgetAlert(RustAuthStep):
        __match_args__: tuple[Literal["kind"], Literal["org_id"], Literal["spend"], Literal["max_budget"]] = (
            "kind",
            "org_id",
            "spend",
            "max_budget",
        )
        @property
        def kind(self) -> Literal["team", "team_soft_budget", "organization", "user", "team_member"]: ...
        @property
        def org_id(self) -> str | None: ...
        @property
        def spend(self) -> float | None: ...
        @property
        def max_budget(self) -> float | None: ...

    @final
    class CheckGuardrailModification(RustAuthStep):
        __match_args__: tuple[()] = ()

    @final
    class TeamOrganization(RustAuthStep):
        __match_args__: tuple[Literal["team_id"]] = ("team_id",)
        @property
        def team_id(self) -> str: ...

    @final
    class CheckObjectAccess(RustAuthStep):
        __match_args__: tuple[()] = ()

    @final
    class Done(RustAuthStep):
        __match_args__: tuple[Literal["outcome"]] = ("outcome",)
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
    def authorize(facts: _AuthorizeFacts) -> RustAuth: ...
    @staticmethod
    def key_max_budget(
        facts: _KeyFacts, key: UserAPIKeyAuth, owner: LiteLLM_UserTable | None = None
    ) -> RustAuth: ...
    @staticmethod
    def verify_jwt(token: str, now: float, settings: RustAuthSettings) -> RustAuth: ...
    @staticmethod
    def verify_jwt_assertion(
        token: str, now: float, settings: RustAuthSettings, *, audience: list[str], required_claims: list[str]
    ) -> RustAuth: ...
    def resume(self) -> RustAuthStep: ...
    def fail(self) -> None: ...
    def unavailable(self, *, dependency: Literal["db"], transient: bool, message: str) -> None: ...
    def done(self) -> None: ...
    def budget_exceeded(self, error: _BudgetExceeded) -> None: ...
    def found(self, found: bool) -> None: ...
    def amount(self, value: float | None = None) -> None: ...
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
    def names(self, names: list[str]) -> None: ...
    def spend(self, spend: float) -> None: ...
    def organization_budget(self, max_budget: float | None = None, spend: float = 0.0) -> None: ...
    def named_budgets(self, budgets: list[tuple[str, float, float]]) -> None: ...
    def member_budget(
        self,
        *,
        spend: float,
        budget: tuple[float | None, float | None, float | None] | None = None,
        team_default: float | None = None,
    ) -> None: ...
    def mapped_key(
        self, facts: _KeyFacts | None = None, row: UserAPIKeyAuth | None = None, *, user_email: str | None = None
    ) -> None: ...
    def key_mapping(self, *, token_hash: str | None, conflict: bool) -> None: ...
    def managed_agent(self, agent: _ManagedAgent | None = None) -> None: ...
    def agent_failure(self, failure: _AgentFailure | None = None) -> None: ...
    def registered_agent(self, agent: _RegisteredAgent | None = None) -> None: ...

def jwt_is_jwt(token: str | None) -> bool: ...
def jwt_unverified_claims(token: str) -> dict[str, object] | None: ...
def jwt_claims(claims: dict[str, object], litellm_jwtauth: LiteLLM_JWTAuth) -> RustMappedClaims: ...
def jwt_is_allowed_domain(user_email: str, litellm_jwtauth: LiteLLM_JWTAuth) -> bool: ...
def jwt_key_mapping_cache_key(claim_field: str, claim_value: str, issuer: str | None = None) -> str: ...
def auth_route_matches_wildcard(route: str, pattern: str) -> bool: ...
def auth_managed_route(route: str) -> Literal["discovery", "realtime", "live", "model", "tool", "other"]: ...
def auth_managed_agent_route_allowed(route: str, method: str | None = None) -> bool: ...
def auth_organization_denial(facts: RouteFacts) -> RustDenial | None: ...
def auth_route_denial(facts: RouteFacts, organization_id: str | None = None) -> RustDenial | None: ...
def auth_viewer_route_denial(facts: RouteFacts) -> RustDenial | None: ...
def auth_user_is_org_admin(facts: RouteFacts) -> bool: ...
def auth_org_context_team(facts: RouteFacts) -> str | None: ...
def auth_mask_user_id(user_id: str) -> str: ...
def auth_key_max_budget_warning_due(facts: _KeyFacts) -> bool: ...
def auth_key_soft_budget_due(facts: _KeyFacts) -> bool: ...
def auth_route_matches_prefix(route: str, allowed_route: str) -> bool: ...
def auth_route_matches_denied(route: str, denied_route: str) -> bool: ...
def auth_first_denied_route(forwarded_routes: Sequence[str], denied_routes: Sequence[str]) -> str | None: ...
def auth_jwt_team_routes_grant(route: str, team_allowed_routes: Sequence[str]) -> bool: ...
def auth_allowed_routes_grant(route: str, *, key: Sequence[str] | None, team: Sequence[str] | None) -> bool: ...
def auth_end_user_id(
    request_body: Mapping[str, object],
    request_headers: Mapping[str, object] | None,
    general_settings: Mapping[str, object],
    validate_end_user_id_in_db: bool,
) -> str | None: ...

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
    "NativeRouterUsage",
    "NativeSend",
    "NativeSensitiveRoutePins",
    "NativeSpendCounters",
    "NativeTableSwapped",
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
    "RustMappedClaims",
    "RustPlan",
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
    "auth_end_user_id",
    "auth_first_denied_route",
    "auth_jwt_team_routes_grant",
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

# One RouterCache op against the router's DualCache. `key` is set for every op but `batch_get`,
# and `increment_usage`, which set `keys`; `value` is the JSON to store (`set`), the amount
# (`increment`) or one amount per key (`increment_usage`); `ttl` is in seconds; `local` reads or
# writes this worker's in-memory tier only. Outcomes: get -> the JSON value or None; batch_get ->
# one JSON value or None per key, in order; increment -> the new int or float; set, delete and
# increment_usage -> None.
class _CacheStep(_Step):
    kind: Literal["cache"]
    op: Literal["get", "set", "increment", "increment_usage", "batch_get", "delete"]
    key: str | None
    keys: list[str] | None
    value: JsonValue
    ttl: float | None
    local: bool

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

# The end of a `RouterCoreHandle.report_success` or `count_success`.
class _CountedStep(_Step):
    kind: Literal["done"]
    key: str

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
_SuccessReportSteps = _CacheStep | _CountedStep | _FailedStep

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
    def usage_key(self, id: str, model: str, metric: str, minute: str) -> str: ...
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
    # Count a success against any id, driven like `report_failure`; `done` carries `key`.
    def count_success(
        self, id: str, callbacks: Mapping[str, bool] | None = None
    ) -> RouterCall[_SuccessReportSteps]: ...
    # [(id, tpm_key, rpm_key)] for this minute, or None for a group with no deployment.
    def usage_keys(self, model_group: str) -> list[tuple[str, str, str]] | None: ...
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
