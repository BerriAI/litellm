import asyncio
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import TYPE_CHECKING, Any, Final, Literal, NoReturn, Protocol

from pydantic import BaseModel, TypeAdapter
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm import DualCache, EmbeddingResponse, ModelResponse, TextCompletionResponse
from litellm._internal_context import with_service_target
from litellm._logging import verbose_proxy_logger
from litellm.caching.redis_batch import active_request_redis_batch
from litellm.exceptions import RateLimitType
from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs
from litellm.proxy._types import CommonProxyErrors, UserAPIKeyAuth
from litellm.proxy.auth.auth_utils import (
    get_key_model_rpm_limit,
    get_key_model_tpm_limit,
)
from litellm.proxy.auth.budget_throttle import throttled_limit
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.proxy.litellm_pre_call_utils import LiteLLMProxyRequestSetup
from litellm.redis.stores.legacy_request_limits import (
    GlobalLimit,
    InvalidSnapshot,
    LegacyRequestLimitsStore,
    LegacySnapshot,
    Limit,
    MinuteLimit,
    TransportLegacyRequestLimitsStore,
    UsageSnapshot,
    limit_key,
    snapshot,
)
from litellm.redis.transport.contract import RedisFailure
from litellm.types.utils import CallTypes, Usage

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    from litellm.proxy.utils import InternalUsageCache as _InternalUsageCache

    Span = _Span
    InternalUsageCache = _InternalUsageCache
else:
    Span = Any
    InternalUsageCache = Any


def _precise_minute(now: datetime) -> str:
    return now.strftime("%Y-%m-%d-%H-%M")


def _response_total_tokens(response_obj: object) -> int:
    if not isinstance(response_obj, (ModelResponse, EmbeddingResponse, TextCompletionResponse)):
        return 0
    response_usage: Final = getattr(response_obj, "usage", None)
    return response_usage.total_tokens if isinstance(response_usage, Usage) else 0


_CALLBACK_DATA: Final = TypeAdapter(dict[str, object])


def _optional_identity(value: object) -> str | None:
    return None if value is None else str(value)


_CONNECTION_SCOPED_CALL_TYPES: Final = frozenset({CallTypes.aresponses_websocket.value})


def _slots_released(kwargs: Mapping[str, object]) -> int:
    """A websocket turn callback releases nothing; the socket's slot is released once at disconnect."""
    return 0 if kwargs.get("call_type") in _CONNECTION_SCOPED_CALL_TYPES else 1


def _usage(value: LegacySnapshot | None | RedisFailure, *, empty_is_missing: bool = False) -> UsageSnapshot | None:
    if empty_is_missing and (not value or isinstance(value, InvalidSnapshot) and value.empty):
        return None
    if value is None or isinstance(value, dict):
        return value
    raise ValueError("invalid legacy request limit snapshot")


class _LocalLimitCache(Protocol):
    async def async_get_cache(self, key: str, *, local_only: bool) -> object: ...

    async def async_batch_get_cache(self, keys: list[str], *, local_only: bool) -> list[object] | None: ...

    async def async_set_cache(self, key: str, value: object, *, local_only: bool, ttl: int | None = None) -> None: ...

    async def async_set_cache_pipeline(
        self, cache_list: Sequence[tuple[str, object]], *, local_only: bool, ttl: int | None = None
    ) -> None: ...


@dataclass(frozen=True, slots=True)
class _LegacyLimitCache:
    cache: DualCache

    @property
    def local(self) -> _LocalLimitCache:
        return self.cache

    def store(self) -> LegacyRequestLimitsStore | None:
        redis: Final = self.cache.redis_cache
        return None if redis is None else TransportLegacyRequestLimitsStore(redis.transport())

    async def read(
        self, key: MinuteLimit, litellm_parent_otel_span: Span | None, *, empty_is_missing: bool = False
    ) -> UsageSnapshot | None:
        local: Final = await self.local.async_get_cache(limit_key(key), local_only=True)
        if local is not None:
            return _usage(snapshot(local), empty_is_missing=empty_is_missing)
        store: Final = self.store()
        if store is None:
            return None
        redis: Final = self.cache.redis_cache
        request: Final = None if redis is None else active_request_redis_batch(redis)
        if request is not None and request.read_as_missing(limit_key(key)):
            return None
        remote: Final = await store.read(key)
        if isinstance(remote, RedisFailure):
            return None
        if remote is not None:
            await self.local.async_set_cache(limit_key(key), remote, local_only=True)
        return _usage(remote, empty_is_missing=empty_is_missing)

    async def read_many(self, limits: Sequence[Limit | None]) -> tuple[LegacySnapshot | None | RedisFailure, ...]:
        present: Final = tuple(limit for limit in limits if limit is not None)
        keys: Final = tuple(limit_key(limit) for limit in present)
        local: Final = await self.local.async_batch_get_cache(list(keys), local_only=True)
        local_by_key: Final = dict(zip(keys, local or (None for _ in keys)))
        store: Final = self.store()
        if store is None:
            return tuple(None if limit is None else snapshot(local_by_key[limit_key(limit)]) for limit in limits)
        missing, reserved = self.cache.reserve_redis_batch_reads(keys)
        needed: Final = tuple(limit for limit in present if limit_key(limit) in missing)
        remote: Final = await store.read_many(needed) if needed else ()
        if isinstance(remote, RedisFailure):
            if remote.kind == "breaker_open":
                self.cache.rollback_redis_batch_reads(reserved)
            return tuple(None if limit is None else snapshot(local_by_key[limit_key(limit)]) for limit in limits)
        remote_by_key: Final = dict(zip((limit_key(limit) for limit in needed), remote))
        for key, value in remote_by_key.items():
            if value is not None and not isinstance(value, RedisFailure):
                await self.local.async_set_cache(key, value, local_only=True)
        merged: Final = {**local_by_key, **remote_by_key}
        return tuple(None if limit is None else _snapshot_result(merged[limit_key(limit)]) for limit in limits)

    async def write_many(
        self,
        cache_list: Sequence[tuple[MinuteLimit, UsageSnapshot]],
        ttl: int,
        litellm_parent_otel_span: Span | None,
        local_only: bool = False,
    ) -> None:
        await self.local.async_set_cache_pipeline(
            tuple((limit_key(limit), value) for limit, value in cache_list), ttl=ttl, local_only=True
        )
        store: Final = self.store()
        if store is not None and not local_only:
            await store.write_many(cache_list, ttl=timedelta(seconds=ttl))

    async def write(
        self, key: MinuteLimit, value: UsageSnapshot, ttl: int, litellm_parent_otel_span: Span | None
    ) -> None:
        await self.local.async_set_cache(limit_key(key), value, ttl=ttl, local_only=True)
        store: Final = self.store()
        if store is not None:
            await store.write(key, value, ttl=timedelta(seconds=ttl))


def _snapshot_result(value: object) -> LegacySnapshot | None | RedisFailure:
    return value if isinstance(value, RedisFailure) else snapshot(value)


class CacheObject(TypedDict):
    current_global_requests: ReadOnly[LegacySnapshot | None | RedisFailure]
    request_count_api_key: ReadOnly[LegacySnapshot | None | RedisFailure]
    request_count_api_key_model: ReadOnly[LegacySnapshot | None | RedisFailure]
    request_count_user_id: ReadOnly[LegacySnapshot | None | RedisFailure]
    request_count_team_id: ReadOnly[LegacySnapshot | None | RedisFailure]
    request_count_end_user_id: ReadOnly[LegacySnapshot | None | RedisFailure]


class _PROXY_MaxParallelRequestsHandler(CustomLogger):
    # Class variables or attributes
    def __init__(self, internal_usage_cache: InternalUsageCache, *, clock: Callable[[], datetime] = datetime.now):
        self.internal_usage_cache = internal_usage_cache
        self._limits: Final = _LegacyLimitCache(internal_usage_cache.dual_cache)
        self._clock: Final = clock

    def print_verbose(self, print_statement) -> None:
        try:
            verbose_proxy_logger.debug(print_statement)
            if litellm.set_verbose:
                print(print_statement)  # noqa: T201
        except Exception:
            pass

    @with_service_target("rate_limits")
    async def check_key_in_limits(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
        max_parallel_requests: int,
        tpm_limit: int,
        rpm_limit: int,
        current: LegacySnapshot | None | RedisFailure,
        request_count_api_key: MinuteLimit,
        rate_limit_type: Literal["key", "model_per_key", "user", "customer", "team"],
        values_to_update_in_cache: list[tuple[MinuteLimit, UsageSnapshot]],
    ) -> dict:
        usage: Final = _usage(current)
        verbose_proxy_logger.info("Current Usage of %s in this minute: %s", rate_limit_type, usage)
        if usage is None:
            if max_parallel_requests == 0 or tpm_limit == 0 or rpm_limit == 0:
                # base case — at least one dimension is set to 0 (effectively
                # disabled). Pick the most specific dimension as the
                # rate_limit_type so dashboards can attribute the failure to
                # the right cap. Order matters: max_parallel_requests is
                # listed first because it's the rarest 0 in practice and the
                # most actionable signal.
                if max_parallel_requests == 0:
                    triggered_type = RateLimitType.CONCURRENT_REQUESTS
                elif tpm_limit == 0:
                    triggered_type = RateLimitType.TOKENS
                else:
                    triggered_type = RateLimitType.REQUESTS
                self.raise_rate_limit_error(
                    additional_details=f"{CommonProxyErrors.max_parallel_request_limit_reached.value}. Hit limit for {rate_limit_type}. Current limits: max_parallel_requests: {max_parallel_requests}, tpm_limit: {tpm_limit}, rpm_limit: {rpm_limit}",
                    rate_limit_type=triggered_type,
                    requested_model=data.get("model") if data else None,
                    descriptor_key=rate_limit_type,
                )
            new_val: UsageSnapshot = {
                "current_requests": 1,
                "current_tpm": 0,
                "current_rpm": 1,
            }
            values_to_update_in_cache.append((request_count_api_key, new_val))
        elif (
            int(usage["current_requests"]) < max_parallel_requests
            and usage["current_tpm"] < tpm_limit
            and usage["current_rpm"] < rpm_limit
        ):
            # Increase count for this token
            new_val = {
                "current_requests": usage["current_requests"] + 1,
                "current_tpm": usage["current_tpm"],
                "current_rpm": usage["current_rpm"] + 1,
            }
            values_to_update_in_cache.append((request_count_api_key, new_val))

        else:
            # Detect which dimension actually tripped the limit so we can
            # surface the right rate_limit_type. Order matches the boolean
            # condition above (concurrent → tpm → rpm) — first match wins.
            if int(usage["current_requests"]) >= max_parallel_requests:
                triggered_type = RateLimitType.CONCURRENT_REQUESTS
            elif usage["current_tpm"] >= tpm_limit:
                triggered_type = RateLimitType.TOKENS
            else:
                triggered_type = RateLimitType.REQUESTS
            requested_model: Final = data.get("model") if data else None
            resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(requested_model)
            raise ProxyRateLimitError(
                detail=f"LiteLLM Rate Limit Handler for rate limit type = {rate_limit_type}. {CommonProxyErrors.max_parallel_request_limit_reached.value}. current rpm: {usage['current_rpm']}, rpm limit: {rpm_limit}, current tpm: {usage['current_tpm']}, tpm limit: {tpm_limit}, current max_parallel_requests: {usage['current_requests']}, max_parallel_requests: {max_parallel_requests}",
                headers={"retry-after": str(self.time_to_next_minute())},
                rate_limit_type=triggered_type,
                model=resolved_model,
                llm_provider=llm_provider,
                descriptor_key=rate_limit_type,
            )

        await self._limits.write_many(
            cache_list=values_to_update_in_cache,
            ttl=60,
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            local_only=True,
        )
        return new_val

    def time_to_next_minute(self) -> float:
        # Get the current time
        now: Final = self._clock()

        # Calculate the next minute
        next_minute: Final = (now + timedelta(minutes=1)).replace(second=0, microsecond=0)

        # Calculate the difference in seconds
        seconds_to_next_minute: Final = (next_minute - now).total_seconds()

        return seconds_to_next_minute

    def raise_rate_limit_error(
        self,
        additional_details: str | None = None,
        rate_limit_type: RateLimitType | None = None,
        requested_model: str | None = None,
        descriptor_key: str | None = None,
    ) -> NoReturn:
        """
        Raise a 429 with a retry-after header for litellm-proxy parallel-request limits.

        Always raises :class:`ProxyRateLimitError` — never returns. Annotated
        ``NoReturn`` so type-checkers know callers after this invocation are
        unreachable. The raised exception is both a
        :class:`litellm.RateLimitError` (so callers can catch by category) and a
        :class:`fastapi.HTTPException` (so the FastAPI dispatcher serializes it
        correctly with status 429 and the supplied headers).

        ``rate_limit_type`` defaults to ``CONCURRENT_REQUESTS`` because every
        existing internal caller of this helper hits the parallel-request cap
        (the global-limit branch in ``async_pre_call_hook`` and the
        all-zeros base case in ``check_key_in_limits``). Callers that know
        the dimension exactly should pass it explicitly.

        ``requested_model`` is resolved via :func:`get_llm_provider` so the
        raised exception carries ``llm_provider`` (and a stripped ``model``)
        for downstream loggers (Prometheus failure metric, observability
        callbacks). Falls back to ``llm_provider="litellm_proxy"`` when the
        model is missing or unparseable — see
        :func:`resolve_llm_provider_for_rate_limit`.
        """
        # additional_details is optional; build the detail with a None-guard
        # so callers that pass nothing don't get the literal string "None"
        # interpolated into the error message.
        error_message = "Max parallel request limit reached"
        if additional_details is not None:
            error_message = error_message + " " + additional_details
        resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(requested_model)
        raise ProxyRateLimitError(
            detail=error_message,
            headers={"retry-after": str(self.time_to_next_minute())},
            rate_limit_type=rate_limit_type or RateLimitType.CONCURRENT_REQUESTS,
            model=resolved_model,
            llm_provider=llm_provider,
            descriptor_key=descriptor_key,
        )

    @with_service_target("rate_limits")
    async def get_all_cache_objects(
        self,
        current_global_requests: str | None,
        request_count_api_key: MinuteLimit | None,
        request_count_api_key_model: MinuteLimit | None,
        request_count_user_id: MinuteLimit | None,
        request_count_team_id: MinuteLimit | None,
        request_count_end_user_id: MinuteLimit | None,
        parent_otel_span: Span | None = None,
    ) -> CacheObject:
        results: Final = await self._limits.read_many(
            (
                GlobalLimit() if current_global_requests is not None else None,
                request_count_api_key,
                request_count_api_key_model,
                request_count_user_id,
                request_count_team_id,
                request_count_end_user_id,
            )
        )
        return CacheObject(
            current_global_requests=results[0],
            request_count_api_key=results[1],
            request_count_api_key_model=results[2],
            request_count_user_id=results[3],
            request_count_team_id=results[4],
            request_count_end_user_id=results[5],
        )

    @with_service_target("rate_limits")
    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,
        call_type: str,
    ) -> None:
        self.print_verbose("Inside Max Parallel Request Pre-Call Hook")
        api_key: Final = LiteLLMProxyRequestSetup.get_logged_api_key(user_api_key_dict)
        max_parallel_requests = user_api_key_dict.max_parallel_requests
        if max_parallel_requests is None:
            max_parallel_requests = sys.maxsize
        if data is None:
            data = {}
        global_max_parallel_requests: Final = data.get("metadata", {}).get("global_max_parallel_requests", None)
        throttle_pct: Final = getattr(user_api_key_dict, "budget_throttle_pct", None)
        tpm_limit = throttled_limit(getattr(user_api_key_dict, "tpm_limit", sys.maxsize), throttle_pct)
        if tpm_limit is None:
            tpm_limit = sys.maxsize
        rpm_limit = throttled_limit(getattr(user_api_key_dict, "rpm_limit", sys.maxsize), throttle_pct)
        if rpm_limit is None:
            rpm_limit = sys.maxsize

        values_to_update_in_cache: list[
            tuple[MinuteLimit, UsageSnapshot]
        ] = []  # values that need to get updated in cache, will run a batch_set_cache after this function

        # ------------
        # Setup values
        # ------------
        new_val: dict | None = None

        if global_max_parallel_requests is not None:
            # get value from cache
            _key: Final = "global_max_parallel_requests"
            current_global_requests = await self.internal_usage_cache.async_get_cache(
                key=_key,
                local_only=True,
                litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            )
            # check if below limit
            if current_global_requests is None:
                current_global_requests = 1
            # if above -> raise error
            if current_global_requests >= global_max_parallel_requests:
                self.raise_rate_limit_error(
                    additional_details=f"Hit Global Limit: Limit={global_max_parallel_requests}, current: {current_global_requests}",
                    requested_model=data.get("model") if data else None,
                )
            # if below -> increment
            else:
                await self.internal_usage_cache.async_increment_cache(
                    key=_key,
                    value=1,
                    local_only=True,
                    litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
                )
        _model = data.get("model", None)

        precise_minute: Final = _precise_minute(self._clock())

        cache_objects: Final[CacheObject] = await self.get_all_cache_objects(
            current_global_requests=(
                "global_max_parallel_requests" if global_max_parallel_requests is not None else None
            ),
            request_count_api_key=(MinuteLimit(str(api_key), precise_minute) if api_key is not None else None),
            request_count_api_key_model=(
                MinuteLimit(str(api_key), precise_minute, _model)
                if api_key is not None and _model is not None
                else None
            ),
            request_count_user_id=(
                MinuteLimit(user_api_key_dict.user_id, precise_minute)
                if user_api_key_dict.user_id is not None
                else None
            ),
            request_count_team_id=(
                MinuteLimit(user_api_key_dict.team_id, precise_minute)
                if user_api_key_dict.team_id is not None
                else None
            ),
            request_count_end_user_id=(
                MinuteLimit(user_api_key_dict.end_user_id, precise_minute)
                if user_api_key_dict.end_user_id is not None
                else None
            ),
            parent_otel_span=user_api_key_dict.parent_otel_span,
        )
        if api_key is not None:
            request_count_api_key = MinuteLimit(str(api_key), precise_minute)
            # CHECK IF REQUEST ALLOWED for key
            await self.check_key_in_limits(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type=call_type,
                max_parallel_requests=max_parallel_requests,
                current=cache_objects["request_count_api_key"],
                request_count_api_key=request_count_api_key,
                tpm_limit=tpm_limit,
                rpm_limit=rpm_limit,
                rate_limit_type="key",
                values_to_update_in_cache=values_to_update_in_cache,
            )

        # Check if request under RPM/TPM per model for a given API Key
        _model = data.get("model", None)
        _tpm_limit_for_key_model: Final = get_key_model_tpm_limit(user_api_key_dict, model_name=_model)
        _rpm_limit_for_key_model: Final = get_key_model_rpm_limit(user_api_key_dict, model_name=_model)
        if _tpm_limit_for_key_model is not None or _rpm_limit_for_key_model is not None:
            request_count_api_key = MinuteLimit(str(api_key), precise_minute, _model)
            tpm_limit_for_model = None
            rpm_limit_for_model = None

            if _model is not None:
                if _tpm_limit_for_key_model:
                    tpm_limit_for_model = _tpm_limit_for_key_model.get(_model)

                if _rpm_limit_for_key_model:
                    rpm_limit_for_model = _rpm_limit_for_key_model.get(_model)

            new_val = await self.check_key_in_limits(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type=call_type,
                max_parallel_requests=sys.maxsize,  # TODO: Support max parallel requests for a model
                current=cache_objects["request_count_api_key_model"],
                request_count_api_key=request_count_api_key,
                tpm_limit=tpm_limit_for_model or sys.maxsize,
                rpm_limit=rpm_limit_for_model or sys.maxsize,
                rate_limit_type="model_per_key",
                values_to_update_in_cache=values_to_update_in_cache,
            )
            _remaining_tokens = None
            _remaining_requests = None
            # Add remaining tokens, requests to metadata
            if new_val:
                if tpm_limit_for_model is not None:
                    _remaining_tokens = tpm_limit_for_model - new_val["current_tpm"]
                if rpm_limit_for_model is not None:
                    _remaining_requests = rpm_limit_for_model - new_val["current_rpm"]

            _remaining_limits_data: Final = {
                f"litellm-key-remaining-tokens-{_model}": _remaining_tokens,
                f"litellm-key-remaining-requests-{_model}": _remaining_requests,
            }

            if "metadata" not in data:
                data["metadata"] = {}
            data["metadata"].update(_remaining_limits_data)

        # check if REQUEST ALLOWED for user_id
        user_id: Final = user_api_key_dict.user_id
        if user_id is not None:
            user_tpm_limit = user_api_key_dict.user_tpm_limit
            user_rpm_limit = user_api_key_dict.user_rpm_limit
            if user_tpm_limit is None:
                user_tpm_limit = sys.maxsize
            if user_rpm_limit is None:
                user_rpm_limit = sys.maxsize

            request_count_api_key = MinuteLimit(user_id, precise_minute)
            # print(f"Checking if {request_count_api_key} is allowed to make request for minute {precise_minute}")
            await self.check_key_in_limits(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type=call_type,
                max_parallel_requests=sys.maxsize,  # TODO: Support max parallel requests for a user
                current=cache_objects["request_count_user_id"],
                request_count_api_key=request_count_api_key,
                tpm_limit=user_tpm_limit,
                rpm_limit=user_rpm_limit,
                rate_limit_type="user",
                values_to_update_in_cache=values_to_update_in_cache,
            )

        # TEAM RATE LIMITS
        ## get team tpm/rpm limits
        team_id: Final = user_api_key_dict.team_id
        if team_id is not None:
            team_tpm_limit = user_api_key_dict.team_tpm_limit
            team_rpm_limit = user_api_key_dict.team_rpm_limit

            if team_tpm_limit is None:
                team_tpm_limit = sys.maxsize
            if team_rpm_limit is None:
                team_rpm_limit = sys.maxsize

            request_count_api_key = MinuteLimit(team_id, precise_minute)
            # print(f"Checking if {request_count_api_key} is allowed to make request for minute {precise_minute}")
            await self.check_key_in_limits(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type=call_type,
                max_parallel_requests=sys.maxsize,  # TODO: Support max parallel requests for a team
                current=cache_objects["request_count_team_id"],
                request_count_api_key=request_count_api_key,
                tpm_limit=team_tpm_limit,
                rpm_limit=team_rpm_limit,
                rate_limit_type="team",
                values_to_update_in_cache=values_to_update_in_cache,
            )

        # End-User Rate Limits
        # Only enforce if user passed `user` to /chat, /completions, /embeddings
        if user_api_key_dict.end_user_id:
            end_user_tpm_limit = getattr(user_api_key_dict, "end_user_tpm_limit", sys.maxsize)
            end_user_rpm_limit = getattr(user_api_key_dict, "end_user_rpm_limit", sys.maxsize)

            if end_user_tpm_limit is None:
                end_user_tpm_limit = sys.maxsize
            if end_user_rpm_limit is None:
                end_user_rpm_limit = sys.maxsize

            # now do the same tpm/rpm checks
            request_count_api_key = MinuteLimit(user_api_key_dict.end_user_id, precise_minute)

            # print(f"Checking if {request_count_api_key} is allowed to make request for minute {precise_minute}")
            await self.check_key_in_limits(
                user_api_key_dict=user_api_key_dict,
                cache=cache,
                data=data,
                call_type=call_type,
                max_parallel_requests=sys.maxsize,  # TODO: Support max parallel requests for an End-User
                request_count_api_key=request_count_api_key,
                current=cache_objects["request_count_end_user_id"],
                tpm_limit=end_user_tpm_limit,
                rpm_limit=end_user_rpm_limit,
                rate_limit_type="customer",
                values_to_update_in_cache=values_to_update_in_cache,
            )

        asyncio.create_task(
            self._limits.write_many(
                cache_list=values_to_update_in_cache,
                ttl=60,
                litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
            )  # don't block execution for cache updates
        )

    @with_service_target("rate_limits")
    async def async_log_success_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        from litellm.proxy.common_utils.callback_utils import (
            get_model_group_from_litellm_kwargs,
        )

        litellm_parent_otel_span: Final[Span | None] = get_parent_otel_span_from_kwargs(kwargs=dict(kwargs))
        try:
            self.print_verbose("INSIDE parallel request limiter ASYNC SUCCESS LOGGING")
            released: Final = _slots_released(kwargs)

            params: Final = _CALLBACK_DATA.validate_python(kwargs["litellm_params"])
            metadata: Final = _CALLBACK_DATA.validate_python(params["metadata"])
            global_max_parallel_requests: Final = metadata.get("global_max_parallel_requests")
            user_api_key: Final = _optional_identity(metadata["user_api_key"])
            user_api_key_user_id: Final = _optional_identity(metadata.get("user_api_key_user_id"))
            user_api_key_team_id: Final = _optional_identity(metadata.get("user_api_key_team_id"))
            user_api_key_model_max_budget: Final = metadata.get("user_api_key_model_max_budget")
            user_api_key_end_user_id: Final = _optional_identity(kwargs.get("user"))
            user_api_key_metadata: Final = _CALLBACK_DATA.validate_python(metadata.get("user_api_key_metadata") or {})
            user_api_key_dict: Final = UserAPIKeyAuth.model_validate(
                {
                    "api_key": user_api_key,
                    "metadata": user_api_key_metadata,
                    "model_max_budget": user_api_key_model_max_budget,
                    "team_metadata": metadata.get("user_api_key_team_metadata"),
                }
            )

            # ------------
            # Setup values
            # ------------

            if global_max_parallel_requests is not None:
                # get value from cache
                _key: Final = "global_max_parallel_requests"
                # decrement
                await self.internal_usage_cache.async_increment_cache(
                    key=_key,
                    value=-released,
                    local_only=True,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                )

            precise_minute: Final = _precise_minute(self._clock())

            total_tokens: int = _response_total_tokens(response_obj)

            # ------------
            # Update usage - API Key
            # ------------

            values_to_update_in_cache: Final[list[tuple[MinuteLimit, UsageSnapshot]]] = []

            if user_api_key is not None:
                request_count_api_key = MinuteLimit(user_api_key, precise_minute)

                current: UsageSnapshot = await self._limits.read(
                    key=request_count_api_key,
                    empty_is_missing=True,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                ) or {
                    "current_requests": 1,
                    "current_tpm": 0,
                    "current_rpm": 0,
                }

                new_val = {
                    "current_requests": max(current["current_requests"] - released, 0),
                    "current_tpm": current["current_tpm"] + total_tokens,
                    "current_rpm": current["current_rpm"],
                }

                self.print_verbose(f"updated_value in success call: {new_val}, precise_minute: {precise_minute}")
                values_to_update_in_cache.append((request_count_api_key, new_val))

            # ------------
            # Update usage - model group + API Key
            # ------------
            model_group: Final = get_model_group_from_litellm_kwargs(dict(kwargs))
            _success_tpm_limit: Final = (
                get_key_model_tpm_limit(user_api_key_dict, model_name=model_group) if model_group is not None else None
            )
            _success_rpm_limit: Final = (
                get_key_model_rpm_limit(user_api_key_dict, model_name=model_group) if model_group is not None else None
            )
            if (
                user_api_key is not None
                and model_group is not None
                and (
                    "model_rpm_limit" in user_api_key_metadata
                    or "model_tpm_limit" in user_api_key_metadata
                    or user_api_key_model_max_budget is not None
                    or _success_tpm_limit is not None
                    or _success_rpm_limit is not None
                )
            ):
                request_count_api_key = MinuteLimit(user_api_key, precise_minute, model_group)

                current = await self._limits.read(
                    key=request_count_api_key,
                    empty_is_missing=True,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                ) or {
                    "current_requests": 1,
                    "current_tpm": 0,
                    "current_rpm": 0,
                }

                new_val = {
                    "current_requests": max(current["current_requests"] - released, 0),
                    "current_tpm": current["current_tpm"] + total_tokens,
                    "current_rpm": current["current_rpm"],
                }

                self.print_verbose(f"updated_value in success call: {new_val}, precise_minute: {precise_minute}")
                values_to_update_in_cache.append((request_count_api_key, new_val))

            # ------------
            # Update usage - User
            # ------------
            if user_api_key_user_id is not None:
                total_tokens = _response_total_tokens(response_obj)

                request_count_api_key = MinuteLimit(user_api_key_user_id, precise_minute)

                current = await self._limits.read(
                    key=request_count_api_key,
                    empty_is_missing=True,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                ) or {
                    "current_requests": 1,
                    "current_tpm": total_tokens,
                    "current_rpm": 1,
                }

                new_val = {
                    "current_requests": max(current["current_requests"] - released, 0),
                    "current_tpm": current["current_tpm"] + total_tokens,
                    "current_rpm": current["current_rpm"],
                }

                self.print_verbose(f"updated_value in success call: {new_val}, precise_minute: {precise_minute}")
                values_to_update_in_cache.append((request_count_api_key, new_val))

            # ------------
            # Update usage - Team
            # ------------
            if user_api_key_team_id is not None:
                total_tokens = _response_total_tokens(response_obj)

                request_count_api_key = MinuteLimit(user_api_key_team_id, precise_minute)

                current = await self._limits.read(
                    key=request_count_api_key,
                    empty_is_missing=True,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                ) or {
                    "current_requests": 1,
                    "current_tpm": total_tokens,
                    "current_rpm": 1,
                }

                new_val = {
                    "current_requests": max(current["current_requests"] - released, 0),
                    "current_tpm": current["current_tpm"] + total_tokens,
                    "current_rpm": current["current_rpm"],
                }

                self.print_verbose(f"updated_value in success call: {new_val}, precise_minute: {precise_minute}")
                values_to_update_in_cache.append((request_count_api_key, new_val))

            # ------------
            # Update usage - End User
            # ------------
            if user_api_key_end_user_id is not None:
                total_tokens = _response_total_tokens(response_obj)

                request_count_api_key = MinuteLimit(user_api_key_end_user_id, precise_minute)

                current = await self._limits.read(
                    key=request_count_api_key,
                    empty_is_missing=True,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                ) or {
                    "current_requests": 1,
                    "current_tpm": total_tokens,
                    "current_rpm": 1,
                }

                new_val = {
                    "current_requests": max(current["current_requests"] - released, 0),
                    "current_tpm": current["current_tpm"] + total_tokens,
                    "current_rpm": current["current_rpm"],
                }

                self.print_verbose(f"updated_value in success call: {new_val}, precise_minute: {precise_minute}")
                values_to_update_in_cache.append((request_count_api_key, new_val))

            await self._limits.write_many(
                cache_list=values_to_update_in_cache,
                ttl=60,
                litellm_parent_otel_span=litellm_parent_otel_span,
            )
        except Exception as e:
            self.print_verbose(e)

    @with_service_target("rate_limits")
    async def async_log_failure_event(
        self, kwargs: Mapping[str, object], response_obj: object, start_time: datetime, end_time: datetime
    ) -> None:
        try:
            self.print_verbose("Inside Max Parallel Request Failure Hook")
            litellm_parent_otel_span: Final[Span | None] = get_parent_otel_span_from_kwargs(kwargs=dict(kwargs))
            params: Final = _CALLBACK_DATA.validate_python(kwargs["litellm_params"])
            _metadata: Final = _CALLBACK_DATA.validate_python(params.get("metadata") or {})
            released: Final = _slots_released(kwargs)
            global_max_parallel_requests: Final = _metadata.get("global_max_parallel_requests", None)
            user_api_key: Final = _optional_identity(_metadata.get("user_api_key"))
            self.print_verbose(f"user_api_key: [set={user_api_key is not None}]")
            if user_api_key is None:
                return

            ## decrement call count if call failed
            if CommonProxyErrors.max_parallel_request_limit_reached.value in str(kwargs["exception"]):
                pass  # ignore failed calls due to max limit being reached
            else:
                # ------------
                # Setup values
                # ------------

                if global_max_parallel_requests is not None:
                    # get value from cache
                    _key: Final = "global_max_parallel_requests"
                    (
                        await self.internal_usage_cache.async_get_cache(
                            key=_key,
                            local_only=True,
                            litellm_parent_otel_span=litellm_parent_otel_span,
                        )
                    )
                    # decrement
                    await self.internal_usage_cache.async_increment_cache(
                        key=_key,
                        value=-released,
                        local_only=True,
                        litellm_parent_otel_span=litellm_parent_otel_span,
                    )

                precise_minute: Final = _precise_minute(self._clock())

                request_count_api_key: Final = MinuteLimit(user_api_key, precise_minute)

                # ------------
                # Update usage
                # ------------
                current: Final = await self._limits.read(
                    key=request_count_api_key,
                    empty_is_missing=True,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                ) or {
                    "current_requests": 1,
                    "current_tpm": 0,
                    "current_rpm": 0,
                }

                new_val: Final = {
                    "current_requests": max(current["current_requests"] - released, 0),
                    "current_tpm": current["current_tpm"],
                    "current_rpm": current["current_rpm"],
                }

                self.print_verbose(f"updated_value in failure call: {new_val}")
                await self._limits.write(
                    request_count_api_key,
                    new_val,
                    ttl=60,
                    litellm_parent_otel_span=litellm_parent_otel_span,
                )  # save in cache for up to 1 min.
        except Exception as e:
            verbose_proxy_logger.exception("Inside Parallel Request Limiter: An exception occurred - %s", e)

    @with_service_target("rate_limits")
    async def async_release_max_parallel_requests_on_disconnect(self, user_api_key_dict: UserAPIKeyAuth) -> None:
        """Release the slot ``async_pre_call_hook`` took for a connection whose per-turn callbacks released none."""
        from litellm.proxy.proxy_server import (
            general_settings,  # pyright: ignore[reportUnknownVariableType]  # module-level settings dict is untyped
        )

        now: Final = self._clock()
        await self.async_log_success_event(  # pyright: ignore[reportUnknownMemberType]  # legacy callback signature is untyped
            kwargs={
                "litellm_params": {
                    "metadata": {
                        "user_api_key": LiteLLMProxyRequestSetup.get_logged_api_key(user_api_key_dict),
                        "user_api_key_user_id": user_api_key_dict.user_id,
                        "user_api_key_team_id": user_api_key_dict.team_id,
                        "user_api_key_model_max_budget": user_api_key_dict.model_max_budget,  # pyright: ignore[reportUnknownMemberType]  # UserAPIKeyAuth.model_max_budget is a bare dict
                        "global_max_parallel_requests": general_settings.get(  # pyright: ignore[reportUnknownMemberType]  # module-level settings dict is untyped
                            "global_max_parallel_requests"
                        ),
                    }
                },
                "user": user_api_key_dict.end_user_id,
            },
            response_obj=None,
            start_time=now,
            end_time=now,
        )

    async def get_internal_user_object(
        self,
        user_id: str,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> dict | None:
        """
        Helper to get the 'Internal User Object'

        It uses the `get_user_object` function from `litellm.proxy.auth.auth_checks`

        We need this because the UserApiKeyAuth object does not contain the rpm/tpm limits for a User AND there could be a perf impact by additionally reading the UserTable.
        """
        from litellm._logging import verbose_proxy_logger
        from litellm.proxy.auth.auth_checks import get_user_object
        from litellm.proxy.proxy_server import prisma_client

        try:
            _user_id_rate_limits: Final = await get_user_object(
                user_id=user_id,
                prisma_client=prisma_client,
                user_api_key_cache=self.internal_usage_cache.dual_cache,
                user_id_upsert=False,
                parent_otel_span=user_api_key_dict.parent_otel_span,
                proxy_logging_obj=None,
            )

            if _user_id_rate_limits is None:
                return None

            return _user_id_rate_limits.model_dump()
        except Exception as e:
            verbose_proxy_logger.debug("Parallel Request Limiter: Error getting user object", str(e))
            return None

    @with_service_target("rate_limits")
    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response) -> None:
        """
        Retrieve the key's remaining rate limits.
        """
        api_key: Final = LiteLLMProxyRequestSetup.get_logged_api_key(user_api_key_dict)
        precise_minute: Final = _precise_minute(self._clock())
        request_count_api_key: Final = MinuteLimit(str(api_key), precise_minute)
        current: Final[UsageSnapshot | None] = await self._limits.read(
            key=request_count_api_key,
            litellm_parent_otel_span=user_api_key_dict.parent_otel_span,
        )

        key_remaining_rpm_limit: int | float | None = None
        key_rpm_limit: int | None = None
        key_remaining_tpm_limit: int | float | None = None
        key_tpm_limit: int | None = None
        if current is not None:
            if user_api_key_dict.rpm_limit is not None:
                key_remaining_rpm_limit = user_api_key_dict.rpm_limit - current["current_rpm"]
                key_rpm_limit = user_api_key_dict.rpm_limit
            if user_api_key_dict.tpm_limit is not None:
                key_remaining_tpm_limit = user_api_key_dict.tpm_limit - current["current_tpm"]
                key_tpm_limit = user_api_key_dict.tpm_limit

        _hidden_params = getattr(response, "_hidden_params", None)
        if _hidden_params is not None and (isinstance(_hidden_params, BaseModel) or isinstance(_hidden_params, dict)):
            if isinstance(_hidden_params, BaseModel):
                _hidden_params = _hidden_params.model_dump()

            _additional_headers: Final = _hidden_params.get("additional_headers", {}) or {}

            if key_remaining_rpm_limit is not None:
                _additional_headers["x-ratelimit-remaining-requests"] = key_remaining_rpm_limit
            if key_rpm_limit is not None:
                _additional_headers["x-ratelimit-limit-requests"] = key_rpm_limit
            if key_remaining_tpm_limit is not None:
                _additional_headers["x-ratelimit-remaining-tokens"] = key_remaining_tpm_limit
            if key_tpm_limit is not None:
                _additional_headers["x-ratelimit-limit-tokens"] = key_tpm_limit

            setattr(
                response,
                "_hidden_params",
                {**_hidden_params, "additional_headers": _additional_headers},
            )

            await super().async_post_call_success_hook(data, user_api_key_dict, response)


PROXY_MaxParallelRequestsHandler: Final = _PROXY_MaxParallelRequestsHandler
