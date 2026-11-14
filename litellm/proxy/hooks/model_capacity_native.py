import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, nullcontext
from contextvars import ContextVar
from dataclasses import dataclass, field
from itertools import count
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, TypeAlias
from weakref import WeakKeyDictionary

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError
from redis.exceptions import RedisError
from typing_extensions import TypeIs  # noqa: TID251  # tells the legacy and native dynamic limiters apart

import litellm
from litellm import Router
from litellm._internal_context import service_target
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.exceptions import ServiceUnavailableError
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.derivations import key_identity, mapping, member
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.integrations.custom_logger_v2.rust import (
    RustHandle,
    RustLogger,
    WireAbort,
    WireManifest,
    _namespace,  # pyright: ignore[reportPrivateUsage]  # the generated class needs this port's base
    register_abort_error,
)
from litellm.litellm_core_utils.add_retry_fallback_headers import (
    ensure_response_additional_headers,
    response_has_hidden_params,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError, map_v3_rate_limit_type
from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
    _PROXY_DynamicRateLimitHandlerV3,  # pyright: ignore[reportPrivateUsage]  # the class behind the public alias, usable in a type
    _SettlementLoggingObject,  # pyright: ignore[reportPrivateUsage]  # the handler's own settlement validation error text
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    RATE_LIMIT_UNVERIFIABLE_MESSAGE,
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitDescriptorRateLimitObject,
    RateLimitedModel,
    RateLimitResponse,
    RateLimitUnverifiableError,
    RequestRateLimiterStash,
    ReservationOperation,
    WindowGuardedIncrement,
    borrowing_team_model_limits,
    callback_kwargs_mapping,
    claim_request_stash_for_data,
    get_or_create_request_stash,
    get_request_stash,
    request_stash_for_callback,
)
from litellm.proxy.hooks.priority_class_share import ActiveTeamRegistry
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.proxy.request_auth_context import REQUEST_AUTH
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RATE_LIMITS_TARGET, CounterIncrement, RateLimitCounter, RateLimitScope
from litellm.types.integrations.custom_logger_v2 import KeyIdentity
from litellm.types.integrations.custom_logger_v2 import Outcome as AttemptOutcome

if TYPE_CHECKING:
    from litellm.rust_bridge._native import NativeModelCapacity, NativeModelCapacityState

NATIVE_FLAG: Final = "LITELLM_NATIVE_DYNAMIC_RATE_LIMITER_V3"
UNLICENSED_PRIORITY_MESSAGE: Final = (
    "PREMIUM FEATURE: Reserving tpm/rpm by priority is a premium feature. Please add a 'LITELLM_LICENSE' to your "
    ".env to enable this.\nGet a license: https://docs.litellm.ai/docs/enterprise."
)
_METADATA_BUCKETS: Final = ("metadata", "litellm_metadata")
_TRACES: Final = frozenset({"untraced", "attempt", "key"})
RATE_LIMIT_KIND: Final = "model_capacity_rate_limit"
RERAISE_KIND: Final = "model_capacity_reraise"
_TOKENS: Final = count(1)
_LOG_LEVELS: Final = MappingProxyType({"error": logging.ERROR, "warning": logging.WARNING, "info": logging.INFO})
_TARGETED_HOOKS: Final = frozenset({Hook.pre_call, Hook.post_call_success, Hook.post_call_failure})


def native_enabled() -> bool:
    return os.getenv(NATIVE_FLAG, "").strip().lower() == "true"


@dataclass(slots=True)
class LiveRequest:
    hook: Hook | None = None
    data: Mapping[str, object] | None = None
    auth: UserAPIKeyAuth | None = None
    response: object = None
    kwargs: Mapping[str, object] | None = None
    state: "NativeModelCapacityState | None" = None
    raised: dict[int, BaseException] = field(default_factory=dict)  # mutable-ok: exceptions the seam re-raises
    responses: dict[int, RateLimitResponse] = field(default_factory=dict)  # mutable-ok: limiter replies by id


_LIVE: Final[ContextVar[LiveRequest | None]] = ContextVar("native_model_capacity_live", default=None)


_ADMISSIONS: Final[WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Lock]] = WeakKeyDictionary()


def _admission(limiter: PROXY_MaxParallelRequestsHandler_v3) -> AbstractAsyncContextManager[object]:
    if limiter.rate_limits.has_redis:
        return nullcontext()
    loop: Final = asyncio.get_running_loop()
    lock: Final = _ADMISSIONS.get(loop)
    if lock is not None:
        return lock
    created: Final = asyncio.Lock()
    _ADMISSIONS[loop] = created
    return created


def _live() -> LiveRequest:
    live: Final = _LIVE.get()
    if live is None:
        raise RuntimeError("the model capacity seam was called outside a live request")
    return live


def auth_from_request(data: object) -> UserAPIKeyAuth | None:
    request: Final = as_str_mapping(data)
    if request is None:
        return None
    for bucket_name in _METADATA_BUCKETS:
        bucket = as_str_mapping(request.get(bucket_name))
        auth = None if bucket is None else bucket.get("user_api_key_auth")
        if isinstance(auth, UserAPIKeyAuth):
            return auth
        if as_str_mapping(auth) is not None:
            try:
                return UserAPIKeyAuth.model_validate(auth)
            except ValidationError:
                continue
    return None


def _unheld(value: object) -> object:
    try:
        hash(value)
    except TypeError:
        return value
    return None


def key_gaps(auth: UserAPIKeyAuth | None) -> Mapping[str, object]:
    if auth is None:
        return {}
    return {
        "team_priority": _unheld(member(mapping(auth.team_metadata), "priority")),
        "key_priority": _unheld(member(mapping(auth.metadata), "priority")),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # the key's untyped metadata dict
    }


def _dumped(record: BaseModel | None) -> object:
    return None if record is None else record.model_dump(mode="python")


def key_inputs(auth: UserAPIKeyAuth) -> Mapping[str, object]:
    args: Final = HookArgs(auth=auth)
    return {"identity": _dumped(key_identity(args)), "gaps": key_gaps(auth)}


def _settlement_error(kwargs: Mapping[str, object]) -> str | None:
    try:
        _SettlementLoggingObject.model_validate(kwargs.get("standard_logging_object") or {})
    except ValidationError as error:
        return str(error)
    return None


def _attempt_gaps(raw: object) -> Mapping[str, object]:
    return {"settlement_error": _settlement_error(callback_kwargs_mapping(raw))}


def _project_kwargs(raw: object) -> object:
    live: Final = _LIVE.get()
    match None if live is None else live.hook:
        case Hook.pre_call:
            return key_gaps(REQUEST_AUTH.get())
        case Hook.attempt_ended:
            return _attempt_gaps(raw)
        case _:
            return None


PROJECTIONS: Final[Mapping[Field, Callable[[object], object]]] = {
    Field.deprecated_kwargs: _project_kwargs,
    Field.deprecated_response_obj: lambda _response: None,
}


class _Counter(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    value: str
    kind: Literal["requests", "tokens"]

    def counter(self) -> RateLimitCounter:
        return RateLimitScope(self.key, self.value).counter(self.kind)


class _RateLimit(BaseModel):
    model_config = ConfigDict(frozen=True)

    requests_per_unit: int | None
    tokens_per_unit: int | None
    window_size: int | None

    def rate_limit(self) -> RateLimitDescriptorRateLimitObject:
        return RateLimitDescriptorRateLimitObject(
            requests_per_unit=self.requests_per_unit,
            tokens_per_unit=self.tokens_per_unit,
            window_size=self.window_size,
        )


class _Descriptor(BaseModel):
    model_config = ConfigDict(frozen=True)

    key: str
    value: str
    rate_limit: _RateLimit | None

    def descriptor(self) -> RateLimitDescriptor:
        return RateLimitDescriptor(
            key=self.key, value=self.value, rate_limit=None if self.rate_limit is None else self.rate_limit.rate_limit()
        )


class _Operation(BaseModel):
    model_config = ConfigDict(frozen=True)

    op: Literal["increment", "guarded"]
    counter: _Counter
    amount: int
    ttl: int | None
    expected_window_start: str | None = None
    backend: Literal["redis", "local"] | None = None

    def operation(self) -> ReservationOperation:
        increment: Final = CounterIncrement(counter=self.counter.counter(), amount=self.amount, ttl=self.ttl)
        if self.op == "increment" or self.expected_window_start is None or self.backend is None:
            return increment
        return WindowGuardedIncrement(
            increment=increment, expected_window_start=self.expected_window_start, reservation_backend=self.backend
        )


_ARGS: Final = TypeAdapter(dict[str, object])
_DESCRIPTORS: Final = TypeAdapter(tuple[_Descriptor, ...])
_INCREMENTS: Final = TypeAdapter(list[dict[Literal["requests", "tokens"], int]])
_OPERATIONS: Final = TypeAdapter(tuple[_Operation, ...])
_PAIRS: Final = TypeAdapter(tuple[tuple[str, object], ...])
_TOKEN_LIST: Final = TypeAdapter(tuple[int, ...])


def _descriptors(raw: object) -> tuple[RateLimitDescriptor, ...]:
    return tuple(descriptor.descriptor() for descriptor in _DESCRIPTORS.validate_python(raw))


def _wire_response(live: LiveRequest, response: RateLimitResponse) -> Mapping[str, object]:
    token: Final = next(_TOKENS)
    live.responses[token] = response  # rebind-ok: the reply is kept by id so publish can stash the same object
    return {
        "id": token,
        "overall_code": response["overall_code"],
        "statuses": [
            {
                "code": status["code"],
                "current_limit": status["current_limit"],
                "limit_remaining": status["limit_remaining"],
                "rate_limit_type": status["rate_limit_type"],
                "descriptor_key": status["descriptor_key"],
            }
            for status in response["statuses"]
        ],
        "reservation_windows": [
            {
                "counter": {
                    "key": window[0].scope.descriptor_key,
                    "value": window[0].scope.descriptor_value,
                    "kind": window[0].kind,
                },
                "start": window[1],
                "backend": window[2],
            }
            for window in response.get("reservation_windows", frozenset())
        ],
    }


def _raised(error: Exception) -> Mapping[str, object]:
    live: Final = _LIVE.get()
    token: Final = next(_TOKENS)
    if live is not None:
        live.raised[token] = error
    kind: Final = (
        "http"
        if isinstance(error, HTTPException)
        else "registry_unavailable"
        if isinstance(error, RedisError | RedisCircuitBreakerOpenError | ValidationError)
        else "other"
    )
    return {"raised": {"token": token, "kind": kind, "message": str(error)}}


def _seam(
    method: Callable[["CapacityDelegate", Mapping[str, object]], Awaitable[object]],
) -> Callable[["CapacityDelegate", object], Awaitable[Mapping[str, object]]]:
    async def call(self: "CapacityDelegate", args: object) -> Mapping[str, object]:
        try:
            return {"ok": await method(self, _ARGS.validate_python(args))}
        except Exception as error:  # noqa: BLE001  # Rust decides which errors fail open and which re-raise
            return _raised(error)

    return call


class CapacityDelegate:
    def __init__(
        self,
        internal_usage_cache: DualCache,
        v3_limiter: PROXY_MaxParallelRequestsHandler_v3,
    ) -> None:
        self.v3_limiter: Final = v3_limiter
        self.active_teams: Final = ActiveTeamRegistry(internal_usage_cache)
        self.llm_router: Router | None = None

    def _router(self) -> Router:
        if self.llm_router is None:
            raise AttributeError("'CapacityDelegate' object has no attribute 'llm_router'")
        return self.llm_router

    def _auth(self) -> UserAPIKeyAuth:
        auth: Final = _live().auth
        if auth is None:
            raise ValueError("the request carries no key auth")
        return auth

    @_seam
    async def settings(self, _args: Mapping[str, object]) -> object:
        from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
            _get_priority_settings,  # pyright: ignore[reportPrivateUsage]  # one settings source
        )

        settings: Final = _get_priority_settings()
        reservation: Final = litellm.priority_reservation
        return {
            "priority_reservation": None
            if reservation is None
            else [[name, value] for name, value in reservation.items()],
            "licensed": os.getenv("LITELLM_LICENSE", None) is not None,
            "default_priority": settings.default_priority,
            "saturation_threshold": settings.saturation_threshold,
            "class_split": settings.priority_class_share_splits_equally_among_active_teams,
            "saturation_check_cache_ttl": settings.saturation_check_cache_ttl,
            "window_size": self.v3_limiter.window_size,
            "rate_limit_type": self.v3_limiter.get_rate_limit_type(),
        }

    @_seam
    async def group(self, args: Mapping[str, object]) -> object:
        group: Final = self._router().registry().group(str(args["model"]))
        if group is None:
            return None
        return {
            "tpm": group.tpm,
            "rpm": group.rpm,
            "input_cost_per_token": group.input_cost_per_token,
            "output_cost_per_token": group.output_cost_per_token,
        }

    @_seam
    async def read_counter(self, args: Mapping[str, object]) -> object:
        counter: Final = _Counter.model_validate(args["counter"]).counter()
        value: Final = await self.v3_limiter.rate_limits.get_from_redis(counter, backfill_ttl=args["backfill_ttl"])  # pyright: ignore[reportArgumentType]  # the settings value as the handler passes it
        return None if isinstance(value, dict) else value

    @_seam
    async def read_local_counter(self, args: Mapping[str, object]) -> object:
        counter: Final = _Counter.model_validate(args["counter"]).counter()
        value: Final = self.v3_limiter.rate_limits.local((counter,))[0]
        return None if isinstance(value, dict) else value

    def _span(self) -> object:
        auth: Final = _live().auth
        return None if auth is None else auth.parent_otel_span

    @_seam
    async def check_and_increment(self, args: Mapping[str, object]) -> object:
        response: Final = await self.v3_limiter.atomic_check_and_increment_by_n(
            descriptors=list(_descriptors(args["descriptors"])),
            increments=_INCREMENTS.validate_python(args["increments"]),
            parent_otel_span=self._span(),  # pyright: ignore[reportArgumentType]  # the key's span as the handler passes it
        )
        return _wire_response(_live(), response)

    @_seam
    async def should_rate_limit(self, args: Mapping[str, object]) -> object:
        response: Final = await self.v3_limiter.should_rate_limit(
            _descriptors(args["descriptors"]),
            parent_otel_span=self._span(),  # pyright: ignore[reportArgumentType]  # the key's span as the handler passes it
            read_only=args["read_only"] is True,
            skip_parallel_check=args["skip_parallel_check"] is True,
        )
        return _wire_response(_live(), response)

    @_seam
    async def apply(self, args: Mapping[str, object]) -> object:
        from litellm.litellm_core_utils.core_helpers import get_parent_otel_span_from_kwargs

        operations: Final = tuple(
            operation.operation() for operation in _OPERATIONS.validate_python(args["operations"])
        )
        live: Final = _LIVE.get()
        trace: Final = args["trace"] if args["trace"] in _TRACES else "untraced"
        if trace == "attempt" and live is not None and live.kwargs is not None:
            await self.v3_limiter.async_increment_reservation_aware_tokens(
                pipeline_operations=operations,
                parent_otel_span=get_parent_otel_span_from_kwargs(dict(live.kwargs)),
            )
            return None
        if trace == "key" and live is not None and live.auth is not None:
            await self.v3_limiter.async_increment_reservation_aware_tokens(
                pipeline_operations=operations, parent_otel_span=live.auth.parent_otel_span
            )
            return None
        await self.v3_limiter.async_increment_reservation_aware_tokens(pipeline_operations=operations)
        return None

    @_seam
    async def borrowing_limits(self, args: Mapping[str, object]) -> object:
        model: Final = str(args["model"])
        rate_limited: Final = self.v3_limiter.rate_limited_model(model) or RateLimitedModel(
            requested=model, group=model
        )
        limits: Final = borrowing_team_model_limits(self._auth(), rate_limited)
        if limits.tpm is None and limits.rpm is None:
            return None
        return {
            "group": rate_limited.group,
            "tpm": limits.tpm,
            "rpm": limits.rpm,
            "max_borrow_multiple": limits.max_borrow_multiple,
        }

    @_seam
    async def model_group(self, args: Mapping[str, object]) -> object:
        model: Final = str(args["model"])
        return (self.v3_limiter.rate_limited_model(model) or RateLimitedModel(requested=model, group=model)).group

    @_seam
    async def estimate_tokens(self, args: Mapping[str, object]) -> object:
        live: Final = _live()
        data: Final = live.data
        if not data:
            return {"kind": "no_body"}
        model: Final = str(args["model"])
        tpm_share: Final = int(str(args["tpm_share"]))
        call_type: Final = args["call_type"] if isinstance(args["call_type"], str) else None
        as_dispatched: Final = dict(data, model=model)
        if self.v3_limiter.get_rate_limit_type() != "input":
            self.v3_limiter.bound_output_for_limit(as_dispatched, tpm_share, call_type)
        try:
            estimate: Final = await offload_token_count(self.v3_limiter.estimate_request_tokens)(
                data=as_dispatched,
                model=model,
                min_configured_tpm_limit=tpm_share,
                call_type=call_type,
                team_id=None if live.auth is None else live.auth.team_id,
                rate_limit_type=self.v3_limiter.get_rate_limit_type(),
            )
        except Exception as error:  # noqa: BLE001  # an unreadable prompt still holds the minimum so admission runs
            from litellm._logging import verbose_proxy_logger

            verbose_proxy_logger.warning(
                "Dynamic rate limiter: token estimate failed for %s, holding one token: %s", model, error
            )
            return {"kind": "failed"}
        return {"kind": "tokens", "tokens": estimate}

    @_seam
    async def cap_output(self, args: Mapping[str, object]) -> object:
        data: Final = _live().data
        call_type: Final = args["call_type"] if isinstance(args["call_type"], str) else None
        self.v3_limiter.bound_output_for_limit(data, int(str(args["tpm_share"])), call_type)
        return None

    @_seam
    async def publish(self, args: Mapping[str, object]) -> object:
        live: Final = _live()
        responses: Final = tuple(live.responses[token] for token in _TOKEN_LIST.validate_python(args["responses"]))
        get_or_create_request_stash().rate_limit_response = (
            responses[0]
            if len(responses) == 1
            else RateLimitResponse(
                overall_code=responses[0]["overall_code"],
                statuses=responses[0]["statuses"] + responses[1]["statuses"],
            )
        )
        return None

    @_seam
    async def stash_call_type(self, _args: Mapping[str, object]) -> object:
        return get_or_create_request_stash().call_type

    @_seam
    async def post_call_success(self, args: Mapping[str, object]) -> object:
        live: Final = _live()
        await self.v3_limiter.async_post_call_success_hook(  # pyright: ignore[reportUnknownMemberType]  # the legacy hook signature
            data=dict(live.data or {}) if not isinstance(live.data, dict) else live.data,
            user_api_key_dict=self._auth(),
            response=live.response,  # pyright: ignore[reportArgumentType]  # the live response the proxy handed this hook
        )
        if not response_has_hidden_params(live.response):
            return None
        additional_headers: Final = ensure_response_additional_headers(live.response)
        for name, value in _PAIRS.validate_python(args["headers"]):
            additional_headers[name] = value  # pyright: ignore[reportArgumentType]  # the priority value as the handler set it
        return None

    @_seam
    async def touch(self, args: Mapping[str, object]) -> object:
        members: Final = await self.active_teams.touch(
            str(args["model"]),
            str(args["member"]),
            args["priority_class"],  # pyright: ignore[reportArgumentType]  # the class as the handler passes it, a reserved priority of any type
            int(str(args["window"])),
        )
        return [[member, priority_class] for member, priority_class in members.items()]

    @_seam
    async def metrics_enabled(self, _args: Mapping[str, object]) -> object:
        from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
            _prometheus_loggers,  # pyright: ignore[reportPrivateUsage]  # one metrics source
        )

        return bool(_prometheus_loggers())

    @_seam
    async def throttled(self, args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger
        from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
            _prometheus_loggers,  # pyright: ignore[reportPrivateUsage]  # one metrics source
        )

        for logger in _prometheus_loggers():
            try:
                logger.increment_dynamic_limiter_throttled(
                    model_group=str(args["model_group"]),
                    team=str(args["team"]),
                    team_alias=args["team_alias"] if isinstance(args["team_alias"], str) else None,
                    priority=args["priority"],  # pyright: ignore[reportArgumentType]  # the priority header value as the handler passes it
                    rate_limit_type=str(args["rate_limit_type"]),
                    reason=str(args["reason"]),
                )
            except Exception as metrics_error:  # noqa: BLE001  # a failing metrics sink must not change the 429
                verbose_proxy_logger.warning("Dynamic rate limiter throttle metric failed: %s", metrics_error)
        return None

    @_seam
    async def log(self, args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.log(_LOG_LEVELS[str(args["level"])], str(args["message"]))
        return None

    @_seam
    async def unlicensed_priority(self, _args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.error(UNLICENSED_PRIORITY_MESSAGE)
        return None

    @_seam
    async def set_share(self, args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger
        from litellm.proxy.hooks.dynamic_rate_limiter_v3 import (
            _prometheus_loggers,  # pyright: ignore[reportPrivateUsage]  # one metrics source
        )

        try:
            for logger in _prometheus_loggers():
                logger.set_dynamic_limiter_share(
                    model_group=str(args["model_group"]),
                    team=str(args["team"]),
                    team_alias=args["team_alias"] if isinstance(args["team_alias"], str) else None,
                    priority=args["priority"],  # pyright: ignore[reportArgumentType]  # the priority value as the handler passes it
                    share_tokens=int(str(args["share_tokens"])),
                    borrowed_tokens=int(str(args["borrowed_tokens"])),
                )
        except Exception as metrics_error:  # noqa: BLE001  # a failing metrics sink must not change settlement
            verbose_proxy_logger.warning("Dynamic rate limiter share metrics failed: %s", metrics_error)
        return None


def _rate_limit_error(wire: WireAbort) -> Exception:
    return _proxy_rate_limit_error((None if wire.error is None else as_str_mapping(wire.error.fields)) or {})


def _proxy_rate_limit_error(fields: Mapping[str, object]) -> ProxyRateLimitError:
    raw_type: Final = fields.get("rate_limit_type")
    model: Final = fields.get("model")
    resolved_model, llm_provider = resolve_llm_provider_for_rate_limit(model if isinstance(model, str) else None)
    descriptor_key: Final = fields.get("descriptor_key")
    return ProxyRateLimitError(
        detail=dict(_PAIRS.validate_python(fields.get("detail") or ())),
        headers=dict(_PAIRS.validate_python(fields.get("headers") or ())),
        rate_limit_type=map_v3_rate_limit_type(raw_type if isinstance(raw_type, str) else None),
        model=resolved_model,
        llm_provider=llm_provider,
        descriptor_key=descriptor_key if isinstance(descriptor_key, str) else None,
    )


def _reraise(wire: WireAbort) -> Exception:
    live: Final = _LIVE.get()
    token: Final = ((None if wire.error is None else as_str_mapping(wire.error.fields)) or {}).get("token")
    raised: Final = None if live is None or not isinstance(token, int) else live.raised.pop(token, None)
    if isinstance(raised, Exception):
        return raised
    return HTTPException(status_code=wire.status, detail={"error": wire.message}, headers=dict(wire.headers))


def _native_answer(answer: Mapping[str, object], live: LiveRequest) -> Mapping[str, object]:
    raised_token: Final = answer.get("raised")
    if isinstance(raised_token, int) and raised_token in live.raised:
        raise live.raised[raised_token]
    if "type_error" in answer:
        raise TypeError(str(answer["type_error"]))
    if "failed" in answer:
        raise RuntimeError(str(answer["failed"]))
    return answer


@dataclass(frozen=True, slots=True)
class _StashedNative:
    capacity: "NativeModelCapacity"
    callback: RustHandle

    def manifest(self) -> str:
        return self.callback.manifest()

    def request_state(self) -> object:
        return None

    def handle(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, _state: object, /
    ) -> Awaitable[object]:
        live: Final = _LIVE.get()
        return self.capacity.handle(hook, fields, call, None if live is None else live.state)


def _request_stash(hook: Hook, data: Mapping[str, object] | None) -> RequestRateLimiterStash | None:
    if hook is Hook.post_call_failure:
        return get_request_stash()
    if data is None:
        return None
    match hook:
        case Hook.pre_call:
            return claim_request_stash_for_data(data) if "model" in data else None
        case Hook.attempt_ended:
            return request_stash_for_callback(callback_kwargs_mapping(data))
        case _:
            return None


def _targeted(hook: Hook, view: object) -> bool:
    if hook is Hook.attempt_ended:
        return getattr(view, Field.outcome.value, None) == AttemptOutcome.success
    return hook in _TARGETED_HOOKS


class NativeDynamicRateLimiter(RustLogger):
    delegate: CapacityDelegate
    capacity: "NativeModelCapacity"

    @property
    def v3_limiter(self) -> PROXY_MaxParallelRequestsHandler_v3:
        return self.delegate.v3_limiter

    def update_variables(self, llm_router: Router) -> None:
        self.delegate.llm_router = llm_router

    def _state(self, stash: RequestRateLimiterStash) -> "NativeModelCapacityState":
        return stash.native_capacity(self.capacity.request_state)

    def _state_for(self, stash: RequestRateLimiterStash | None) -> "NativeModelCapacityState | None":
        return None if stash is None else self._state(stash)

    async def run_rust(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        data: Final = as_str_mapping(getattr(view, Field.deprecated_kwargs.value, None))
        live: Final = LiveRequest(
            hook=hook,
            data=data if hook is not Hook.attempt_ended else None,
            auth=(REQUEST_AUTH.get() or auth_from_request(data)) if hook is not Hook.attempt_ended else None,
            response=getattr(view, Field.deprecated_response_obj.value, None),
            kwargs=data if hook is Hook.attempt_ended else None,
            state=self._state_for(_request_stash(hook, data)),
        )
        token: Final = _LIVE.set(live)
        try:
            with service_target(RATE_LIMITS_TARGET) if _targeted(hook, view) else nullcontext():
                if hook is Hook.post_call_success:
                    identity: Final = getattr(view, Field.key_identity.value, None)
                    headers: Final = self.capacity.post_call_headers(
                        {
                            "identity": _dumped(identity if isinstance(identity, KeyIdentity) else None),
                            "gaps": key_gaps(REQUEST_AUTH.get()),
                        }
                    )
                    answer: Final = await self.delegate.post_call_success({"headers": headers})
                    raised: Final = as_str_mapping(answer.get("raised"))
                    raised_token: Final = None if raised is None else raised.get("token")
                    error: Final = None if not isinstance(raised_token, int) else live.raised.pop(raised_token, None)
                    if error is not None:
                        from litellm._logging import verbose_proxy_logger

                        verbose_proxy_logger.error(
                            "Error in dynamic rate limiter v3 post-call hook: %s", error, exc_info=error
                        )
                    return None
                if hook is not Hook.pre_call:
                    return await super().run_rust(hook, reads, view, ctx)
                async with _admission(self.v3_limiter):
                    return await super().run_rust(hook, reads, view, ctx)
        finally:
            _LIVE.reset(token)

    async def async_post_call_failure_hook(
        self,
        request_data: Mapping[str, object],
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
    ) -> None:
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            await self.run_rust(
                Hook.post_call_failure,
                frozenset(),
                None,
                CallContext(hook=Hook.post_call_failure, call=None, state=None),
            )
        finally:
            REQUEST_AUTH.reset(auth_token)

    async def async_fallback_target_allowed(
        self, user_api_key_dict: UserAPIKeyAuth, model: str, request_data: Mapping[str, object], *, commit: bool = True
    ) -> bool:
        state: Final = self._state(get_or_create_request_stash())
        live: Final = LiveRequest(data={**request_data, "model": model}, auth=user_api_key_dict, state=state)
        token: Final = _LIVE.set(live)
        try:
            async with _admission(self.v3_limiter):
                answer: Final = await self.capacity.fallback_target_allowed(
                    state, key_inputs(user_api_key_dict), model, commit
                )
        finally:
            _LIVE.reset(token)
        return _native_answer(answer, live).get("allowed") is True

    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None:
        if requested_model is None:
            return
        state: Final = self._state_for(get_request_stash())
        live: Final = LiveRequest(auth=user_api_key_dict, state=state)
        token: Final = _LIVE.set(live)
        try:
            async with _admission(self.v3_limiter):
                answer: Final = await self.capacity.enforce_frame(state, key_inputs(user_api_key_dict), requested_model)
        finally:
            _LIVE.reset(token)
        raised_token: Final = answer.get("raised")
        if answer.get("unverifiable") is True and isinstance(raised_token, int):
            unverifiable: Final = live.raised.get(raised_token)
            if isinstance(unverifiable, RateLimitUnverifiableError):
                raise ServiceUnavailableError(
                    message=RATE_LIMIT_UNVERIFIABLE_MESSAGE, llm_provider="litellm_proxy", model=requested_model
                ) from unverifiable
        refused: Final = as_str_mapping(_native_answer(answer, live).get("refused"))
        if refused is not None:
            raise _proxy_rate_limit_error(refused)


def build_native_dynamic_rate_limiter(
    internal_usage_cache: DualCache, time_provider: Callable[[], object] | None = None
) -> NativeDynamicRateLimiter:
    from litellm.rust_bridge._native import NativeModelCapacity

    register_abort_error(RATE_LIMIT_KIND, _rate_limit_error)
    register_abort_error(RERAISE_KIND, _reraise)
    v3_limiter: Final = PROXY_MaxParallelRequestsHandler_v3(
        InternalUsageCache(dual_cache=internal_usage_cache),
        time_provider=time_provider,  # pyright: ignore[reportArgumentType]  # the handler's clock
    )
    delegate: Final = CapacityDelegate(internal_usage_cache, v3_limiter)
    capacity: Final = NativeModelCapacity(delegate)
    native: Final = _StashedNative(capacity, capacity.callback())
    manifest: Final = WireManifest.model_validate_json(native.manifest())
    cls: Final = type(manifest.selection.name, (NativeDynamicRateLimiter,), _namespace(native, manifest))
    logger: Final = cls(native, PROJECTIONS)
    logger.delegate = delegate
    logger.capacity = capacity
    return logger


@dataclass(slots=True)
class _Registered:
    logger: NativeDynamicRateLimiter | None = None
    enabled: bool | None = None


_REGISTERED: Final = _Registered()


def registered_native_dynamic_rate_limiter() -> NativeDynamicRateLimiter | None:
    return _REGISTERED.logger


def native_dynamic_rate_limiter(internal_usage_cache: DualCache, llm_router: object) -> NativeDynamicRateLimiter | None:
    if _REGISTERED.enabled is None:
        _REGISTERED.enabled = native_enabled()
    if not _REGISTERED.enabled:
        return None
    if _REGISTERED.logger is None:
        _REGISTERED.logger = build_native_dynamic_rate_limiter(internal_usage_cache)
    if isinstance(llm_router, Router):
        _REGISTERED.logger.update_variables(llm_router)
    return _REGISTERED.logger


DynamicRateLimiter: TypeAlias = _PROXY_DynamicRateLimitHandlerV3 | NativeDynamicRateLimiter


def is_dynamic_rate_limiter(callback: object) -> TypeIs[DynamicRateLimiter]:  # guard-ok: one isinstance on both classes
    return isinstance(callback, _PROXY_DynamicRateLimitHandlerV3 | NativeDynamicRateLimiter)


__all__ = [
    "NATIVE_FLAG",
    "CapacityDelegate",
    "NativeDynamicRateLimiter",
    "build_native_dynamic_rate_limiter",
    "is_dynamic_rate_limiter",
    "native_dynamic_rate_limiter",
    "native_enabled",
    "registered_native_dynamic_rate_limiter",
]
