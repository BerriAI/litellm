import asyncio
import logging
import os
from collections.abc import Awaitable, Callable, Coroutine, Mapping, Sequence
from contextlib import nullcontext
from contextvars import Context, ContextVar, copy_context
from dataclasses import dataclass, field
from datetime import datetime
from itertools import count
from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, cast  # noqa: TID251  # a future a host coroutine yields is untyped

from fastapi import HTTPException
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

import litellm
from litellm import Router
from litellm._internal_context import reporting_nested_tool_call_failure, service_target
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
from litellm.litellm_core_utils.core_helpers import as_str_mapping, attribute_or_none
from litellm.litellm_core_utils.token_counter import offload_token_count
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError, map_v3_rate_limit_type
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    RATE_LIMIT_UNVERIFIABLE_MESSAGE,
    BorrowerKind,
    LimiterPostCall,
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitedModel,
    RateLimitUnverifiableError,
    RememberedTokenEstimate,
    RequestRateLimiterStash,
    callback_kwargs_mapping,
    cap_output_for_limit,
    claim_request_stash_for_data,
    dispatched_token_estimate,
    get_or_create_request_stash,
    get_request_stash,
    limiter_post_call_success,
    model_limit_borrowers,
    native_rate_limit_store,
    rate_limit_window_size,
    request_stash_for_callback,
    resolve_rate_limited_model,
    token_rate_limit_type,
    tpm_token_reservation_enabled,
)
from litellm.proxy.hooks.parallel_request_limiter_v3_native import (
    NativeBatchEnqueuedTokenStore,
    native_engine,
    post_call_request,
    pre_call_request,
)
from litellm.proxy.hooks.parallel_request_limiter_v3_native import policy as limiter_policy
from litellm.proxy.hooks.priority_class_share import ActiveTeamRegistry
from litellm.proxy.hooks.rate_limiter_utils import resolve_llm_provider_for_rate_limit
from litellm.proxy.request_auth_context import REQUEST_AUTH
from litellm.proxy.utils import InternalUsageCache
from litellm.redis.stores.rate_limits import RATE_LIMITS_TARGET, RateLimitStore
from litellm.types.integrations.custom_logger_v2 import KeyIdentity
from litellm.types.integrations.custom_logger_v2 import Outcome as AttemptOutcome
from litellm.types.utils import PriorityReservationSettings, Usage

if TYPE_CHECKING:
    from litellm.integrations.prometheus import PrometheusLogger
    from litellm.rust_bridge._native import NativeModelCapacity, NativeModelCapacityState, NativeRequestRedis

UNLICENSED_PRIORITY_MESSAGE: Final = (
    "PREMIUM FEATURE: Reserving tpm/rpm by priority is a premium feature. Please add a 'LITELLM_LICENSE' to your "
    ".env to enable this.\nGet a license: https://docs.litellm.ai/docs/enterprise."
)
_METADATA_BUCKETS: Final = ("metadata", "litellm_metadata")
_TRACES: Final = frozenset({"untraced", "attempt", "key"})
RATE_LIMIT_KIND: Final = "model_capacity_rate_limit"
RERAISE_KIND: Final = "model_capacity_reraise"
_TOKENS: Final = count(1)
MediaModelResolver = Callable[[str, str | None], tuple[str, ...]]
_LOG_LEVELS: Final = MappingProxyType({"error": logging.ERROR, "warning": logging.WARNING, "info": logging.INFO})
_TARGETED_HOOKS: Final = frozenset({Hook.pre_call, Hook.post_call_success, Hook.post_call_failure})
_SETTLING_HOOKS: Final = frozenset({Hook.post_call_failure.value, Hook.attempt_ended.value})


@dataclass(slots=True)
class LiveRequest:
    hook: Hook | None = None
    data: Mapping[str, object] | None = None
    auth: UserAPIKeyAuth | None = None
    stash: RequestRateLimiterStash | None = None
    response: object = None
    kwargs: Mapping[str, object] | None = None
    state: "NativeModelCapacityState | None" = None
    raised: dict[int, BaseException] = field(default_factory=dict)  # mutable-ok: exceptions the seam re-raises
    suspended: dict[int, "_Suspended"] = field(default_factory=dict)  # mutable-ok: host calls Rust resumes


_LIVE: Final[ContextVar[LiveRequest | None]] = ContextVar("native_model_capacity_live", default=None)


@dataclass(frozen=True, slots=True)
class _Suspended:
    coroutine: Coroutine[object, object, object]
    yielded: object
    context: Context


def _step(
    coroutine: Coroutine[object, object, object], context: Context, value: object, error: BaseException | None
) -> tuple[bool, object]:
    try:
        yielded: Final = context.run(coroutine.send, value) if error is None else context.run(coroutine.throw, error)
    except StopIteration as stop:
        finished: object = stop.value  # pyright: ignore[reportAny]  # the value the host coroutine returned
        return True, finished
    return False, yielded


async def _resumed(out: object) -> object:
    if not isinstance(out, asyncio.Future):
        await asyncio.sleep(0)
        return None
    future: Final = cast("asyncio.Future[object]", out)  # cast-ok: the future a host coroutine yields
    future._asyncio_future_blocking = False
    return await future


async def _finish(suspended: _Suspended) -> object:
    out: object = suspended.yielded
    while True:
        value: object = None
        error: BaseException | None = None
        try:
            value = await _resumed(out)
        except BaseException as caught:  # noqa: BLE001  # the suspended call sees what its await raised, as under a task
            error = caught
        finished, out = _step(suspended.coroutine, suspended.context, value, error)
        if finished:
            return out


def _loop_running() -> bool:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return False
    return True


def _members(members: object) -> list[list[object]]:
    found: Final = as_str_mapping(members)
    if found is None:
        raise TypeError(f"the active team registry answered {type(members).__name__}")
    return [[member, priority_class] for member, priority_class in found.items()]


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
    hierarchy: Final = auth.customer_hierarchy
    return {
        "team_priority": _unheld(member(mapping(auth.team_metadata), "priority")),
        "key_priority": _unheld(member(mapping(auth.metadata), "priority")),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # the key's untyped metadata dict
        "customer_org_alias": None if hierarchy is None else hierarchy.customer_org_alias,
    }


def _dumped(record: BaseModel | None) -> object:
    return None if record is None else record.model_dump(mode="python")


def key_inputs(auth: UserAPIKeyAuth) -> Mapping[str, object]:
    args: Final = HookArgs(auth=auth)
    return {"identity": _dumped(key_identity(args)), "gaps": key_gaps(auth)}


class _SettlementMetadata(BaseModel):
    model_config = ConfigDict(frozen=True)

    user_api_key_auth_metadata: Mapping[str, object] | None = None
    user_api_key_team_id: str | None = None
    user_api_key_team_alias: str | None = None
    user_api_key_hash: str | None = None
    user_api_key_alias: str | None = None
    user_api_key_customer_org_id: str | None = None
    user_api_key_customer_org_alias: str | None = None


class _SettlementLoggingObject(BaseModel):
    model_config = ConfigDict(frozen=True)

    metadata: _SettlementMetadata | None = None


def _prometheus_loggers() -> "Sequence[PrometheusLogger]":
    from litellm.integrations.prometheus import PrometheusLogger

    return tuple(
        logger
        for logger in litellm.logging_callback_manager.get_custom_loggers_for_type(PrometheusLogger)
        if isinstance(logger, PrometheusLogger)
    )


def _attempt_gaps(raw: object) -> Mapping[str, object]:
    try:
        logged: Final = _SettlementLoggingObject.model_validate(
            callback_kwargs_mapping(raw).get("standard_logging_object") or {}
        )
    except ValidationError as error:
        return {"settlement_error": str(error)}
    metadata: Final = logged.metadata or _SettlementMetadata()
    return {
        "customer_org_id": metadata.user_api_key_customer_org_id,
        "customer_org_alias": metadata.user_api_key_customer_org_alias,
    }


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


_ARGS: Final = TypeAdapter(dict[str, object])
_HOLDER_KINDS: Final[TypeAdapter[BorrowerKind]] = TypeAdapter(BorrowerKind)
_PAIRS: Final = TypeAdapter(tuple[tuple[str, object], ...])


def _registry_unavailable(error: Exception) -> bool:
    from litellm.rust_bridge._native import NativeRedisStoreError

    if isinstance(error, RedisCircuitBreakerOpenError | ValidationError | NativeRedisStoreError):
        return True
    try:
        from redis.exceptions import RedisError
    except ImportError:
        return False
    return isinstance(error, RedisError)


def _raised(error: Exception) -> Mapping[str, object]:
    live: Final = _LIVE.get()
    token: Final = next(_TOKENS)
    if live is not None:
        live.raised[token] = error
    kind: Final = (
        "http"
        if isinstance(error, HTTPException)
        else "registry_unavailable"
        if _registry_unavailable(error)
        else "other"
    )
    return {"raised": {"token": token, "kind": kind, "message": str(error)}}


def _host_call(
    method: Callable[["CapacityHost", Mapping[str, object]], Awaitable[object]],
) -> Callable[["CapacityHost", object], Awaitable[Mapping[str, object]]]:
    async def call(self: "CapacityHost", args: object) -> Mapping[str, object]:
        try:
            return {"ok": await method(self, _ARGS.validate_python(args))}
        except Exception as error:  # noqa: BLE001  # Rust decides which errors fail open and which re-raise
            return _raised(error)

    return call


def _host_call_sync(
    method: Callable[["CapacityHost", Mapping[str, object]], object],
) -> Callable[["CapacityHost", object], Mapping[str, object]]:
    def call(self: "CapacityHost", args: object) -> Mapping[str, object]:
        try:
            return {"ok": method(self, _ARGS.validate_python(args))}
        except Exception as error:  # noqa: BLE001  # Rust decides which errors fail open and which re-raise
            return _raised(error)

    return call


class CapacityHost:
    def __init__(
        self,
        internal_usage_cache: DualCache,
        post_call: LimiterPostCall,
        window_size: int,
        media_model_resolver: MediaModelResolver | None = None,
    ) -> None:
        self.post_call: Final = post_call
        self.window_size: Final = window_size
        self.media_model_resolver: Final = media_model_resolver
        self.usage_cache: Final = internal_usage_cache
        self.active_teams: Final = ActiveTeamRegistry(internal_usage_cache)
        self.deferred_touches: Final[dict[int, tuple[str, str, object, int]]] = {}
        self.llm_router: Router | None = None

    def _router(self) -> Router:
        if self.llm_router is None:
            raise AttributeError("'CapacityHost' object has no attribute 'llm_router'")
        return self.llm_router

    def _auth(self) -> UserAPIKeyAuth:
        auth: Final = _live().auth
        if auth is None:
            raise ValueError("the request carries no key auth")
        return auth

    def facts(self, model: str | None, *, call_type: bool = False) -> Mapping[str, object]:
        return {
            "settings": self.settings({}),
            **(
                {"group": self.group({"model": model}), "borrowing": self.borrowing_limits({"model": model})}
                if model is not None
                else {}
            ),
            **({"call_type": self.stash_call_type({})} if call_type else {}),
        }

    @_host_call_sync
    def settings(self, _args: Mapping[str, object]) -> object:
        settings: Final = litellm.priority_reservation_settings or PriorityReservationSettings()
        reservation: Final = litellm.priority_reservation
        return {
            "priority_reservation": None
            if reservation is None
            else [[name, value] for name, value in reservation.items()],
            "licensed": os.getenv("LITELLM_LICENSE", None) is not None,
            "default_priority": settings.default_priority,
            "saturation_threshold": settings.saturation_threshold,
            "class_split": list(settings.priority_class_share_config.split_equally_among_active),
            "saturation_check_cache_ttl": settings.saturation_check_cache_ttl,
            "window_size": self.window_size,
            "rate_limit_type": token_rate_limit_type(),
            "policy": limiter_policy(),
        }

    @_host_call_sync
    def group(self, args: Mapping[str, object]) -> object:
        group: Final = self._router().registry().group(str(args["model"]))
        if group is None:
            return None
        return {
            "tpm": group.tpm,
            "rpm": group.rpm,
            "input_cost_per_token": group.input_cost_per_token,
            "output_cost_per_token": group.output_cost_per_token,
        }

    @_host_call_sync
    def borrowing_limits(self, args: Mapping[str, object]) -> object:
        model: Final = str(args["model"])
        rate_limited: Final = resolve_rate_limited_model(model) or RateLimitedModel(requested=model, group=model)
        return [
            {
                "kind": borrower.kind,
                "entity_id": borrower.entity_id,
                "alias": borrower.alias,
                "group": rate_limited.group,
                "tpm": borrower.limits.tpm,
                "rpm": borrower.limits.rpm,
                "max_borrow_multiple": borrower.limits.max_borrow_multiple,
            }
            for borrower in model_limit_borrowers(self._auth(), rate_limited)
        ]

    @_host_call
    async def model_group(self, args: Mapping[str, object]) -> object:
        model: Final = str(args["model"])
        return (resolve_rate_limited_model(model) or RateLimitedModel(requested=model, group=model)).group

    @_host_call
    async def estimate_tokens(self, args: Mapping[str, object]) -> object:
        live: Final = _live()
        data: Final = live.data
        if not data:
            return {"kind": "no_body"}
        model: Final = args["model"]
        tpm_share: Final = int(str(args["tpm_share"]))
        call_type: Final = args["call_type"] if isinstance(args["call_type"], str) else None
        stash: Final = live.stash
        team_id: Final = (
            live.auth.team_id if live.auth is not None else None if stash is None else stash.request_team_id
        )
        as_dispatched: Final = dict(data) if not isinstance(model, str) else dict(data, model=model)
        media_model_resolver: Final = (
            {} if self.media_model_resolver is None else {"media_model_resolver": self.media_model_resolver}
        )
        rate_limit_type: Final = token_rate_limit_type()
        if live.hook is not Hook.attempt_ended and rate_limit_type != "input":
            cap_output_for_limit(as_dispatched, tpm_share, call_type)
        dispatched_model: Final = as_dispatched.get("model")
        remembered: Final = None if stash is None else stash.token_estimate
        try:
            estimate: Final = (
                remembered.estimate
                if live.hook is Hook.attempt_ended and remembered is not None and remembered.model == dispatched_model
                else await offload_token_count(dispatched_token_estimate)(
                    data=as_dispatched,
                    min_configured_tpm_limit=tpm_share,
                    call_type=call_type,
                    team_id=team_id,
                    **media_model_resolver,
                )
            )
        except Exception as error:  # noqa: BLE001  # an unreadable prompt still holds the minimum so admission runs
            from litellm._logging import verbose_proxy_logger

            verbose_proxy_logger.warning(
                "Dynamic rate limiter: token estimate failed for %s, holding one token: %s", model, error
            )
            return {"kind": "failed"}
        if stash is not None and live.hook is not Hook.attempt_ended:
            stash.token_estimate = RememberedTokenEstimate(
                dispatched_model if isinstance(dispatched_model, str) else None, estimate
            )
        if live.hook is Hook.attempt_ended and estimate.unpriced_media:
            return {"kind": "unpriced"}
        return {"kind": "tokens", "tokens": estimate.charged(rate_limit_type)}

    def cap_output(self, data: object, tpm_share: int, call_type: str | None) -> None:
        cap_output_for_limit(data, tpm_share, call_type)

    @_host_call_sync
    def stash_call_type(self, _args: Mapping[str, object]) -> object:
        return get_or_create_request_stash().call_type

    @_host_call
    async def post_call_success(self, args: Mapping[str, object]) -> object:
        live: Final = _live()
        with service_target(RATE_LIMITS_TARGET):
            await limiter_post_call_success(self.post_call, live.data or {}, self._auth(), live.response)
        if not response_has_hidden_params(live.response):
            return None
        additional_headers: Final = ensure_response_additional_headers(live.response)
        for name, value in _PAIRS.validate_python(args["headers"]):
            additional_headers[name] = value  # pyright: ignore[reportArgumentType]  # the priority value as the handler set it
        return None

    @_host_call_sync
    def touch(self, args: Mapping[str, object]) -> object:
        touch_args: Final = (str(args["model"]), str(args["member"]), args["priority_class"], int(str(args["window"])))
        token: Final = next(_TOKENS)
        if self.usage_cache.redis_cache is not None and not _loop_running():
            self.deferred_touches[token] = touch_args
            return {"pending": token}
        context: Final = copy_context()
        coroutine: Final = self.active_teams.touch(*touch_args)  # pyright: ignore[reportArgumentType]  # the class as the handler passes it, a reserved priority of any type
        finished, out = _step(coroutine, context, None, None)
        if finished:
            return {"members": _members(out)}
        _live().suspended[token] = _Suspended(coroutine, out, context)
        return {"pending": token}

    @_host_call
    async def touched(self, args: Mapping[str, object]) -> object:
        token: Final = int(str(args["pending"]))
        deferred: Final = self.deferred_touches.pop(token, None)
        if deferred is not None:
            return _members(await self.active_teams.touch(*deferred))  # pyright: ignore[reportArgumentType]  # the class as the handler passes it, a reserved priority of any type
        return _members(await _finish(_live().suspended.pop(token)))

    @_host_call
    async def metrics_enabled(self, _args: Mapping[str, object]) -> object:
        return bool(_prometheus_loggers())

    @_host_call
    async def throttled(self, args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger

        for logger in _prometheus_loggers():
            try:
                logger.increment_dynamic_limiter_throttled(
                    model_group=str(args["model_group"]),
                    holder_kind=_HOLDER_KINDS.validate_python(args["holder_kind"]),
                    holder_id=str(args["holder_id"]),
                    holder_alias=args["holder_alias"] if isinstance(args["holder_alias"], str) else None,
                    priority=args["priority"],  # pyright: ignore[reportArgumentType]  # the priority header value as the handler passes it
                    rate_limit_type=str(args["rate_limit_type"]),
                    reason=str(args["reason"]),
                )
            except Exception as metrics_error:  # noqa: BLE001  # a failing metrics sink must not change the 429
                verbose_proxy_logger.warning("Dynamic rate limiter throttle metric failed: %s", metrics_error)
        return None

    @_host_call_sync
    def log(self, args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.log(_LOG_LEVELS[str(args["level"])], str(args["message"]))
        return None

    @_host_call_sync
    def unlicensed_priority(self, _args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger

        verbose_proxy_logger.error(UNLICENSED_PRIORITY_MESSAGE)
        return None

    @_host_call
    async def set_share(self, args: Mapping[str, object]) -> object:
        from litellm._logging import verbose_proxy_logger

        try:
            for logger in _prometheus_loggers():
                logger.set_dynamic_limiter_share(
                    model_group=str(args["model_group"]),
                    holder_kind=_HOLDER_KINDS.validate_python(args["holder_kind"]),
                    holder_id=str(args["holder_id"]),
                    holder_alias=args["holder_alias"] if isinstance(args["holder_alias"], str) else None,
                    priority=args["priority"],  # pyright: ignore[reportArgumentType]  # the priority value as the handler passes it
                    share_tokens=int(str(args["share_tokens"])),
                    borrowed_tokens=int(str(args["borrowed_tokens"])),
                )
        except Exception as metrics_error:  # noqa: BLE001  # a failing metrics sink must not change settlement
            verbose_proxy_logger.warning("Dynamic rate limiter share metrics failed: %s", metrics_error)
        return None


def shared_rate_limit_store(internal_usage_cache: DualCache) -> RateLimitStore:
    registered: Final = next(
        (
            callback.rate_limits
            for callback in litellm.callbacks
            if isinstance(callback, PROXY_MaxParallelRequestsHandler_v3)
            and callback.internal_usage_cache.dual_cache is internal_usage_cache
        ),
        None,
    )
    if registered is not None:
        return registered
    return native_rate_limit_store(internal_usage_cache)


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
    fields: Final = (None if wire.error is None else as_str_mapping(wire.error.fields)) or {}
    if fields.get("unverifiable") is True:
        return RateLimitUnverifiableError()
    token: Final = fields.get("token")
    raised: Final = None if live is None or not isinstance(token, int) else live.raised.pop(token, None)
    if isinstance(raised, Exception):
        return raised
    return HTTPException(status_code=wire.status, detail={"error": wire.message}, headers=dict(wire.headers))


def _native_answer(answer: Mapping[str, object], live: LiveRequest) -> Mapping[str, object]:
    if answer.get("unverifiable") is True:
        raise RateLimitUnverifiableError()
    raised_token: Final = answer.get("raised")
    if isinstance(raised_token, int) and raised_token in live.raised:
        raise live.raised[raised_token]
    if "type_error" in answer:
        raise TypeError(str(answer["type_error"]))
    if "failed" in answer:
        raise RuntimeError(str(answer["failed"]))
    return answer


@dataclass(frozen=True, slots=True)
class _CapacityHandle:
    capacity: "NativeModelCapacity"
    callback: RustHandle
    delegate: CapacityHost
    rate_limits: RateLimitStore

    def manifest(self) -> str:
        return self.callback.manifest()

    def request_state(self) -> object:
        return None

    def handle(
        self, hook: str, fields: Mapping[str, object], call: Mapping[str, object] | None, _state: object, /
    ) -> Awaitable[object]:
        live: Final = _LIVE.get()
        pre_call: Final = hook == Hook.pre_call.value
        model: Final = fields.get(Field.model.value) if pre_call else None
        return self.capacity.handle(
            hook,
            fields,
            call,
            None if live is None else live.state,
            self.delegate.facts(model if isinstance(model, str) else None),
            request=self._request(hook),
        )

    def _request(self, hook: str) -> "NativeRequestRedis | None":
        if hook == Hook.pre_call.value:
            return pre_call_request()
        if hook in _SETTLING_HOOKS and self.rate_limits.has_redis:
            return post_call_request()
        return None


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
        return attribute_or_none(view, Field.outcome.value) == AttemptOutcome.success
    return hook in _TARGETED_HOOKS


class NativeDynamicRateLimiter(RustLogger):
    inspects_request_content: ClassVar[bool] = False
    delegate: CapacityHost
    capacity: "NativeModelCapacity"
    rate_limits: RateLimitStore
    window_size: int

    def __new__(
        cls,
        internal_usage_cache: DualCache,
        time_provider: Callable[[], datetime] | None = None,
        *,
        rate_limits: RateLimitStore | None = None,
        media_model_resolver: MediaModelResolver | None = None,
    ) -> "NativeDynamicRateLimiter":
        from litellm.rust_bridge._native import NativeModelCapacity, NativeRateLimits

        register_abort_error(RATE_LIMIT_KIND, _rate_limit_error)
        register_abort_error(RERAISE_KIND, _reraise)
        store: Final = rate_limits or shared_rate_limit_store(internal_usage_cache)
        handle: Final = store.handle
        if not isinstance(handle, NativeRateLimits):
            raise TypeError(f"model capacity counts on the native rate limit store, not {type(handle).__name__}")
        window_size: Final = rate_limit_window_size()
        usage_cache: Final = InternalUsageCache(dual_cache=internal_usage_cache)
        engine: Final = native_engine(
            store, window_size, tpm_token_reservation_enabled(), time_provider or datetime.now
        )
        post_call: Final = LimiterPostCall(
            rate_limits=store,
            native=engine,
            batch_enqueued_token_store=NativeBatchEnqueuedTokenStore(usage_cache, engine),
            memory_lock=asyncio.Lock(),
        )
        delegate: Final = CapacityHost(internal_usage_cache, post_call, window_size, media_model_resolver)
        capacity: Final = NativeModelCapacity(
            delegate,
            handle,
            window_size,
            tpm_token_reservation_enabled(),
            None if time_provider is None else (lambda: time_provider().timestamp()),
        )
        native: Final = _CapacityHandle(capacity, capacity.callback(), delegate, store)
        manifest: Final = WireManifest.model_validate_json(native.manifest())
        logger: Final = object.__new__(type(manifest.selection.name, (cls,), _namespace(native, manifest)))
        RustLogger.__init__(logger, native, PROJECTIONS)
        logger.delegate = delegate
        logger.capacity = capacity
        logger.rate_limits = store
        logger.window_size = window_size
        return logger

    def __init__(
        self,
        internal_usage_cache: DualCache,
        time_provider: Callable[[], datetime] | None = None,
        *,
        rate_limits: RateLimitStore | None = None,
        media_model_resolver: MediaModelResolver | None = None,
    ) -> None:
        pass

    def update_variables(self, llm_router: Router) -> None:
        self.delegate.llm_router = llm_router

    def _state(self, stash: RequestRateLimiterStash) -> "NativeModelCapacityState":
        return stash.native_capacity(self.capacity.request_state)

    def _state_for(self, stash: RequestRateLimiterStash | None) -> "NativeModelCapacityState | None":
        return None if stash is None else self._state(stash)

    async def run_rust(
        self, hook: Hook, reads: frozenset[Field], view: object, ctx: CallContext[object]
    ) -> Patch | Abort | None:
        if hook is Hook.post_call_failure and reporting_nested_tool_call_failure():
            return None
        data: Final = as_str_mapping(getattr(view, Field.deprecated_kwargs.value, None))
        stash: Final = _request_stash(hook, data)
        auth: Final = (REQUEST_AUTH.get() or auth_from_request(data)) if hook is not Hook.attempt_ended else None
        if hook is Hook.pre_call and stash is not None:
            stash.request_body = data
            stash.request_team_id = None if auth is None else auth.team_id
        live: Final = LiveRequest(
            hook=hook,
            data=data if hook is not Hook.attempt_ended else None if stash is None else stash.request_body,
            auth=auth,
            stash=stash,
            response=getattr(view, Field.deprecated_response_obj.value, None),
            kwargs=data if hook is Hook.attempt_ended else None,
            state=self._state_for(stash),
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
                return self._capped(await super().run_rust(hook, reads, view, ctx), live)
        finally:
            _LIVE.reset(token)

    def _capped(self, verdict: Patch | Abort | None, live: LiveRequest) -> Patch | Abort | None:
        cap: Final = None if live.state is None else live.state.take_output_cap()
        if cap is None:
            return verdict
        try:
            self.delegate.cap_output(live.data, *cap)
        except HTTPException:
            raise
        except Exception as error:  # noqa: BLE001  # Python allows the request when capping its output fails
            from litellm._logging import verbose_proxy_logger

            verbose_proxy_logger.error("Error in dynamic rate limiter: %s, allowing request", error)
            return None
        return verdict

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
            answer: Final = await self.capacity.fallback_target_allowed(
                state,
                key_inputs(user_api_key_dict),
                model,
                commit,
                self.delegate.facts(model, call_type=True),
                request=pre_call_request(),
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
            answer: Final = await self.capacity.enforce_frame(
                state,
                key_inputs(user_api_key_dict),
                requested_model,
                self.delegate.facts(requested_model),
                request=pre_call_request(),
            )
        finally:
            _LIVE.reset(token)
        if answer.get("unverifiable") is True:
            raise ServiceUnavailableError(
                message=RATE_LIMIT_UNVERIFIABLE_MESSAGE, llm_provider="litellm_proxy", model=requested_model
            ) from RateLimitUnverifiableError()
        refused: Final = as_str_mapping(_native_answer(answer, live).get("refused"))
        if refused is not None:
            raise _proxy_rate_limit_error(refused)

    async def charge_realtime_capacity_turn(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None, usage: Usage
    ) -> None:
        if requested_model is None:
            return
        state: Final = self._state_for(get_request_stash())
        live: Final = LiveRequest(auth=user_api_key_dict, state=state)
        token: Final = _LIVE.set(live)
        prompt_details: Final = usage.prompt_tokens_details
        cached_tokens: Final = 0 if prompt_details is None else prompt_details.cached_tokens or 0
        try:
            answer: Final = await self.capacity.charge_turn(
                state,
                key_inputs(user_api_key_dict),
                requested_model,
                (usage.prompt_tokens, usage.completion_tokens, usage.total_tokens, cached_tokens),
                self.delegate.facts(requested_model),
            )
        finally:
            _LIVE.reset(token)
        _ = _native_answer(answer, live)


def build_native_dynamic_rate_limiter(
    internal_usage_cache: DualCache,
    time_provider: Callable[[], datetime] | None = None,
    *,
    rate_limits: RateLimitStore | None = None,
    media_model_resolver: MediaModelResolver | None = None,
) -> NativeDynamicRateLimiter:
    return NativeDynamicRateLimiter(
        internal_usage_cache, time_provider, rate_limits=rate_limits, media_model_resolver=media_model_resolver
    )


@dataclass(slots=True)
class _Registered:
    logger: NativeDynamicRateLimiter | None = None


_REGISTERED: Final = _Registered()


def registered_native_dynamic_rate_limiter() -> NativeDynamicRateLimiter | None:
    return _REGISTERED.logger


def native_dynamic_rate_limiter(internal_usage_cache: DualCache, llm_router: object) -> NativeDynamicRateLimiter:
    if _REGISTERED.logger is None:
        _REGISTERED.logger = build_native_dynamic_rate_limiter(internal_usage_cache)
    if isinstance(llm_router, Router):
        _REGISTERED.logger.update_variables(llm_router)
    return _REGISTERED.logger


__all__ = [
    "CapacityHost",
    "NativeDynamicRateLimiter",
    "build_native_dynamic_rate_limiter",
    "native_dynamic_rate_limiter",
    "registered_native_dynamic_rate_limiter",
    "shared_rate_limit_store",
]
