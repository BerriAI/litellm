from collections.abc import Callable, Mapping, Sequence
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime
from typing import Final

from litellm import DualCache, Router
from litellm.types.router import ModelGroupInfo
from litellm.integrations.custom_logger_v2.call import CallFacts, CallIds, CallKind, CallTarget, CallTiming
from litellm.integrations.custom_logger_v2.derivations import DERIVED_ATTEMPT_FIELDS
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import HookArgs, with_sources
from litellm.integrations.custom_logger_v2.legacy import (
    PROXY_REQUEST_KEYS,
    Admit,
    KwargsFields,
    hook_subscribers,
    kwargs_after,
    run_subscribers,
)
from litellm.litellm_core_utils.litellm_logging import LOGGING_KEYS
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.hooks.model_capacity_native import (
    NativeDynamicRateLimiter,
    build_native_dynamic_rate_limiter,
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    BorrowingTeamModelLimits,
    PROXY_MaxParallelRequestsHandler_v3,
    RateLimitDescriptor,
    RateLimitedModel,
    borrowing_team_model_limits,
)
from litellm.proxy.hooks.priority_class_share import ActiveTeamRegistry
from litellm.proxy.request_auth_context import REQUEST_AUTH
from litellm.redis.stores.rate_limits import RateLimitScope
from litellm.proxy.utils import HookRun, InternalUsageCache, hook_fields, raise_v2_abort
from litellm.proxy.v2_proxy_fields import (
    DERIVED_POST_CALL_FAILURE_PROXY_FIELDS,
    DERIVED_POST_CALL_SUCCESS_FIELDS,
    DERIVED_RESPONSE_HEADERS_FIELDS,
    PRE_CALL_SOURCES,
)
from litellm.types.integrations.custom_logger_v2 import Outcome as AttemptOutcome


@dataclass(frozen=True, slots=True)
class TeamShare:
    share: RateLimitDescriptor
    ceiling: RateLimitDescriptor | None


def _team_descriptor(
    limiter: PROXY_MaxParallelRequestsHandler_v3, key: str, value: str, limits: BorrowingTeamModelLimits
) -> RateLimitDescriptor:
    return RateLimitDescriptor(
        key=key,
        value=value,
        rate_limit={
            "tokens_per_unit": limits.tpm,
            "requests_per_unit": limits.rpm,
            "window_size": limiter.window_size,
        },
    )


def borrowing_team_share(
    handler: "NativeDynamicRateLimitHandler", model: str, auth: UserAPIKeyAuth
) -> TeamShare | None:
    limiter: Final = handler.v3_limiter
    rate_limited: Final = limiter.rate_limited_model(model) or RateLimitedModel(requested=model, group=model)
    limits: Final = borrowing_team_model_limits(auth, rate_limited)
    if limits.tpm is None and limits.rpm is None:
        return None
    value: Final = f"{auth.team_id}:{rate_limited.group}"
    ceiling: Final = limits.borrow_ceiling()
    return TeamShare(
        share=_team_descriptor(limiter, "model_per_team_saturated", value, limits),
        ceiling=None
        if ceiling is None
        else _team_descriptor(limiter, "model_per_team_max_borrow_multiple", value, ceiling),
    )


def model_tracking_descriptor(
    handler: "NativeDynamicRateLimitHandler", model: str, model_group_info: ModelGroupInfo
) -> RateLimitDescriptor:
    return RateLimitDescriptor(
        key="model_saturation_check",
        value=model,
        rate_limit={
            "requests_per_unit": model_group_info.rpm or None,
            "tokens_per_unit": model_group_info.tpm or None,
            "window_size": handler.v3_limiter.window_size,
        },
    )


async def model_saturation(
    handler: "NativeDynamicRateLimitHandler", model: str, model_group_info: ModelGroupInfo
) -> float:
    rate_limits: Final = handler.v3_limiter.rate_limits
    scope: Final = RateLimitScope("model_saturation_check", model)
    requests: Final = await rate_limits.get_from_redis(scope.counter("requests"))
    tokens: Final = await rate_limits.get_from_redis(scope.counter("tokens"))
    return max(
        0.0,
        0.0 if not model_group_info.rpm or requests is None else int(str(requests)) / model_group_info.rpm,
        0.0 if not model_group_info.tpm or tokens is None else float(str(tokens)) / model_group_info.tpm,
    )


_SENT: Final[ContextVar[list[RateLimitDescriptor] | None]] = ContextVar("native_capacity_sent", default=None)


def _record_descriptors(limiter: PROXY_MaxParallelRequestsHandler_v3) -> None:
    if "should_rate_limit" in vars(limiter):
        return
    original: Final = limiter.should_rate_limit

    async def recorded(descriptors: Sequence[RateLimitDescriptor], *args: object, **kwargs: object) -> object:
        sent: Final = _SENT.get()
        if sent is not None:
            sent.extend(descriptors)
        return await original(descriptors, *args, **kwargs)

    vars(limiter)["should_rate_limit"] = recorded


async def priority_based_descriptors(
    handler: "NativeDynamicRateLimitHandler", model: str, user_api_key_dict: UserAPIKeyAuth, priority: str | None
) -> list[RateLimitDescriptor]:
    assert (user_api_key_dict.metadata or {}).get("priority") == priority
    _record_descriptors(handler.v3_limiter)
    sent: Final[list[RateLimitDescriptor]] = []
    token: Final = _SENT.set(sent)
    try:
        await handler.async_pre_call_hook(
            user_api_key_dict, DualCache(), {"model": model, "max_tokens": 0}, "acompletion"
        )
    finally:
        _SENT.reset(token)
    return list(
        {descriptor["value"]: descriptor for descriptor in sent if descriptor["key"] == "priority_model"}.values()
    )


async def _no_legacy(_logger: object, _data: object) -> None:
    return None


def _call(data: Mapping[str, object], call_type: str) -> CallFacts:
    return CallFacts(
        ids=CallIds(call_id=str(data.get("litellm_call_id") or "")),
        kind=CallKind(call_type=call_type),
        target=CallTarget(model=str(data.get("model"))),
        timing=CallTiming(start=1.0),
    )


class NativeDynamicRateLimitHandler:
    def __init__(self, internal_usage_cache: DualCache, time_provider: Callable[[], datetime] | None = None) -> None:
        self.logger: Final[NativeDynamicRateLimiter] = build_native_dynamic_rate_limiter(
            internal_usage_cache, time_provider=time_provider
        )
        self.callbacks: Final[list[object]] = [self.logger]

    @property
    def v3_limiter(self) -> PROXY_MaxParallelRequestsHandler_v3:
        return self.logger.v3_limiter

    @property
    def internal_usage_cache(self) -> InternalUsageCache:
        return self.logger.v3_limiter.internal_usage_cache

    @property
    def active_teams(self) -> ActiveTeamRegistry:
        return self.logger.delegate.active_teams

    @property
    def llm_router(self) -> Router:
        router: Final = self.logger.delegate.llm_router
        assert router is not None
        return router

    @v3_limiter.setter
    def v3_limiter(self, limiter: PROXY_MaxParallelRequestsHandler_v3) -> None:
        vars(self.logger.delegate)["v3_limiter"] = limiter

    @active_teams.setter
    def active_teams(self, registry: ActiveTeamRegistry) -> None:
        vars(self.logger.delegate)["active_teams"] = registry

    def update_variables(self, llm_router: Router) -> None:
        self.logger.update_variables(llm_router)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict[str, object],  # mutable-ok: the legacy pre-call hook rewrites the proxy's request dict
        call_type: str,
    ) -> None:
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            result: Final = await run_subscribers(
                hook_subscribers(Hook.pre_call, self.callbacks),
                with_sources(
                    KwargsFields(data, PROXY_REQUEST_KEYS),
                    PRE_CALL_SOURCES,
                    HookArgs(auth=user_api_key_dict, request=data, call_type=call_type),
                ),
                _no_legacy,
                call=_call(data, call_type),
            )
            after: Final = kwargs_after(raise_v2_abort(result))
        finally:
            REQUEST_AUTH.reset(auth_token)
        if isinstance(after, dict) and after is not data:
            data.clear()
            data.update(after)
        return None

    async def async_post_call_success_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> object:
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            await run_subscribers(
                hook_subscribers(Hook.post_call_success, self.callbacks),
                with_sources(
                    hook_fields(data, HookRun(response=response), Field.response),
                    DERIVED_POST_CALL_SUCCESS_FIELDS,
                    HookArgs(auth=user_api_key_dict, request=data),
                ),
                _no_legacy,
            )
        finally:
            REQUEST_AUTH.reset(auth_token)
        return response

    async def async_post_call_response_headers_hook(
        self, data: dict[str, object], user_api_key_dict: UserAPIKeyAuth, response: object
    ) -> dict[str, str] | None:  # mutable-ok: the legacy hook returns the headers it adds
        merged: Final[dict[str, str]] = {}
        await run_subscribers(
            hook_subscribers(Hook.response_headers, self.callbacks, admit=Admit.every_entry),
            with_sources(
                hook_fields(data, HookRun(response=merged), Field.response_headers),
                DERIVED_RESPONSE_HEADERS_FIELDS,
                HookArgs(auth=user_api_key_dict, request=data),
            ),
            _no_legacy,
        )
        return merged or None

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],  # mutable-ok: the native runs on the proxy's request dict
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        auth_token: Final = REQUEST_AUTH.set(user_api_key_dict)
        try:
            await run_subscribers(
                hook_subscribers(Hook.post_call_failure, self.callbacks, admit=Admit.every_entry),
                with_sources(
                    hook_fields(request_data, HookRun(response=original_exception), Field.error),
                    DERIVED_POST_CALL_FAILURE_PROXY_FIELDS,
                    HookArgs(auth=user_api_key_dict, request=request_data),
                ),
                _no_legacy,
            )
        finally:
            REQUEST_AUTH.reset(auth_token)

    async def _attempt_ended(self, kwargs: dict[str, object], response_obj: object, failed: bool) -> None:
        call_type: Final = kwargs.get("call_type")
        await run_subscribers(
            hook_subscribers(Hook.attempt_ended, self.callbacks, admit=Admit.every_entry),
            with_sources(
                KwargsFields(kwargs, LOGGING_KEYS),
                DERIVED_ATTEMPT_FIELDS,
                HookArgs(
                    request=kwargs,
                    result=response_obj,
                    outcome=AttemptOutcome.failure if failed else AttemptOutcome.success,
                ),
            ),
            _no_legacy,
            call=_call(kwargs, call_type if isinstance(call_type, str) else "acompletion"),
        )

    async def async_log_success_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._attempt_ended(kwargs, response_obj, failed=False)

    async def async_log_failure_event(
        self, kwargs: dict[str, object], response_obj: object, start_time: object, end_time: object
    ) -> None:
        await self._attempt_ended(kwargs, response_obj, failed=True)

    async def async_fallback_target_allowed(
        self, user_api_key_dict: UserAPIKeyAuth, model: str, request_data: Mapping[str, object], *, commit: bool = True
    ) -> bool:
        return await self.logger.async_fallback_target_allowed(user_api_key_dict, model, request_data, commit=commit)

    async def enforce_token_limits_for_frame(
        self, user_api_key_dict: UserAPIKeyAuth, requested_model: str | None
    ) -> None:
        await self.logger.enforce_token_limits_for_frame(
            user_api_key_dict=user_api_key_dict, requested_model=requested_model
        )
