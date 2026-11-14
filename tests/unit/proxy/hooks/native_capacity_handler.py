from collections.abc import Callable, Mapping
from datetime import datetime
from typing import Final

from litellm import DualCache, Router
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
from litellm.proxy.hooks.parallel_request_limiter_v3 import PROXY_MaxParallelRequestsHandler_v3
from litellm.proxy.hooks.priority_class_share import ActiveTeamRegistry
from litellm.proxy.request_auth_context import REQUEST_AUTH
from litellm.proxy.utils import HookRun, InternalUsageCache, hook_fields, raise_v2_abort
from litellm.proxy.v2_proxy_fields import (
    DERIVED_POST_CALL_FAILURE_PROXY_FIELDS,
    DERIVED_POST_CALL_SUCCESS_FIELDS,
    DERIVED_RESPONSE_HEADERS_FIELDS,
    PRE_CALL_SOURCES,
)
from litellm.types.integrations.custom_logger_v2 import Outcome as AttemptOutcome


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
