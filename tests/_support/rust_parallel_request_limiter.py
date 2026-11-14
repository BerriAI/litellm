import copy
import dataclasses
import uuid
from collections.abc import Mapping
from types import SimpleNamespace
from typing import Final

import pytest
from pydantic import TypeAdapter

from litellm.caching.caching import DualCache
from litellm.integrations.custom_logger_v2.call import CallContext
from litellm.integrations.custom_logger_v2.enforcement import FieldView
from litellm.integrations.custom_logger_v2.enums import Field, Hook
from litellm.integrations.custom_logger_v2.field_sources import FieldSource, HookArgs, with_sources
from litellm.integrations.custom_logger_v2.legacy import PROXY_REQUEST_KEYS, KwargsFields
from litellm.integrations.custom_logger_v2.outcomes import Abort, Patch
from litellm.integrations.custom_logger_v2.patches import apply_patch
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.utils import CallTypes
from litellm.proxy.hooks.parallel_request_limiter_v3 import (
    _NATIVE_STASH,  # pyright: ignore[reportPrivateUsage]  # the replicas read the native's stash in the limiter's own shape
    ParallelSlotAcquisition,
    RateLimitDescriptor,
    RateLimitedModel,
    RateLimitResponse,
    RequestRateLimiterStash,
    _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # the replicas subclass the limiter the glue wraps
    _reservation_operation,  # pyright: ignore[reportPrivateUsage]  # the replicas read the native's stash in the limiter's own shape
    _window_identities,  # pyright: ignore[reportPrivateUsage]  # the replicas read the native's stash in the limiter's own shape
)
from litellm.proxy.hooks.parallel_request_limiter_v3 import get_request_stash as _live_request_stash
from litellm.redis.stores.rate_limits import RateLimitScope
from litellm.proxy.hooks.parallel_request_limiter_v3_native import (
    NATIVE_PARALLEL_REQUEST_LIMITER_ENV,
    ParallelRequestLimiterNativeLogger,
    PreCallAuthScope,
    native_parallel_request_limiter,
)
from litellm.proxy.v2_proxy_fields import DERIVED_POST_CALL_FAILURE_PROXY_FIELDS, PRE_CALL_SOURCES, proxy_call_facts
from litellm.rust_bridge._native import NativeParallelRequestLimiter
from litellm.types.passthrough_endpoints.pass_through_endpoints import EndpointType

_READS: Final = frozenset({Field.deprecated_kwargs})
_DESCRIPTORS: Final = TypeAdapter(list[RateLimitDescriptor])


def _view(
    data: dict[str, object], auth: object, call_type: str | None, sources: Mapping[Field, FieldSource]
) -> FieldView:
    return FieldView(
        with_sources(
            KwargsFields(data, PROXY_REQUEST_KEYS),
            sources,
            HookArgs(auth=auth, request=data, call_type=call_type),
        )
    )


def _context(
    hook: Hook, data: dict[str, object], call_type: str | None, auth: object, endpoint_type: str
) -> CallContext[object]:
    return CallContext(hook=hook, call=proxy_call_facts(data, call_type, auth, endpoint_type), state=None)


def _raise(outcome: object) -> None:
    if isinstance(outcome, Abort) and outcome.exception is not None:
        raise outcome.exception


_NATIVES: Final[list[NativeParallelRequestLimiter]] = []


def get_request_stash() -> RequestRateLimiterStash | None:
    live: Final = _live_request_stash()
    if live is None or live.native_state is None or not _NATIVES:
        return live
    native: Final = _NATIVES[-1]
    if not native.owned(live.native_state):
        return live
    exported: Final = _NATIVE_STASH.validate_python(native.view(live.native_state))
    slot: Final = exported["parallel_slot"]
    model: Final = exported["reserved_model"]
    response: Final = exported["rate_limit_response"]
    itpm: Final = exported["itpm"]
    otpm: Final = exported["otpm"]
    return dataclasses.replace(
        live,
        rate_limit_response=None
        if response is None
        else RateLimitResponse(overall_code=response["overall_code"], statuses=list(response["statuses"])),
        parallel_slot=None
        if slot is None
        else ParallelSlotAcquisition(
            slot_id=slot["slot_id"], gauges=tuple(RateLimitScope(key, value) for key, value in slot["gauges"])
        ),
        reserved_tokens=exported["reserved_tokens"],
        reserved_model=None if model is None else RateLimitedModel(requested=model[0], group=model[1]),
        reserved_scopes=frozenset(exported["reserved_scopes"]),
        fallback_models=frozenset(exported["fallback_models"]),
        reservation_released=exported["reservation_released"],
        io_reservation_released=exported["io_reservation_released"],
        tpm_limited_tags=frozenset(exported["tpm_limited_tags"]),
        ptu_reserved_tokens=exported["ptu_reserved_tokens"],
        ptu_reserved_window_identities=_window_identities(exported["ptu_windows"]),
        itpm_reserved_tokens=itpm["tokens"],
        itpm_reserved_scopes=frozenset(itpm["scopes"]),
        itpm_reserved_window_identities=_window_identities(itpm["windows"]),
        otpm_reserved_tokens=otpm["tokens"],
        otpm_reserved_scopes=frozenset(otpm["scopes"]),
        otpm_reserved_window_identities=_window_identities(otpm["windows"]),
        batch_enqueued_reservation=exported["batch_enqueued_reservation"],
        batch_tpd_refund_ops=tuple(_reservation_operation(op) for op in exported["batch_tpd_refunds"]),
    )


def rate_limit_descriptors() -> list[RateLimitDescriptor] | None:
    live: Final = _live_request_stash()
    if live is None or live.native_state is None or not _NATIVES:
        return None
    native: Final = _NATIVES[-1]
    if not native.owned(live.native_state):
        return None
    return _DESCRIPTORS.validate_python(native.descriptors(live.native_state)) or None


class RustParallelRequestLimiter(_PROXY_MaxParallelRequestsHandler_v3):
    def __init__(self, *args: object, **kwargs: object) -> None:
        with pytest.MonkeyPatch.context() as flag:
            flag.setenv(NATIVE_PARALLEL_REQUEST_LIMITER_ENV, "true")
            super().__init__(*args, **kwargs)  # pyright: ignore[reportArgumentType]  # the replicas pass the limiter's own arguments
        assert self.native_companion is not None
        _NATIVES.append(self.native_companion)
        self.native_logger: Final[ParallelRequestLimiterNativeLogger] = native_parallel_request_limiter(self)
        self.minted_call_ids: Final[set[str]] = set()

    def _proxy_call_id(
        self, data: dict[str, object], call_type: str
    ) -> None:  # mutable-ok: the proxy stamps the call id into the body
        current: Final = data.get("litellm_call_id")
        if (current is not None and current not in self.minted_call_ids) or call_type == CallTypes.call_mcp_tool.value:
            return
        minted: Final = str(uuid.uuid4())
        self.minted_call_ids.add(minted)
        data["litellm_call_id"] = minted

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: DualCache,
        data: dict,  # pyright: ignore[reportMissingTypeArgument]  # the legacy hook's signature
        call_type: str,
        endpoint_type: EndpointType = EndpointType.GENERIC,
    ) -> Exception | str | dict[str, object] | None:
        self._proxy_call_id(data, call_type)
        with PreCallAuthScope(user_api_key_dict):
            outcome: Final = await self.native_logger.run_rust(
                Hook.pre_call,
                _READS,
                _view(data, user_api_key_dict, call_type, PRE_CALL_SOURCES),
                _context(Hook.pre_call, data, call_type, user_api_key_dict, endpoint_type.value),
            )
        _raise(outcome)
        return await super().async_pre_call_hook(user_api_key_dict, cache, data, call_type, endpoint_type=endpoint_type)

    async def async_post_call_success_hook(self, data: dict, user_api_key_dict: UserAPIKeyAuth, response) -> None:  # pyright: ignore[reportMissingTypeArgument, reportMissingParameterType]  # the legacy hook's signature
        outcome: Final = await self.native_logger.run_rust(
            Hook.post_call_success,
            frozenset({Field.deprecated_kwargs, Field.deprecated_response_obj}),
            SimpleNamespace(deprecated_kwargs=data, deprecated_response_obj=response),
            _context(Hook.post_call_success, data, None, user_api_key_dict, EndpointType.GENERIC.value),
        )
        _raise(outcome)
        return await super().async_post_call_success_hook(data, user_api_key_dict, response)

    async def async_logging_hook(self, kwargs: dict, result, call_type: str):  # pyright: ignore[reportMissingTypeArgument, reportMissingParameterType]  # the legacy hook's signature
        payload: Final = kwargs.get("standard_logging_object")
        prepared: Final = await self.native_logger.run_rust(
            Hook.log_prepare,
            frozenset({Field.deprecated_kwargs, Field.standard_logging_payload}),
            SimpleNamespace(deprecated_kwargs=kwargs, standard_logging_payload=payload),
            _context(Hook.log_prepare, kwargs, call_type, None, EndpointType.GENERIC.value),
        )
        if isinstance(prepared, Patch) and payload is not None:
            kwargs["standard_logging_object"] = apply_patch(
                {Field.standard_logging_payload: copy.deepcopy(payload)}, prepared
            )[Field.standard_logging_payload]
        return await super().async_logging_hook(kwargs, result, call_type)

    async def async_post_call_failure_hook(
        self,
        request_data: dict[str, object],  # mutable-ok: the base hook declares this parameter as a dict
        original_exception: Exception,
        user_api_key_dict: UserAPIKeyAuth,
        traceback_str: str | None = None,
    ) -> None:
        await self.native_logger.run_rust(
            Hook.post_call_failure,
            _READS,
            _view(request_data, user_api_key_dict, None, DERIVED_POST_CALL_FAILURE_PROXY_FIELDS),
            _context(Hook.post_call_failure, request_data, None, user_api_key_dict, EndpointType.GENERIC.value),
        )
        return await super().async_post_call_failure_hook(
            request_data, original_exception, user_api_key_dict, traceback_str
        )


def estimate_tokens_for_request(
    handler: _PROXY_MaxParallelRequestsHandler_v3,
    data: object,
    call_type: str | None = None,
    min_configured_tpm_limit: int | None = None,
    configured_output_tokens: int | None = None,
) -> int:
    native: Final = handler.native_companion
    assert native is not None
    estimated: Final = native.estimate_tokens_for_request(
        data, call_type or "", min_configured_tpm_limit, configured_output_tokens
    )
    if estimated is not None:
        return estimated
    return handler._estimate_tokens_for_request(  # pyright: ignore[reportPrivateUsage]  # the estimate the native hands back to the host
        data,
        min_configured_tpm_limit=min_configured_tpm_limit,
        call_type=call_type,
        configured_output_tokens=configured_output_tokens,
    )


def estimate_input_and_output_tokens(
    handler: _PROXY_MaxParallelRequestsHandler_v3,
    data: object,
    call_type: str | None = None,
    min_configured_tpm_limit: int | None = None,
    configured_output_tokens: int | None = None,
    include_media: bool = True,
) -> tuple[int, int]:
    native: Final = handler.native_companion
    assert native is not None
    estimated: Final = native.estimate_input_and_output_tokens(
        data, call_type or "", min_configured_tpm_limit, configured_output_tokens, include_media
    )
    if estimated is not None:
        return estimated
    return handler._estimate_input_and_output_tokens(  # pyright: ignore[reportPrivateUsage]  # the estimate the native hands back to the host
        data,
        min_configured_tpm_limit=min_configured_tpm_limit,
        call_type=call_type,
        configured_output_tokens=configured_output_tokens,
        include_media=include_media,
    )


PROXY_MaxParallelRequestsHandler_v3 = RustParallelRequestLimiter

__all__ = [
    "PROXY_MaxParallelRequestsHandler_v3",
    "RustParallelRequestLimiter",
    "estimate_input_and_output_tokens",
    "estimate_tokens_for_request",
    "get_request_stash",
    "rate_limit_descriptors",
]
