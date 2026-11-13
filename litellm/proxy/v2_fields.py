import time
from collections.abc import Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final

from litellm.integrations.custom_logger_v2.call import CallFacts, CallIds, CallKind, CallTarget, CallTiming
from litellm.integrations.custom_logger_v2.derivations import (
    AUTH_SLOTS,
    POST_CALL_FAILURE_SLOTS,
    PRE_CALL_AUTH_SLOTS,
    mapping,
    member,
)
from litellm.integrations.custom_logger_v2.enums import Field
from litellm.integrations.custom_logger_v2.slots import HookArgs, Slot, derived, mapped
from litellm.types.integrations.custom_logger_v2 import OutputCap, RateLimitStatuses, coerce_record


def _call_id(args: HookArgs) -> str | None:
    call_id: Final = member(args.request, "litellm_call_id")
    return call_id if isinstance(call_id, str) else None


def rate_limit_statuses(args: HookArgs) -> RateLimitStatuses | None:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import get_request_stash_for_call

    stash: Final = get_request_stash_for_call(_call_id(args))
    response: Final = None if stash is None else stash.rate_limit_response
    if response is None:
        return None
    return coerce_record(
        RateLimitStatuses, {"overall_code": response.get("overall_code"), "statuses": response.get("statuses")}
    )


def write_rate_limit_statuses(args: HookArgs, value: object) -> None:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import claim_request_stash_for_data

    record: Final = value if isinstance(value, RateLimitStatuses) else RateLimitStatuses.model_validate(value)
    claim_request_stash_for_data(mapping(args.request) or {}).rate_limit_response = {  # pyright: ignore[reportAttributeAccessIssue]  # the stash's TypedDict is built from the validated record
        "overall_code": record.overall_code or "OK",
        "statuses": [status.model_dump(mode="json", exclude_none=True) for status in record.statuses or ()],
    }


def output_cap(args: HookArgs) -> OutputCap | None:
    from litellm.proxy.hooks.parallel_request_limiter_v3 import (
        _PROXY_MaxParallelRequestsHandler_v3,  # pyright: ignore[reportPrivateUsage]  # the limiter's own explicit-cap rule
    )

    request: Final = mapping(args.request)
    if request is None:
        return None
    explicit: Final = _PROXY_MaxParallelRequestsHandler_v3._get_explicit_output_cap(request, args.call_type)  # pyright: ignore[reportPrivateUsage]  # the limiter's own explicit-cap rule
    return coerce_record(OutputCap, {"explicit": explicit})


PRE_CALL_SLOTS: Final[Mapping[Field, Slot]] = MappingProxyType(
    {
        **PRE_CALL_AUTH_SLOTS,
        Field.rate_limit_statuses: mapped(rate_limit_statuses, write_rate_limit_statuses),
        Field.output_cap: derived(output_cap),
    }
)
POST_CALL_SUCCESS_SLOTS: Final = AUTH_SLOTS
POST_CALL_FAILURE_PROXY_SLOTS: Final = POST_CALL_FAILURE_SLOTS
RESPONSE_HEADERS_SLOTS: Final = AUTH_SLOTS


def proxy_call_facts(
    data: Mapping[str, object] | None,
    call_type: str | None,
    auth: object = None,
    endpoint_type: str | None = None,
) -> CallFacts:
    request: Final = data or {}
    model: Final = request.get("model")
    session: Final = request.get("litellm_session_id")
    route: Final = getattr(auth, "request_route", None)
    return CallFacts(
        ids=CallIds(
            call_id=str(request.get("litellm_call_id") or ""),
            session_id=session if isinstance(session, str) else None,
        ),
        kind=CallKind(
            call_type=str(call_type or ""),
            route=route if isinstance(route, str) else None,
            stream=request.get("stream") is True,
            endpoint_type=endpoint_type,
        ),
        target=CallTarget(model=model if isinstance(model, str) else ""),
        timing=CallTiming(start=time.time()),
    )


def post_call_facts(data: Mapping[str, object] | None, auth: object = None, logging_obj: object = None) -> CallFacts:
    request: Final = data or {}
    facts_of: Final = getattr(logging_obj or request.get("litellm_logging_obj"), "v2_call_facts", None)
    logged: Final = facts_of() if callable(facts_of) else None
    if not isinstance(logged, CallFacts):
        call_type: Final = request.get("call_type")
        return proxy_call_facts(request, call_type if isinstance(call_type, str) else None, auth)
    route: Final = getattr(auth, "request_route", None)
    return replace(logged, kind=replace(logged.kind, route=route)) if isinstance(route, str) else logged


__all__ = [
    "POST_CALL_FAILURE_PROXY_SLOTS",
    "POST_CALL_SUCCESS_SLOTS",
    "PRE_CALL_SLOTS",
    "RESPONSE_HEADERS_SLOTS",
    "output_cap",
    "post_call_facts",
    "proxy_call_facts",
    "rate_limit_statuses",
    "write_rate_limit_statuses",
]
