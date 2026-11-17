import time
from collections.abc import Callable, Mapping
from dataclasses import replace
from types import MappingProxyType
from typing import Final

from pydantic import ValidationError

from litellm.integrations.custom_logger_v2.call import CallFacts, CallIds, CallKind, CallTarget, CallTiming
from litellm.integrations.custom_logger_v2.derivations import (
    DERIVED_AUTH_FIELDS,
    DERIVED_POST_CALL_FAILURE_FIELDS,
    PRE_CALL_AUTH_SOURCES,
    mapping,
    member,
)
from litellm.integrations.custom_logger_v2.enums import Field
from litellm.integrations.custom_logger_v2.field_sources import FieldSource, HookArgs, derived, mapped
from litellm.types.integrations.custom_logger_v2 import LimiterRequest, OutputCap, RateLimitStatuses, coerce_record


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
        _get_explicit_output_cap,  # pyright: ignore[reportPrivateUsage]  # the limiter's own explicit-cap rule
    )

    request: Final = mapping(args.request)
    if request is None:
        return None
    return coerce_record(OutputCap, {"explicit": _get_explicit_output_cap(request, args.call_type)})


def _raised(error: Exception | None, prefix: str) -> Mapping[str, object]:
    if error is None:
        return {}
    return {f"{prefix}_error_type": type(error).__name__, f"{prefix}_error_text": str(error)}


def _attempt(read: Callable[[], object]) -> tuple[object, Exception | None]:
    try:
        return read(), None
    except Exception as error:  # noqa: BLE001  # the port reproduces what the limiter does when its reader raises
        return None, error


def _fallback_on_saturation(request: Mapping[str, object]) -> tuple[Mapping[str, object], tuple[str, ...]]:
    from litellm.proxy.common_utils.proxy_rate_limit_error import (
        _FALLBACK_ON_SATURATION_FLAG,  # pyright: ignore[reportPrivateUsage]  # the capacity callback's own flag parse
        FALLBACK_ON_SATURATION_SETTING,
        _metadata_bucket,  # pyright: ignore[reportPrivateUsage]  # the capacity callback's own settings read
        saturation_fallback_walk_armed,
    )

    raw: Final = _metadata_bucket(request.get("router_settings_override")).get(FALLBACK_ON_SATURATION_SETTING)
    try:
        enabled: Final = _FALLBACK_ON_SATURATION_FLAG.validate_python(raw, strict=True) is True
    except ValidationError:
        text, error = _attempt(lambda: repr(raw))
        return {
            "fallback_on_saturation": False,
            "fallback_on_saturation_invalid": text,
            "fallback_walk_armed": saturation_fallback_walk_armed(),
        }, () if error is None else ("fallback_on_saturation_invalid",)
    return {"fallback_on_saturation": enabled, "fallback_walk_armed": saturation_fallback_walk_armed()}, ()


def _metadata_get(request: Mapping[str, object], bucket: str, name: str) -> object:
    # The limiter read these buckets with ``.get``, so a bucket that is not a mapping raises the same error.
    found: Final[object] = getattr(request.get(bucket) or {}, "get")(name)  # noqa: B009  # the bucket's own ``get``, raising as the limiter did
    return found


def request_session_id(request: Mapping[str, object]) -> str | None:
    """The agent session a request belongs to: ``litellm_session_id``, else the session id in its metadata."""
    session_id: Final = request.get("litellm_session_id")
    if session_id:
        return str(session_id)
    for bucket in ("metadata", "litellm_metadata"):
        metadata_session_id = _metadata_get(request, bucket, "session_id")
        if metadata_session_id:
            return str(metadata_session_id)
    return None


def resolved_agent_id(auth: object, request: Mapping[str, object]) -> object:
    """The agent a request is made for. The key's own agent wins over one the request metadata names."""
    key_agent_id: Final = getattr(auth, "agent_id", None)
    if key_agent_id:
        return key_agent_id
    return _metadata_get(request, "metadata", "agent_id") or _metadata_get(request, "litellm_metadata", "agent_id")


def limiter_request(args: HookArgs) -> LimiterRequest | None:
    from litellm.proxy.common_utils.http_parsing_utils import get_tags_from_request_body

    request: Final = mapping(args.request)
    if request is None:
        return None
    session_id, session_error = _attempt(lambda: request_session_id(request))
    agent_id, agent_error = _attempt(lambda: resolved_agent_id(args.auth, request))
    saturation, saturation_rejected = _fallback_on_saturation(request)
    return coerce_record(
        LimiterRequest,
        {
            "tags": tuple(get_tags_from_request_body(request)),
            "session_id": session_id,
            "resolved_agent_id": agent_id,
            **_raised(session_error, "session_id"),
            **_raised(agent_error, "resolved_agent_id"),
            **saturation,
        },
        rejected=saturation_rejected,
    )


PRE_CALL_SOURCES: Final[Mapping[Field, FieldSource]] = MappingProxyType(
    {
        **PRE_CALL_AUTH_SOURCES,
        Field.rate_limit_statuses: mapped(rate_limit_statuses, write_rate_limit_statuses),
        Field.output_cap: derived(output_cap),
        Field.limiter_request: derived(limiter_request),
    }
)
DERIVED_POST_CALL_SUCCESS_FIELDS: Final = DERIVED_AUTH_FIELDS
DERIVED_POST_CALL_FAILURE_PROXY_FIELDS: Final = DERIVED_POST_CALL_FAILURE_FIELDS
DERIVED_RESPONSE_HEADERS_FIELDS: Final = DERIVED_AUTH_FIELDS


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
    "DERIVED_POST_CALL_FAILURE_PROXY_FIELDS",
    "DERIVED_POST_CALL_SUCCESS_FIELDS",
    "DERIVED_RESPONSE_HEADERS_FIELDS",
    "PRE_CALL_SOURCES",
    "limiter_request",
    "output_cap",
    "post_call_facts",
    "proxy_call_facts",
    "rate_limit_statuses",
    "request_session_id",
    "resolved_agent_id",
    "write_rate_limit_statuses",
]
