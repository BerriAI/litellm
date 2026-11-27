"""Tripo-shaped client errors for the native passthrough routes.

The shared gate still raises the proxy exception. Only ``/tripo/`` and
``/cn.tripo/`` rewrite the response. Codes 9101-9105 are local: Tripo's catalog
has no budget, blocked-key, expired-key, model-access, or proxy rate-limit code.
Other routes keep the envelope they already return.
"""

from secrets import token_hex
from typing import Final

from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from starlette.exceptions import HTTPException as StarletteHTTPException
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.auth_utils import get_request_route
from litellm.proxy.common_utils.proxy_rate_limit_error import ProxyRateLimitError

_AUTH_REQUIRED_MESSAGES: Final = (
    "Authentication Error, No api key passed in.",
    "Authentication Error, Malformed API Key passed in.",
)
_UNKNOWN_KEY_MESSAGE: Final = "Authentication Error, Invalid proxy server token passed."
_BLOCKED_KEY_MESSAGE: Final = "Authentication Error, Key is blocked"
_MALFORMED_JSON_MESSAGE: Final = "Invalid JSON payload"
_BUDGET_CODE: Final = 9101
_BLOCKED_CODE: Final = 9102
_EXPIRED_CODE: Final = 9103
_MODEL_ACCESS_CODE: Final = 9104
_RATE_LIMIT_CODE: Final = 9105
_BUDGET_SUGGESTION: Final = "Check the configured budget"
_BLOCKED_SUGGESTION: Final = "Unblock the key or use a different key"
_EXPIRED_SUGGESTION: Final = "Use a key that has not expired"
_MODEL_ACCESS_SUGGESTION: Final = "Grant this caller access to the model"
_RATE_LIMIT_SUGGESTION: Final = "Retry after the limit resets"
_MODEL_ACCESS_DENIED: Final = frozenset(
    (
        ProxyErrorTypes.key_model_access_denied,
        ProxyErrorTypes.team_model_access_denied,
        ProxyErrorTypes.user_model_access_denied,
        ProxyErrorTypes.customer_model_access_denied,
        ProxyErrorTypes.org_model_access_denied,
        ProxyErrorTypes.project_model_access_denied,
        ProxyErrorTypes.agent_model_access_denied,
    )
)


class TripoErrorContent(TypedDict):
    code: ReadOnly[int]
    status: ReadOnly[str]
    message: ReadOnly[str]
    suggestion: ReadOnly[str]
    request_id: ReadOnly[str]


class _TripoClientFault(BaseModel):
    model_config = ConfigDict(frozen=True)

    http_status: int
    code: int
    message: str
    suggestion: str


_AUTH_REQUIRED: Final = _TripoClientFault(
    http_status=status.HTTP_401_UNAUTHORIZED,
    code=2,
    message="Authentication required",
    suggestion="Check the Bearer API key",
)
_AUTH_FAILED: Final = _TripoClientFault(
    http_status=status.HTTP_401_UNAUTHORIZED,
    code=1002,
    message="Authentication failed",
    suggestion="Check if your credentials is valid, and ensure you set it correctly",
)
_MALFORMED_BODY: Final = _TripoClientFault(
    http_status=status.HTTP_400_BAD_REQUEST,
    code=1003,
    message="The request body is malformed",
    suggestion="Refer to the API documentation for request schema",
)


def is_tripo_pass_through_route(route: str) -> bool:
    return route.startswith("/tripo/") or route.startswith("/cn.tripo/")


def tripo_error_content(fault: _TripoClientFault, request_id: str) -> TripoErrorContent:
    return {
        "code": fault.code,
        "status": "error",
        "message": fault.message,
        "suggestion": fault.suggestion,
        "request_id": request_id,
    }


def tripo_error_response(fault: _TripoClientFault, headers: dict[str, str] | None = None) -> JSONResponse:
    return JSONResponse(
        status_code=fault.http_status,
        content=dict(tripo_error_content(fault, token_hex(16))),
        headers=headers,
    )


def uses_tripo_error_envelope(request: Request, exc: ProxyException) -> bool:
    return is_tripo_pass_through_route(get_request_route(request)) and _fault_for_proxy_exception(exc) is not None


def uses_tripo_rate_limit_envelope(request: Request, exc: StarletteHTTPException) -> bool:
    return isinstance(exc, ProxyRateLimitError) and is_tripo_pass_through_route(get_request_route(request))


def tripo_rate_limit_json_response(request: Request, exc: StarletteHTTPException) -> JSONResponse:
    detail: Final = exc.detail
    fault: Final = _TripoClientFault(
        http_status=exc.status_code,
        code=_RATE_LIMIT_CODE,
        message=detail if isinstance(detail, str) else str(detail),
        suggestion=_RATE_LIMIT_SUGGESTION,
    )
    return _answered(request, fault, _string_headers(exc), exc)


def tripo_error_json_response(exc: ProxyException, request: Request) -> JSONResponse:
    fault: Final = _fault_for_proxy_exception(exc)
    if fault is None:
        status_code: Final = int(exc.code) if exc.code.isdigit() else status.HTTP_500_INTERNAL_SERVER_ERROR
        _close_tripo_error_span(request, status_code, exc)
        return JSONResponse(status_code=status_code, content={"error": exc.to_dict()}, headers=exc.headers)
    return _answered(request, fault, exc.headers, exc)


def _fault_for_proxy_exception(exc: ProxyException) -> _TripoClientFault | None:
    status_code: Final = int(exc.code) if exc.code.isdigit() else 0
    if exc.type == ProxyErrorTypes.budget_exceeded:
        return _gate_fault(exc, _BUDGET_CODE, _BUDGET_SUGGESTION)
    if exc.type == ProxyErrorTypes.expired_key:
        return _gate_fault(exc, _EXPIRED_CODE, _EXPIRED_SUGGESTION)
    if exc.type in _MODEL_ACCESS_DENIED:
        return _gate_fault(exc, _MODEL_ACCESS_CODE, _MODEL_ACCESS_SUGGESTION)
    if exc.type == ProxyErrorTypes.token_not_found_in_db and status_code == status.HTTP_401_UNAUTHORIZED:
        return _AUTH_FAILED
    if exc.type == ProxyErrorTypes.auth_error and status_code == status.HTTP_401_UNAUTHORIZED:
        return _auth_error_fault(exc)
    if (
        exc.type == "invalid_request_error"
        and status_code == status.HTTP_400_BAD_REQUEST
        and exc.message.startswith(_MALFORMED_JSON_MESSAGE)
    ):
        return _MALFORMED_BODY
    return None


def _auth_error_fault(exc: ProxyException) -> _TripoClientFault | None:
    if exc.message.startswith(_BLOCKED_KEY_MESSAGE):
        return _gate_fault(exc, _BLOCKED_CODE, _BLOCKED_SUGGESTION)
    if exc.message.startswith(_AUTH_REQUIRED_MESSAGES):
        return _AUTH_REQUIRED
    if exc.message.startswith(_UNKNOWN_KEY_MESSAGE):
        return _AUTH_FAILED
    return None


def _gate_fault(exc: ProxyException, code: int, suggestion: str) -> _TripoClientFault:
    status_code: Final = int(exc.code) if exc.code.isdigit() else status.HTTP_500_INTERNAL_SERVER_ERROR
    return _TripoClientFault(http_status=status_code, code=code, message=exc.message, suggestion=suggestion)


def _string_headers(exc: StarletteHTTPException) -> dict[str, str] | None:
    if not exc.headers:
        return None
    return {key: value for key, value in exc.headers.items()}


def _answered(
    request: Request, fault: _TripoClientFault, headers: dict[str, str] | None, exc: Exception
) -> JSONResponse:
    _close_tripo_error_span(request, fault.http_status, exc)
    return tripo_error_response(fault, headers)


def _close_tripo_error_span(request: Request, status_code: int, exc: Exception) -> None:
    from litellm.proxy.proxy_server import (
        _close_dangling_otel_server_span,  # pyright: ignore[reportPrivateUsage]  # proxy_server keeps the span-close helper private; a JSONResponse returned instead of a raised error must stamp the OTel server span like the global ProxyException handler does
    )

    _close_dangling_otel_server_span(request, status_code, exc=exc)
