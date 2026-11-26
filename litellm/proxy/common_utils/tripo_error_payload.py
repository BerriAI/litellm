"""Tripo-shaped client errors for the native passthrough routes.

Auth and body parsing fail in the shared proxy gate, before a Tripo route runs.
Only the missing-key, unknown-key, and malformed-JSON cases are rewritten, and
only on ``/tripo/`` and ``/cn.tripo/``. Budget, rate-limit, and model-access
failures keep the proxy envelope so other callers of those checks stay put.
"""

from secrets import token_hex
from typing import Final

from fastapi import Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict
from typing_extensions import ReadOnly, TypedDict

from litellm.proxy._types import ProxyErrorTypes, ProxyException
from litellm.proxy.auth.auth_utils import get_request_route

_AUTH_REQUIRED_MESSAGES: Final = (
    "Authentication Error, No api key passed in.",
    "Authentication Error, Malformed API Key passed in.",
)
_UNKNOWN_KEY_MESSAGE: Final = "Authentication Error, Invalid proxy server token passed."
_MALFORMED_JSON_MESSAGE: Final = "Invalid JSON payload"


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


def tripo_error_json_response(exc: ProxyException, request: Request) -> JSONResponse:
    from litellm.proxy.proxy_server import (
        _close_dangling_otel_server_span,  # pyright: ignore[reportPrivateUsage]  # proxy_server keeps the span-close helper private; a JSONResponse returned instead of a raised error must stamp the OTel server span like the global ProxyException handler does
    )

    fault: Final = _fault_for_proxy_exception(exc)
    if fault is None:
        status_code: Final = int(exc.code) if exc.code.isdigit() else status.HTTP_500_INTERNAL_SERVER_ERROR
        _close_dangling_otel_server_span(request, status_code, exc=exc)
        return JSONResponse(status_code=status_code, content={"error": exc.to_dict()}, headers=exc.headers)
    _close_dangling_otel_server_span(request, fault.http_status, exc=exc)
    return tripo_error_response(fault, exc.headers)


def _fault_for_proxy_exception(exc: ProxyException) -> _TripoClientFault | None:
    status_code: Final = int(exc.code) if exc.code.isdigit() else 0
    if exc.type == ProxyErrorTypes.token_not_found_in_db and status_code == status.HTTP_401_UNAUTHORIZED:
        return _AUTH_FAILED
    if exc.type == ProxyErrorTypes.auth_error and status_code == status.HTTP_401_UNAUTHORIZED:
        return _auth_error_fault(exc.message)
    if (
        exc.type == "invalid_request_error"
        and status_code == status.HTTP_400_BAD_REQUEST
        and exc.message.startswith(_MALFORMED_JSON_MESSAGE)
    ):
        return _MALFORMED_BODY
    return None


def _auth_error_fault(message: str) -> _TripoClientFault | None:
    if message.startswith(_AUTH_REQUIRED_MESSAGES):
        return _AUTH_REQUIRED
    if message.startswith(_UNKNOWN_KEY_MESSAGE):
        return _AUTH_FAILED
    return None
