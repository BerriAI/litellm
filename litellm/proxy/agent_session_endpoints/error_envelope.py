"""
Standardized error envelope for the /v2 agent/session/run API.

Cursor's API and the Cursor SDK expect every error response to look like::

    {"error": {"code": "...", "message": "...", "status": <int>, "details"?: ...}}

FastAPI's defaults are ``{"detail": "..."}`` for HTTPException and
``{"detail": [{"loc": ..., "msg": ..., "type": ...}]}`` for ValidationError.
We translate both into the common envelope so SDK consumers can rely on
a single error shape.

Scoping: the envelope only applies to requests under ``/v2/...`` —
non-v2 endpoints keep FastAPI's defaults so we don't accidentally
reshape error responses for the rest of the proxy. Each handler
short-circuits when the request path doesn't match.
"""

from http import HTTPStatus
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import ORJSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

V2_PATH_PREFIX = "/v2/"


def _is_v2_request(request: Request) -> bool:
    """True iff the request path is under the /v2 namespace."""
    return request.url.path.startswith(V2_PATH_PREFIX)


def _http_status_code_to_name(status_code: int) -> str:
    """Map an HTTP status code to a snake_case error code.

    Examples: 404 -> "not_found", 409 -> "conflict", 503 ->
    "service_unavailable". Falls back to ``"http_<code>"`` for codes
    HTTPStatus doesn't recognize.
    """
    try:
        phrase = HTTPStatus(status_code).phrase  # e.g. "Not Found"
    except ValueError:
        return f"http_{status_code}"
    return phrase.lower().replace(" ", "_").replace("-", "_")


def _envelope(
    code: str,
    message: str,
    status_code: int,
    details: Optional[Any] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "code": code,
        "message": message,
        "status": status_code,
    }
    if details is not None:
        payload["details"] = details
    return {"error": payload}


def _coerce_detail_to_message(detail: Any) -> str:
    """Pull a human-readable string out of FastAPI's polymorphic ``detail``.

    Detail can be a plain string, a dict ``{"error": "..."}``, or a list
    of validation errors. We collapse list/dict shapes to a short
    one-liner so the SDK gets a single ``message`` field even when the
    underlying handler raised something exotic.
    """
    if detail is None:
        return ""
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict):
        # Common alternate shape: {"error": "..."}.
        for k in ("error", "message", "msg"):
            v = detail.get(k)
            if isinstance(v, str) and v:
                return v
        return str(detail)
    if isinstance(detail, list):
        # Validation-error-style list of {loc, msg, type} dicts.
        msgs = []
        for entry in detail:
            if isinstance(entry, dict) and "msg" in entry:
                msgs.append(str(entry["msg"]))
        if msgs:
            return "; ".join(msgs)
        return str(detail)
    return str(detail)


async def http_exception_handler(
    request: Request,
    exc: StarletteHTTPException,
) -> ORJSONResponse:
    """Wrap any HTTPException raised under /v2/ in the standard envelope.

    Non-v2 paths fall back to FastAPI's default response shape so we
    don't disturb other proxy endpoints.
    """
    if not _is_v2_request(request):
        # Mirror FastAPI's default for non-v2 paths.
        return ORJSONResponse(
            {"detail": exc.detail},
            status_code=exc.status_code,
            headers=getattr(exc, "headers", None),
        )

    code = _http_status_code_to_name(exc.status_code)
    message = _coerce_detail_to_message(exc.detail) or code
    return ORJSONResponse(
        _envelope(code, message, exc.status_code),
        status_code=exc.status_code,
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(
    request: Request,
    exc: RequestValidationError,
) -> ORJSONResponse:
    """Wrap 422 ValidationError raised under /v2/ in the standard envelope.

    The original list of ``{loc, msg, type}`` dicts is preserved under
    ``error.details`` so SDK callers can render per-field errors.
    """
    raw_errors: List[Dict[str, Any]] = exc.errors()
    if not _is_v2_request(request):
        # Default FastAPI shape.
        return ORJSONResponse(
            {"detail": raw_errors},
            status_code=422,
        )
    return ORJSONResponse(
        _envelope(
            code="validation_error",
            message="Request validation failed",
            status_code=422,
            details=raw_errors,
        ),
        status_code=422,
    )


def register_v2_exception_handlers(app: FastAPI) -> None:
    """Install the /v2 error-envelope handlers on ``app``.

    Idempotent: re-registering replaces the previous handler. Call this
    after ``app.include_router(...)`` for the v2 routers and after any
    other handlers the proxy installs.
    """
    app.add_exception_handler(StarletteHTTPException, http_exception_handler)
    app.add_exception_handler(RequestValidationError, validation_exception_handler)
