"""Typed errors raised by XctClient."""

from typing import Any, Optional


class XctError(Exception):
    """Base class. Carries the HTTP status + raw body for debugging."""

    def __init__(self, message: str, *, status: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status = status
        self.body = body


class AuthError(XctError):
    """401 or 403."""


class RateLimitError(XctError):
    """429."""


class CapabilityNotFoundError(XctError):
    """404 for entity-shaped paths."""


def from_response(status: int, body: Any) -> XctError:
    if status in (401, 403):
        return AuthError(
            _msg(body) or "Authentication failed", status=status, body=body
        )
    if status == 404:
        return CapabilityNotFoundError(
            _msg(body) or "Not found", status=status, body=body
        )
    if status == 429:
        return RateLimitError(_msg(body) or "Rate limit", status=status, body=body)
    return XctError(_msg(body) or f"HTTP {status}", status=status, body=body)


def _msg(body: Any) -> str:
    if isinstance(body, dict):
        # FastAPI default error shape; tolerate both string and array detail.
        detail = body.get("detail")
        if isinstance(detail, str):
            return detail
        if isinstance(detail, list) and detail:
            first = detail[0]
            if isinstance(first, dict):
                return str(first.get("msg") or first)
        if "error" in body:
            err = body["error"]
            return err if isinstance(err, str) else str(err)
    return ""
