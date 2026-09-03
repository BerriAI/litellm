"""Shapes the ``error`` object the proxy answers with so it matches OpenAI's contract:
``type`` is a required string and ``param`` is nullable, neither of which the literal
string ``"None"`` satisfies."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from fastapi import HTTPException, status

from litellm.proxy._types import ProxyException

_OPENAI_ERROR_TYPE_BY_STATUS: Final[Mapping[int, str]] = MappingProxyType(
    {
        status.HTTP_401_UNAUTHORIZED: "authentication_error",
        status.HTTP_403_FORBIDDEN: "permission_error",
        status.HTTP_429_TOO_MANY_REQUESTS: "rate_limit_error",
    }
)


def attribute_of(value: object, name: str, default: object = None) -> object:
    return getattr(value, name, default)


def error_status_code(exc: object, default: int) -> int:
    """The HTTP status an exception carries, or ``default`` when it carries none."""
    carried: Final = attribute_of(exc, "status_code")
    return carried if isinstance(carried, int) and not isinstance(carried, bool) else default


def openai_error_type(exc: object, status_code: int) -> str:
    """OpenAI types ``error.type`` as a required string, so an exception carrying none
    falls back to the type its status code stands for."""
    carried: Final = attribute_of(exc, "type")
    if isinstance(carried, str):
        return carried
    mapped: Final = _OPENAI_ERROR_TYPE_BY_STATUS.get(status_code)
    if mapped is not None:
        return mapped
    if status_code < status.HTTP_500_INTERNAL_SERVER_ERROR:
        return "invalid_request_error"
    return "internal_server_error"


def openai_error_param(exc: object) -> str | None:
    """OpenAI types ``error.param`` as nullable, so an exception carrying none
    serializes as JSON ``null``."""
    carried: Final = attribute_of(exc, "param")
    return carried if isinstance(carried, str) else None


def proxy_exception_for(exc: Exception, default_status_code: int) -> ProxyException:
    """The answer a route's blanket ``except Exception`` owes its caller.

    Such a handler has no idea what failed, so it reads the type off the exception
    instead of asserting one, and keeps the status each branch already answered with:
    an ``HTTPException`` carries its own, anything else gets the route's default."""
    if isinstance(exc, ProxyException):
        return exc
    if isinstance(exc, HTTPException):
        carried_status: Final = error_status_code(exc, default_status_code)
        return ProxyException(
            message=str(attribute_of(exc, "detail", exc)),
            type=openai_error_type(exc, carried_status),
            param=openai_error_param(exc),
            code=carried_status,
        )
    return ProxyException(
        message=str(exc),
        type=openai_error_type(exc, default_status_code),
        param=openai_error_param(exc),
        code=default_status_code,
    )
