"""Shapes the ``error`` object the proxy answers with so it matches OpenAI's contract:
``type`` is a required string and ``param`` is nullable, neither of which the literal
string ``"None"`` satisfies."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from fastapi import status

from litellm.constants import STRINGIFIED_NONE
from litellm.proxy._types import ProxyException

LITELLM_CALL_ID_HEADER: Final = "x-litellm-call-id"

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
    """The HTTP status an exception carries as ``status_code`` or, the way ``ProxyException``
    stores it, as a stringified ``code``; ``default`` when it carries neither."""
    carried: Final = attribute_of(exc, "status_code")
    if isinstance(carried, int) and not isinstance(carried, bool):
        return carried
    stringified: Final = attribute_of(exc, "code")
    return int(stringified) if isinstance(stringified, str) and stringified.isdecimal() else default


def openai_error_type(exc: object, status_code: int) -> str:
    """OpenAI types ``error.type`` as a required string, so an exception carrying none
    falls back to the type its status code stands for."""
    carried: Final = attribute_of(exc, "type")
    if isinstance(carried, str) and carried != STRINGIFIED_NONE:
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
    return carried if isinstance(carried, str) and carried != STRINGIFIED_NONE else None


def litellm_call_id_headers(litellm_call_id: str | None) -> dict[str, str] | None:  # mutable-ok: ProxyException.headers
    if litellm_call_id is None:
        return None
    return {LITELLM_CALL_ID_HEADER: litellm_call_id}  # mutable-ok: ProxyException mutates its headers dict


def with_litellm_call_id(exc: ProxyException, litellm_call_id: str | None) -> ProxyException:
    """The same error object, answering with ``x-litellm-call-id`` when it was raised without one."""
    if litellm_call_id is not None:
        exc.headers.setdefault(LITELLM_CALL_ID_HEADER, litellm_call_id)
    return exc
