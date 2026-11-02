"""Shapes the ``error`` object the proxy answers with so it matches OpenAI's contract:
``type`` is a required string and ``param`` is nullable, neither of which the literal
string ``"None"`` satisfies."""

import sys
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from fastapi import HTTPException, status
from pydantic import TypeAdapter

from litellm._logging import redact_internal_details_from_client_fields
from litellm.constants import STRINGIFIED_NONE
from litellm.proxy._types import ProxyErrorTypes, ProxyException, proxy_exception_status_code
from litellm.proxy.db.db_lookup_gate import DBLookupDeadlineExceeded
from litellm.proxy.db.exception_handler import PrismaDBExceptionHandler, exception_chain

LITELLM_CALL_ID_HEADER: Final = "x-litellm-call-id"

_OPENAI_ERROR_TYPE_BY_STATUS: Final[Mapping[int, str]] = MappingProxyType(
    {
        status.HTTP_401_UNAUTHORIZED: "authentication_error",
        status.HTTP_403_FORBIDDEN: "permission_error",
        status.HTTP_429_TOO_MANY_REQUESTS: "rate_limit_error",
    }
)

_AUTH_FAILURE_STATUS_CODES: Final = frozenset({status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN})

_HTTP_ERROR_STATUSES: Final = range(status.HTTP_400_BAD_REQUEST, 600)

_ERROR_BODY: Final = TypeAdapter(dict[str, object])

_FAILOVER_SQLSTATES: Final = frozenset({"25006", "57P01", "57P02", "57P03"})

_FAILOVER_REFUSALS: Final = (
    "is not currently accepting connections",
    "the database system is starting up",
    "the database system is shutting down",
    "the database system is in recovery mode",
    "the database system is not yet accepting connections",
    "terminating connection due to administrator command",
)


def attribute_of(value: object, name: str, default: object = None) -> object:
    return getattr(value, name, default)


def error_status_code(exc: object, default: int) -> int:
    """The 4xx or 5xx HTTP status an exception carries as ``status_code`` (an int or a decimal string) or,
    the way ``ProxyException`` stores it, as a stringified ``code``; ``default`` when it carries neither."""
    carried: Final = _http_error_status(attribute_of(exc, "status_code"))
    if carried is not None:
        return carried
    stringified: Final = attribute_of(exc, "code")
    from_code: Final = _http_error_status(stringified) if isinstance(stringified, str) else None
    return default if from_code is None else from_code


def _http_error_status(value: object) -> int | None:
    match value:
        case bool():
            return None
        case int() if value in _HTTP_ERROR_STATUSES:
            return value
        case str() if len(value) == 3 and value.isascii() and value.isdigit() and int(value) in _HTTP_ERROR_STATUSES:
            return int(value)
        case _:
            return None


def openai_error_type(exc: object, status_code: int) -> str:
    """OpenAI types ``error.type`` as a required string, so an exception carrying none
    falls back to the type its status code stands for."""
    carried: Final = attribute_of(exc, "type")
    if isinstance(carried, str) and carried != STRINGIFIED_NONE:
        return carried
    return openai_error_type_for_status(status_code)


def openai_error_type_for_status(status_code: int) -> str:
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


def relayed_provider_specific_fields(
    exc: object,
) -> dict[str, object] | None:  # mutable-ok: ProxyException.provider_specific_fields is a dict
    body: Final = attribute_of(exc, "body")
    if not isinstance(body, dict):
        return None
    fields: Final = _ERROR_BODY.validate_python(body).get("provider_specific_fields")
    if not isinstance(fields, dict):
        return None
    return _ERROR_BODY.validate_python(redact_internal_details_from_client_fields(_ERROR_BODY.validate_python(fields)))


def proxy_exception_for(exc: Exception, default_status_code: int) -> ProxyException:
    match exc:
        case ProxyException():
            return exc
        case _ if _is_database_outage(exc):
            return ProxyException(
                message=PrismaDBExceptionHandler.database_unavailable_message(exc),
                type=ProxyErrorTypes.no_db_connection.value,
                param=None,
                code=status.HTTP_503_SERVICE_UNAVAILABLE,
            )
        case HTTPException(status_code=status_code, detail=detail):
            return openai_shaped_proxy_exception(exc, str(detail), status_code)
        case _:
            return openai_shaped_proxy_exception(
                exc, str(exc) or type(exc).__name__, error_status_code(exc, default_status_code)
            )


def _is_database_outage(exc: Exception) -> bool:
    return any(map(_is_database_outage_link, exception_chain(exc)))


def _is_database_outage_link(exc: BaseException) -> bool:
    match exc:
        case DBLookupDeadlineExceeded():
            return True
        case Exception() if _prisma_is_loaded():
            return _is_prisma_outage(exc)
        case _:
            return False


def _prisma_is_loaded() -> bool:
    return sys.modules.get("prisma") is not None


def _is_prisma_outage(exc: Exception) -> bool:
    if PrismaDBExceptionHandler.is_prisma_engine_internal_error(exc):
        return True
    if type(exc).__module__.partition(".")[0] != "prisma":
        return False
    return PrismaDBExceptionHandler.is_database_service_unavailable_error(exc) or _is_failover_refusal(exc)


def _is_failover_refusal(exc: Exception) -> bool:
    if PrismaDBExceptionHandler.is_read_only_transaction_error(exc):
        return True
    if PrismaDBExceptionHandler.postgres_sqlstate(exc) in _FAILOVER_SQLSTATES:
        return True
    message: Final = str(exc).lower()
    return any(refusal in message for refusal in _FAILOVER_REFUSALS)


def answered_status_code(message: str, status_code: int) -> int:
    return int(proxy_exception_status_code(message, str(status_code)))


def route_failure_proxy_exception(
    exc: Exception,
    message: str,
    default_status_code: int,
    headers: dict[str, str] | None = None,  # mutable-ok: ProxyException.headers
    openai_code: str | None = None,
) -> ProxyException:
    raised_status: Final = error_status_code(exc, default_status_code)
    answered_status: Final = answered_status_code(message, raised_status)
    return ProxyException(
        message=message,
        type=_route_failure_error_type(exc, raised_status, answered_status),
        param=openai_error_param(exc),
        headers=headers,
        openai_code=openai_code,
        code=answered_status,
    )


def _route_failure_error_type(exc: Exception, raised_status: int, answered_status: int) -> str:
    if answered_status != raised_status and answered_status in _AUTH_FAILURE_STATUS_CODES:
        return ProxyErrorTypes.auth_error.value
    return openai_error_type(exc, answered_status)


def openai_shaped_proxy_exception(exc: Exception, message: str, status_code: int) -> ProxyException:
    answered_status: Final = answered_status_code(message, status_code)
    return _proxy_exception(exc, message, answered_status, _catch_all_error_type(exc, answered_status))


def status_typed_proxy_exception(exc: HTTPException) -> ProxyException:
    message: Final = str(exc.detail)
    answered_status: Final = answered_status_code(message, exc.status_code)
    return _proxy_exception(exc, message, answered_status, openai_error_type(exc, answered_status))


def _proxy_exception(exc: Exception, message: str, answered_status: int, error_type: str) -> ProxyException:
    return ProxyException(message=message, type=error_type, param=openai_error_param(exc), code=answered_status)


def _catch_all_error_type(exc: Exception, status_code: int) -> str:
    if _rejects_the_caller(exc, status_code):
        return ProxyErrorTypes.auth_error.value
    return openai_error_type(exc, status_code)


def _rejects_the_caller(exc: Exception, status_code: int) -> bool:
    raised_as_http_answer: Final = isinstance(exc, HTTPException)
    status_from_upstream: Final = error_status_code(exc, 0) == status_code
    return status_code in _AUTH_FAILURE_STATUS_CODES and (raised_as_http_answer or not status_from_upstream)


def litellm_call_id_headers(litellm_call_id: str | None) -> dict[str, str] | None:  # mutable-ok: ProxyException.headers
    if litellm_call_id is None:
        return None
    return {LITELLM_CALL_ID_HEADER: litellm_call_id}


def with_litellm_call_id(exc: ProxyException, litellm_call_id: str | None) -> ProxyException:
    """The same error object, answering with ``x-litellm-call-id`` when it was raised without one."""
    if litellm_call_id is not None:
        exc.headers.setdefault(LITELLM_CALL_ID_HEADER, litellm_call_id)
    return exc


def headers_with_litellm_call_id(headers: Mapping[str, str] | None, litellm_call_id: str) -> Mapping[str, str]:
    """``headers`` plus ``x-litellm-call-id``, keeping the value they already carry under that name."""
    if headers is None:
        return MappingProxyType({LITELLM_CALL_ID_HEADER: litellm_call_id})
    return MappingProxyType({LITELLM_CALL_ID_HEADER: litellm_call_id, **headers})
