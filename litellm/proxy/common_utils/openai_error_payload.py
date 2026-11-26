"""Shapes the ``error`` object the proxy answers with so it matches OpenAI's contract:
``type`` is a required string and ``param`` is nullable, neither of which the literal
string ``"None"`` satisfies."""

import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from fastapi import status
from pydantic import TypeAdapter

from litellm._logging import (
    redact_internal_details_from_client_fields,
    redact_internal_details_from_client_message,
)
from litellm.constants import PROVIDER_ERROR_BODY_ATTRIBUTE, STRINGIFIED_NONE
from litellm.litellm_core_utils.core_helpers import http_error_status
from litellm.proxy._types import ProxyException

LITELLM_CALL_ID_HEADER: Final = "x-litellm-call-id"

_OPENAI_ERROR_TYPE_BY_STATUS: Final[Mapping[int, str]] = MappingProxyType(
    {
        status.HTTP_401_UNAUTHORIZED: "authentication_error",
        status.HTTP_403_FORBIDDEN: "permission_error",
        status.HTTP_429_TOO_MANY_REQUESTS: "rate_limit_error",
    }
)

_ERROR_BODY: Final = TypeAdapter(dict[str, object])

_NO_PROVIDER_SPECIFIC_FIELDS: Final[Mapping[str, object]] = MappingProxyType({})

_OPENAI_ERROR_TYPE_SHAPE: Final = re.compile(r"[a-z][a-z0-9_]*")

_OPENAI_ERROR_OBJECT_FIELDS: Final = frozenset(
    {"object", "message", "type", "param", "code", "provider_specific_fields"}
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
    """OpenAI types ``error.type`` as a required snake_case string, so an exception carrying none,
    or a backend's own class name such as vLLM's ``BadRequestError``, falls back to the type its
    status code stands for."""
    carried: Final = _relayable_text(attribute_of(exc, "type"))
    if _OPENAI_ERROR_TYPE_SHAPE.fullmatch(carried):
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
    return _relayable_text(attribute_of(exc, "param")) or None


def redacted_raw_error_param(exc: object) -> str | None:
    carried: Final = attribute_of(exc, "param", STRINGIFIED_NONE)
    return redact_internal_details_from_client_message(carried) if isinstance(carried, str) else None


def _relayable_text(carried: object) -> str:
    if not isinstance(carried, str) or carried == STRINGIFIED_NONE:
        return ""
    return redact_internal_details_from_client_message(carried).strip()


def relayed_provider_specific_fields(
    exc: object,
) -> dict[str, object] | None:  # mutable-ok: ProxyException.provider_specific_fields is a dict
    carried: Final = attribute_of(exc, "provider_specific_fields")
    relayed: Final = {
        **_error_body_provider_specific_fields(exc),
        **(_ERROR_BODY.validate_python(carried) if isinstance(carried, dict) else {}),
    }
    return redacted_provider_specific_fields(relayed) if relayed else None


def redacted_provider_specific_fields(
    carried: object,
) -> dict[str, object] | None:  # mutable-ok: ProxyException.provider_specific_fields is a dict
    if not isinstance(carried, dict):
        return None
    redacted: Final = _ERROR_BODY.validate_python(
        redact_internal_details_from_client_fields(_ERROR_BODY.validate_python(carried))
    )
    return redacted if _is_json_compliant(redacted) else None


def _error_body_provider_specific_fields(exc: object) -> Mapping[str, object]:
    body: Final = attribute_of(exc, "body")
    carried: Final = body if isinstance(body, dict) else attribute_of(exc, PROVIDER_ERROR_BODY_ATTRIBUTE)
    if not isinstance(carried, dict):
        return _NO_PROVIDER_SPECIFIC_FIELDS
    error_object: Final = _ERROR_BODY.validate_python(carried)
    declared: Final = error_object.get("provider_specific_fields")
    if isinstance(declared, dict):
        return _ERROR_BODY.validate_python(declared)
    return _openai_error_object_extras(error_object)


def _is_json_compliant(fields: Mapping[str, object]) -> bool:
    try:
        _ = json.dumps(fields, allow_nan=False)
    except (TypeError, ValueError, RecursionError):
        return False
    return True


def _openai_error_object_extras(error_object: Mapping[str, object]) -> Mapping[str, object]:
    relayable: Final = error_object.keys() if isinstance(error_object.get("message"), str) else {"code"}
    return MappingProxyType(
        {key: value for key, value in error_object.items() if key in relayable and _is_provider_specific(key, value)}
    )


def _is_provider_specific(key: str, value: object) -> bool:
    if key == "code":
        return value not in (None, "") and http_error_status(value) is None
    return key not in _OPENAI_ERROR_OBJECT_FIELDS


def litellm_call_id_headers(litellm_call_id: str | None) -> dict[str, str] | None:  # mutable-ok: ProxyException.headers
    if litellm_call_id is None:
        return None
    return {LITELLM_CALL_ID_HEADER: litellm_call_id}


def failed_call_headers(
    litellm_call_id: str | None, response_cost: float
) -> dict[str, str]:  # mutable-ok: ProxyException.headers
    cost_headers: Final = {"x-litellm-response-cost": str(response_cost)} if response_cost > 0 else {}
    return {**(litellm_call_id_headers(litellm_call_id) or {}), **cost_headers}


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
