import re
from collections.abc import Mapping
from typing import Final

from litellm.constants import (
    CALLER_SESSION_ID_METADATA_KEY,
    SESSION_ID_FROM_TRACE_ID_PARAM_KEY,
    SESSION_ID_GENERATED_METADATA_KEY,
)
from litellm.litellm_core_utils.core_helpers import as_str_mapping

_HTTP_HEADER_NAME_PATTERN: Final = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_FORBIDDEN_AFFINITY_HEADERS: Final = frozenset(
    {
        "api-key",
        "authorization",
        "connection",
        "content-length",
        "content-type",
        "cookie",
        "host",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "proxy-connection",
        "set-cookie",
        "te",
        "trailer",
        "transfer-encoding",
        "upgrade",
        "www-authenticate",
        "x-api-key",
        "x-goog-api-key",
    }
)


def validate_provider_affinity_header_name(header: str) -> str:
    if not _HTTP_HEADER_NAME_PATTERN.fullmatch(header):
        raise ValueError("provider_affinity_header must be a valid HTTP header name")
    if header.lower() in _FORBIDDEN_AFFINITY_HEADERS:
        raise ValueError("provider_affinity_header cannot be an authentication, cookie, or transport header")
    return header


def _get_value(value: object, key: str) -> object | None:
    if isinstance(value, Mapping):
        return value.get(key)
    return getattr(value, key, None)


def _get_provider_affinity_header_name(litellm_params: object | None, default_header: str | None = None) -> str | None:
    header: Final = _get_value(litellm_params, "provider_affinity_header") if litellm_params is not None else None
    if header is None:
        return default_header
    if not isinstance(header, str):
        raise TypeError("provider_affinity_header must be a string")
    return validate_provider_affinity_header_name(header)


def _metadata_values(litellm_params: object) -> tuple[object, ...]:
    return tuple(
        value for key in ("metadata", "litellm_metadata") if (value := _get_value(litellm_params, key)) is not None
    )


def _non_generated_litellm_session_id(litellm_params: object) -> str | None:
    has_generated_session_id: Final = any(
        _get_value(metadata, SESSION_ID_GENERATED_METADATA_KEY) for metadata in _metadata_values(litellm_params)
    )
    litellm_session_id: Final = _get_value(litellm_params, "litellm_session_id")
    return str(litellm_session_id) if litellm_session_id and not has_generated_session_id else None


def _metadata_session_id(litellm_params: object) -> str | None:
    for metadata in _metadata_values(litellm_params):
        if not _get_value(metadata, SESSION_ID_GENERATED_METADATA_KEY) and (
            value := _get_value(metadata, "session_id")
        ):
            return str(value)
    return None


def get_caller_litellm_session_id(litellm_params: object | None) -> str | None:
    if litellm_params is None or _get_value(litellm_params, SESSION_ID_FROM_TRACE_ID_PARAM_KEY):
        return None
    return _non_generated_litellm_session_id(litellm_params)


def get_requested_session_id(litellm_params: object | None) -> str | None:
    if litellm_params is None:
        return None
    direct_session_id: Final = _get_value(litellm_params, "session_id")
    return str(direct_session_id) if direct_session_id else _metadata_session_id(litellm_params)


def get_stable_session_id(litellm_params: object | None) -> str | None:
    if litellm_params is None:
        return None
    direct_session_id: Final = _get_value(litellm_params, "session_id")
    if direct_session_id:
        return str(direct_session_id)
    return _non_generated_litellm_session_id(litellm_params) or _metadata_session_id(litellm_params)


def get_caller_session_id(litellm_params: object | None) -> str | None:
    """Resolve a caller conversation without accepting a generated session or a promoted trace id."""
    if litellm_params is None:
        return None
    metadata_values: Final = _metadata_values(litellm_params)
    preserved_session: Final = next(
        (
            value
            for metadata in metadata_values
            if isinstance(value := _get_value(metadata, CALLER_SESSION_ID_METADATA_KEY), str) and value
        ),
        None,
    )
    if preserved_session:
        return preserved_session
    direct_session: Final = _get_value(litellm_params, "session_id")
    if isinstance(direct_session, str) and direct_session:
        return direct_session

    proxy_request: Final = _get_value(litellm_params, "proxy_server_request")
    headers: Final = as_str_mapping(_get_value(proxy_request, "headers"))
    trace_header: Final = next(
        (value for key, value in (headers or {}).items() if key.lower() == "x-litellm-trace-id"), None
    )
    candidates: Final = (
        get_caller_litellm_session_id(litellm_params),
        *(
            _get_value(metadata, "session_id")
            for metadata in metadata_values
            if not _get_value(metadata, SESSION_ID_GENERATED_METADATA_KEY)
        ),
    )
    return next((value for value in candidates if isinstance(value, str) and value and value != trace_header), None)


def add_provider_affinity_header(
    headers: Mapping[str, object], litellm_params: object | None, default_header: str | None = None
) -> dict[str, object]:  # mutable-ok: downstream handlers add auth and signing headers
    header_name: Final = _get_provider_affinity_header_name(litellm_params, default_header)
    if header_name is None or any(key.lower() == header_name.lower() for key in headers):
        return dict(headers)

    session_id: Final = get_stable_session_id(litellm_params)
    if session_id is None:
        return dict(headers)
    if any(character in session_id for character in ("\r", "\n", "\0")):
        raise ValueError("session_id cannot contain HTTP header control characters")
    return {**headers, header_name: session_id}
