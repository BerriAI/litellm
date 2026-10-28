import re
from collections.abc import Mapping
from typing import Final

from litellm.constants import SESSION_ID_GENERATED_METADATA_KEY

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
        isinstance(metadata, Mapping) and metadata.get(SESSION_ID_GENERATED_METADATA_KEY)
        for metadata in _metadata_values(litellm_params)
    )
    litellm_session_id: Final = _get_value(litellm_params, "litellm_session_id")
    return str(litellm_session_id) if litellm_session_id and not has_generated_session_id else None


def _metadata_session_id(litellm_params: object) -> str | None:
    for metadata in _metadata_values(litellm_params):
        if (
            isinstance(metadata, Mapping)
            and not metadata.get(SESSION_ID_GENERATED_METADATA_KEY)
            and (value := metadata.get("session_id"))
        ):
            return str(value)
    return None


def _metadata_field_equals(litellm_params: object, field: str, value: str) -> bool:
    return any(_get_value(metadata, field) == value for metadata in _metadata_values(litellm_params))


def get_caller_litellm_session_id(litellm_params: object | None) -> str | None:
    if litellm_params is None:
        return None
    session_id: Final = _non_generated_litellm_session_id(litellm_params)
    copies_trace_id: Final = session_id is not None and (
        _metadata_field_equals(litellm_params, "trace_id", session_id)
        and not _metadata_field_equals(litellm_params, "session_id", session_id)
    )
    return None if copies_trace_id else session_id


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
