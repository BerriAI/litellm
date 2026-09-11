import re
from collections.abc import Mapping
from typing import Final

from litellm.constants import SESSION_ID_GENERATED_METADATA_KEY

PROVIDER_AFFINITY_REDACTED_VALUE: Final = "[REDACTED]"

_HTTP_HEADER_NAME_PATTERN: Final = re.compile(r"^[!#$%&'*+\-.^_`|~0-9A-Za-z]+$")
_FORBIDDEN_AFFINITY_HEADERS: Final = frozenset(
    {
        "api-key",
        "authorization",
        "connection",
        "content-length",
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


def _get_provider_affinity_header_name(litellm_params: object | None) -> str | None:
    header: Final = _get_value(litellm_params, "provider_affinity_header") if litellm_params is not None else None
    if header is None:
        return None
    if not isinstance(header, str):
        raise TypeError("provider_affinity_header must be a string")
    return validate_provider_affinity_header_name(header)


def get_stable_session_id(litellm_params: object | None) -> str | None:
    if litellm_params is None:
        return None

    metadata_values: Final[tuple[object, ...]] = tuple(
        value for key in ("metadata", "litellm_metadata") if (value := _get_value(litellm_params, key)) is not None
    )
    if any(
        isinstance(metadata, Mapping) and metadata.get(SESSION_ID_GENERATED_METADATA_KEY)
        for metadata in metadata_values
    ):
        return None

    for key in ("litellm_session_id", "session_id"):
        value = _get_value(litellm_params, key)
        if value:
            return str(value)

    for metadata in metadata_values:
        if isinstance(metadata, Mapping) and (value := metadata.get("session_id")):
            return str(value)
    return None


def add_provider_affinity_header(  # mutable-ok: downstream handlers add auth and signing headers
    headers: Mapping[str, object], litellm_params: object | None
) -> dict[str, object]:  # mutable-ok: downstream handlers add auth and signing headers
    header_name: Final = _get_provider_affinity_header_name(litellm_params)
    if header_name is None or any(key.lower() == header_name.lower() for key in headers):
        return dict(headers)  # mutable-ok: downstream handlers add auth and signing headers

    session_id: Final = get_stable_session_id(litellm_params)
    if session_id is None:
        return dict(headers)  # mutable-ok: downstream handlers add auth and signing headers
    if any(character in session_id for character in ("\r", "\n", "\0")):
        raise ValueError("session_id cannot contain HTTP header control characters")
    return {**headers, header_name: session_id}  # mutable-ok: downstream handlers add auth and signing headers


def redact_provider_affinity_header(  # mutable-ok: logging callbacks may enrich the payload
    headers: Mapping[str, object], litellm_params: object | None
) -> dict[str, object]:  # mutable-ok: logging callbacks may enrich the payload
    header_name: Final = _get_provider_affinity_header_name(litellm_params)
    if header_name is None:
        return dict(headers)  # mutable-ok: logging callbacks may enrich the payload
    return {  # mutable-ok: logging callbacks may enrich the returned payload
        key: PROVIDER_AFFINITY_REDACTED_VALUE if key.lower() == header_name.lower() else value
        for key, value in headers.items()
    }
