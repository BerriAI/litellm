"""Reading the caller and conversation a request belongs to, off its metadata.

The proxy derives both before routing and drops them on the request: any
``x-*-session-id`` or trace-id header, and Anthropic's ``metadata.user_id``,
both land in ``metadata["session_id"]`` (``litellm_pre_call_utils``), while the
authenticated key's hash lands in ``metadata["user_api_key_hash"]``. Which of
the two metadata dicts carries them depends on the endpoint, so both are read in
the same precedence order everywhere.

The key hash is the trust boundary. A session id is caller-supplied in the end,
so anything keyed by one has to be namespaced by the authenticated key as well;
otherwise two callers reusing the same id read and write each other's entries.
"""

from collections.abc import Mapping

_METADATA_KEYS = ("litellm_metadata", "metadata")


def _metadata_dicts(request_kwargs: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    return tuple(metadata for key in _METADATA_KEYS if isinstance(metadata := request_kwargs.get(key), Mapping))


def _first_value(request_kwargs: Mapping[str, object], field: str) -> str | None:
    for metadata in _metadata_dicts(request_kwargs):
        value = metadata.get(field)
        if value is not None:
            return str(value)
    return None


def session_id_from_request(request_kwargs: Mapping[str, object]) -> str | None:
    """The conversation this request continues, or ``None`` when nothing names one."""
    return _first_value(request_kwargs, "session_id")


def user_api_key_hash_from_request(request_kwargs: Mapping[str, object]) -> str | None:
    """The authenticated caller's key hash, or ``None`` outside the proxy."""
    return _first_value(request_kwargs, "user_api_key_hash")


def session_scope(request_kwargs: Mapping[str, object], namespace: str, discriminator: str) -> str | None:
    """A cache key for per-session state, or ``None`` when the request names no session.

    ``discriminator`` separates two features keyed off the same session, and the
    caller's key hash separates two callers who picked the same session id. Falling
    back to ``unscoped`` covers direct Router use, where there is no authenticated
    caller to scope by and no cross-caller collision to prevent.
    """
    session_id = session_id_from_request(request_kwargs)
    if session_id is None:
        return None
    caller_scope = user_api_key_hash_from_request(request_kwargs) or "unscoped"
    return f"{discriminator}:v1:{namespace}:{caller_scope}:{session_id}"
