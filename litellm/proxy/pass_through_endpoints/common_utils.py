from collections.abc import Iterable, Mapping
from functools import lru_cache
from typing import Final

from fastapi import Request

from litellm._logging import verbose_proxy_logger
from litellm.types.passthrough_endpoints.pass_through_endpoints import (
    PassThroughAuthMode,
    pass_through_auth_mode,
)

PASS_THROUGH_MISSING_AUTH_WARNING: Final = (
    "pass_through_endpoints entry %r sets no `auth`: any valid LiteLLM key may call it. "
    "Set `auth: true` to restrict it to keys granted `allowed_passthrough_routes`, "
    "or `auth: false` to serve it without a key. "
    "A future release will treat a missing `auth` as `auth: true`."
)

PASS_THROUGH_INVALID_AUTH_WARNING: Final = (
    "pass_through_endpoints entry %r sets `auth: %s`, which is not a boolean, "
    "so any valid LiteLLM key may call it. "
    "Set `auth: true` to restrict it to keys granted `allowed_passthrough_routes`, "
    "or `auth: false` to serve it without a key."
)


@lru_cache(maxsize=256)
def _warn_once_per_process(path: str, auth: str | None) -> None:
    if auth is None:
        verbose_proxy_logger.warning(PASS_THROUGH_MISSING_AUTH_WARNING, path)
        return
    verbose_proxy_logger.warning(PASS_THROUGH_INVALID_AUTH_WARNING, path, auth)


def _unenforced_auth(entry: Mapping[str, object]) -> tuple[str, str | None] | None:
    auth: Final = entry.get("auth")
    if pass_through_auth_mode(auth) is not PassThroughAuthMode.ANY_KEY:
        return None
    return (str(entry.get("path")), None if auth is None else repr(auth))


def warn_pass_through_entries_without_auth(entries: Iterable[Mapping[str, object]]) -> None:
    for path, auth in filter(None, map(_unenforced_auth, entries)):
        _warn_once_per_process(path, auth)


def get_litellm_virtual_key(request: Request) -> str:
    """
    Extract and format API key from request headers.
    Prioritizes x-litellm-api-key over Authorization header.


    Vertex JS SDK uses `Authorization` header, we use `x-litellm-api-key` to pass litellm virtual key

    """
    litellm_api_key: Final = request.headers.get("x-litellm-api-key")
    if litellm_api_key:
        return f"Bearer {litellm_api_key}"
    return request.headers.get("Authorization", "")
