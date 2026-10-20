"""
Constants for Copilot integration
"""

import asyncio
import hashlib
import os
from collections.abc import Callable, Mapping, MutableMapping
from functools import partial
from typing import (
    Final,
    Protocol,
    cast,  # noqa: TID251  # casts pin the untyped litellm clients and cache to the local Protocols
)
from urllib.parse import urlsplit
from uuid import uuid4

import httpx
from pydantic import ConfigDict, TypeAdapter, with_config
from typing_extensions import NotRequired, ReadOnly, TypedDict

from litellm.caching.in_memory_cache import InMemoryCache
from litellm.llms.base_llm.chat.transformation import BaseLLMException

COPILOT_VERSION: Final = "0.26.7"
EDITOR_PLUGIN_VERSION: Final = f"copilot-chat/{COPILOT_VERSION}"
USER_AGENT: Final = f"GitHubCopilotChat/{COPILOT_VERSION}"
API_VERSION: Final = "2025-04-01"
DEFAULT_GITHUB_COPILOT_API_BASE: Final = "https://api.githubcopilot.com"
_GITHUB_API_HOST: Final = "api.github.com"
_COPILOT_USER_URL: Final = f"https://{_GITHUB_API_HOST}/copilot_internal/user"
_PLAN_API_BASE_TTL_SECONDS: Final = 3600
_PLAN_API_BASE_RETRY_SECONDS: Final = 60
_PLAN_API_BASE_TIMEOUT_SECONDS: Final = 10.0
DEFAULT_COPILOT_INTEGRATION_ID: Final = "vscode-chat"
DEFAULT_COPILOT_EDITOR_VERSION: Final = "vscode/1.115.0"
DEFAULT_COPILOT_EDITOR_PLUGIN_VERSION: Final = EDITOR_PLUGIN_VERSION
DEFAULT_COPILOT_USER_AGENT: Final = USER_AGENT

_COPILOT_AUTH_HEADER_CONFIG = (
    ("accept", "GITHUB_COPILOT_ACCEPT", "application/json"),
    ("content-type", "GITHUB_COPILOT_CONTENT_TYPE", "application/json"),
    ("copilot-integration-id", "GITHUB_COPILOT_INTEGRATION_ID", DEFAULT_COPILOT_INTEGRATION_ID),
    ("editor-version", "GITHUB_COPILOT_EDITOR_VERSION", DEFAULT_COPILOT_EDITOR_VERSION),
    ("editor-plugin-version", "GITHUB_COPILOT_EDITOR_PLUGIN_VERSION", DEFAULT_COPILOT_EDITOR_PLUGIN_VERSION),
    ("user-agent", "GITHUB_COPILOT_USER_AGENT", DEFAULT_COPILOT_USER_AGENT),
)
_COPILOT_REQUEST_HEADER_CONFIG = _COPILOT_AUTH_HEADER_CONFIG + (
    ("openai-intent", "GITHUB_COPILOT_OPENAI_INTENT", "conversation-panel"),
    ("x-github-api-version", "GITHUB_COPILOT_API_VERSION", API_VERSION),
    (
        "x-vscode-user-agent-library-version",
        "GITHUB_COPILOT_USER_AGENT_LIBRARY_VERSION",
        "electron-fetch",
    ),
)


class _PlanApiBaseCache(Protocol):
    def get_cache(self, key: str) -> object: ...

    def set_cache(self, key: str, value: object, ttl: float | None = None) -> object: ...

    def flush_cache(self) -> None: ...


class _SyncGetClient(Protocol):
    def get(
        self, url: str, headers: Mapping[str, str] | None = None, timeout: float | None = None
    ) -> httpx.Response: ...


class _AsyncGetClient(Protocol):
    async def get(
        self, url: str, headers: Mapping[str, str] | None = None, timeout: float | None = None
    ) -> httpx.Response: ...


@with_config(ConfigDict(extra="allow"))
class _CopilotUserEndpoints(TypedDict):
    api: ReadOnly[NotRequired[object]]


@with_config(ConfigDict(extra="allow"))
class _CopilotUserPayload(TypedDict):
    endpoints: ReadOnly[NotRequired[_CopilotUserEndpoints]]


_COPILOT_USER: Final = TypeAdapter(_CopilotUserPayload)
_PLAN_API_BASES: Final = cast(  # cast-ok: pins the untyped module cache to _PlanApiBaseCache
    _PlanApiBaseCache,
    InMemoryCache(max_size_in_memory=1000, default_ttl=_PLAN_API_BASE_TTL_SECONDS),
)
_PLAN_API_BASE_LOOKUPS: Final[  # mutable-ok: single-flight registry of in-progress plan host lookups per token
    dict[str, "asyncio.Task[None]"]
] = {}


class GithubCopilotError(BaseLLMException):
    def __init__(
        self,
        status_code,
        message,
        request: httpx.Request | None = None,
        response: httpx.Response | None = None,
        headers: httpx.Headers | dict | None = None,
        body: dict | None = None,
    ):
        super().__init__(
            status_code=status_code,
            message=message,
            request=request,
            response=response,
            headers=headers,
            body=body,
        )


class GetDeviceCodeError(GithubCopilotError):
    pass


class GetAccessTokenError(GithubCopilotError):
    pass


class GetAPIKeyError(GithubCopilotError):
    pass


class InvalidAPIBaseError(GithubCopilotError):
    pass


def _get_copilot_header_value(environment_variable: str, default: str | None) -> str | None:
    value = os.getenv(environment_variable)
    if value is None:
        return default
    return value or None


def _get_configured_copilot_headers(
    config: tuple[tuple[str, str, str | None], ...],
) -> dict[str, str]:
    return {
        header: value
        for header, environment_variable, default in config
        if (value := _get_copilot_header_value(environment_variable, default)) is not None
    }


def get_copilot_auth_headers(
    access_token: str | None = None,
) -> dict[str, str]:  # mutable-ok: returned straight into httpx handlers whose headers params require dict
    headers: Final = _get_configured_copilot_headers(_COPILOT_AUTH_HEADER_CONFIG)
    return {**headers, "authorization": f"token {access_token}"} if access_token else headers


def get_copilot_default_headers(
    api_key: str,
) -> dict[str, str]:  # mutable-ok: callers merge caller headers into the returned dict
    return {
        **_get_configured_copilot_headers(_COPILOT_REQUEST_HEADER_CONFIG),
        "Authorization": f"Bearer {api_key}",
        "x-request-id": str(uuid4()),
    }


def _hostname(url: str) -> str | None:
    try:
        return urlsplit(url).hostname
    except ValueError:
        return None


def _is_valid_api_base(api_base: str) -> bool:
    try:
        parsed_api_base: Final = urlsplit(api_base)
    except ValueError:
        return False
    scheme: Final = parsed_api_base.scheme.lower()
    hostname: Final = parsed_api_base.hostname
    return (
        hostname is not None
        and scheme in ("http", "https")
        and parsed_api_base.username is None
        and parsed_api_base.password is None
        and not parsed_api_base.query
        and not parsed_api_base.fragment
    )


def _reject_enterprise_token_exchange_url() -> None:
    api_key_url: Final = os.getenv("GITHUB_COPILOT_API_KEY_URL")
    if api_key_url and _hostname(api_key_url) != _GITHUB_API_HOST:
        raise InvalidAPIBaseError(
            status_code=500,
            message=(
                "GITHUB_COPILOT_API_KEY_URL is no longer used. Set GITHUB_COPILOT_API_BASE to your Copilot API host, "
                "such as https://copilot-api.<subdomain>.ghe.com"
            ),
        )


def _configured_copilot_api_base_source(api_base: str | None) -> tuple[str, str | None]:
    return ("api_base", api_base) if api_base else ("GITHUB_COPILOT_API_BASE", os.getenv("GITHUB_COPILOT_API_BASE"))


def _configured_copilot_api_base(api_base: str | None) -> str | None:
    source, configured_api_base = _configured_copilot_api_base_source(api_base)
    if not configured_api_base:
        _reject_enterprise_token_exchange_url()
        return None
    if not _is_valid_api_base(configured_api_base):
        raise InvalidAPIBaseError(
            status_code=500,
            message=(f"GitHub Copilot {source} must be an HTTP or HTTPS URL without credentials, query, or fragment"),
        )
    return configured_api_base


def _trusted_plan_api_base(endpoints_api: object) -> str | None:
    if not isinstance(endpoints_api, str) or not _is_valid_api_base(endpoints_api):
        return None
    try:
        parsed: Final = urlsplit(endpoints_api)
        port: Final = parsed.port
    except ValueError:
        return None
    hostname: Final = (parsed.hostname or "").lower()
    is_copilot_host: Final = parsed.scheme.lower() == "https" and hostname.endswith(".githubcopilot.com")
    return endpoints_api.rstrip("/") if is_copilot_host and port in (None, 443) else None


def _plan_api_base_from(response: httpx.Response) -> tuple[str, int]:
    if not response.is_success:
        return DEFAULT_GITHUB_COPILOT_API_BASE, _PLAN_API_BASE_RETRY_SECONDS
    try:
        endpoints: Final = _COPILOT_USER.validate_python(response.json()).get("endpoints")
    except ValueError:
        return DEFAULT_GITHUB_COPILOT_API_BASE, _PLAN_API_BASE_RETRY_SECONDS
    endpoints_api: Final = endpoints.get("api") if endpoints is not None else None
    return _trusted_plan_api_base(endpoints_api) or DEFAULT_GITHUB_COPILOT_API_BASE, _PLAN_API_BASE_TTL_SECONDS


def _plan_api_base_key(github_token: str) -> str:
    return hashlib.sha256(github_token.encode()).hexdigest()


def _cached_plan_api_base(key: str) -> str | None:
    cached: Final = _PLAN_API_BASES.get_cache(key)
    return cached if isinstance(cached, str) else None


def _remember_plan_api_base(key: str, resolved: tuple[str, int]) -> None:
    api_base, ttl = resolved
    _PLAN_API_BASES.set_cache(key, api_base, ttl=ttl)


def _plan_api_base_is_needed(github_token: str, api_base: str | None) -> bool:
    return (
        _configured_copilot_api_base(api_base) is None
        and _cached_plan_api_base(_plan_api_base_key(github_token)) is None
    )


def _lookup_plan_api_base(github_token: str) -> tuple[str, int]:
    import litellm

    client: Final = cast(_SyncGetClient, litellm.module_level_client)  # cast-ok: pins the untyped client
    try:
        response: Final = client.get(
            _COPILOT_USER_URL, headers=get_copilot_auth_headers(github_token), timeout=_PLAN_API_BASE_TIMEOUT_SECONDS
        )
    except httpx.HTTPError:
        return DEFAULT_GITHUB_COPILOT_API_BASE, _PLAN_API_BASE_RETRY_SECONDS
    return _plan_api_base_from(response)


async def _alookup_plan_api_base(github_token: str) -> tuple[str, int]:
    from litellm.llms.custom_httpx import http_handler

    client: Final = cast(  # cast-ok: pins the untyped client factory to the local Protocol
        Callable[[str], _AsyncGetClient], http_handler.get_async_httpx_client
    )("github_copilot")
    try:
        response: Final = await client.get(
            _COPILOT_USER_URL, headers=get_copilot_auth_headers(github_token), timeout=_PLAN_API_BASE_TIMEOUT_SECONDS
        )
    except httpx.HTTPError:
        return DEFAULT_GITHUB_COPILOT_API_BASE, _PLAN_API_BASE_RETRY_SECONDS
    return _plan_api_base_from(response)


def prefetch_copilot_plan_api_base(github_token: str, api_base: str | None = None) -> None:
    if _plan_api_base_is_needed(github_token, api_base):
        _remember_plan_api_base(_plan_api_base_key(github_token), _lookup_plan_api_base(github_token))


async def _alookup_and_remember_plan_api_base(key: str, github_token: str) -> None:
    _remember_plan_api_base(key, await _alookup_plan_api_base(github_token))


def _forget_plan_api_base_lookup(key: str, lookup: "asyncio.Task[None]") -> None:
    if _PLAN_API_BASE_LOOKUPS.get(key) is lookup:
        _PLAN_API_BASE_LOOKUPS.pop(key, None)


def _plan_api_base_lookup(key: str, github_token: str) -> "asyncio.Task[None]":
    loop: Final = asyncio.get_running_loop()
    in_flight: Final = _PLAN_API_BASE_LOOKUPS.get(key)
    if in_flight is not None and in_flight.get_loop() is loop:
        return in_flight
    lookup: Final = loop.create_task(_alookup_and_remember_plan_api_base(key, github_token))
    _PLAN_API_BASE_LOOKUPS[key] = lookup
    lookup.add_done_callback(partial(_forget_plan_api_base_lookup, key))
    return lookup


async def aprefetch_copilot_plan_api_base(github_token: str, api_base: str | None = None) -> None:
    if _plan_api_base_is_needed(github_token, api_base):
        await asyncio.shield(_plan_api_base_lookup(_plan_api_base_key(github_token), github_token))


def get_copilot_api_base(api_base: str | None = None, github_token: Callable[[], str | None] | None = None) -> str:
    configured_api_base: Final = _configured_copilot_api_base(api_base)
    if configured_api_base is not None:
        return configured_api_base
    token: Final = github_token() if github_token is not None else None
    plan_api_base: Final = _cached_plan_api_base(_plan_api_base_key(token)) if token else None
    return plan_api_base or DEFAULT_GITHUB_COPILOT_API_BASE


async def aget_copilot_api_base(github_token: str, api_base: str | None = None) -> str:
    await aprefetch_copilot_plan_api_base(github_token, api_base)
    return get_copilot_api_base(api_base, lambda: github_token)


def pin_session_authorization(
    headers: MutableMapping[str, str],  # mutable-ok: the helper fixes caller-owned headers in place
    session_token: str,
) -> None:
    """The stored per-user session token owns Authorization outright: drop every
    caller-supplied bearer header (any casing) so it can never displace the
    session's on the wire."""
    for key in tuple(k for k in headers if k.lower() == "authorization"):
        del headers[key]
    headers["Authorization"] = f"Bearer {session_token}"
