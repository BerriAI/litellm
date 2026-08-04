"""
Microsoft Graph directory-search provider.

Implements `DirectorySearchProvider` (see `base.py`) for Microsoft Entra ID.
Uses app-only (client credentials) Graph auth - distinct from the delegated
user-login flow in `MicrosoftSSOHandler`.
"""

import asyncio
import hashlib
import time
from types import MappingProxyType

import httpx
from fastapi import HTTPException

from litellm._logging import verbose_proxy_logger
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.secret_managers.main import get_secret_bool, get_secret_str
from litellm.types.llms.custom_http import httpxSpecialProvider
from litellm.types.proxy.management_endpoints.internal_user_endpoints import DirectoryUser

MICROSOFT_GRAPH_DEFAULT_ENDPOINT = "https://graph.microsoft.com/v1.0"
MICROSOFT_GRAPH_DEFAULT_SCOPE = "https://graph.microsoft.com/.default"
MICROSOFT_DIRECTORY_SEARCH_MAX_RESULTS = 10
# 60s safety margin so a cached token doesn't expire mid-request.
MICROSOFT_GRAPH_TOKEN_EXPIRY_SKEW_SECONDS = 60

_microsoft_graph_token_cache: dict[str, str | float] = {}  # mutable-ok: cache mutated in place to expire tokens
# Serializes token (re)fetches so concurrent searches on a cold/expired
# cache converge on a single Graph token request instead of a stampede.
_microsoft_graph_token_lock = asyncio.Lock()


def _is_microsoft_directory_search_enabled() -> bool:
    return get_secret_bool("MICROSOFT_DIRECTORY_SEARCH_ENABLED") is True


def is_microsoft_directory_search_configured() -> bool:
    if not _is_microsoft_directory_search_enabled():
        return False
    return all(
        get()
        for get in (
            _get_microsoft_directory_tenant,
            _get_microsoft_directory_client_id,
            _get_microsoft_directory_client_secret,
        )
    )


def _first_configured_secret(*secret_names: str) -> str | None:
    """Returns the first secret_name whose value is set (even if empty),
    falling back to the next only when a name is entirely unset. This keeps
    an explicitly-set-but-empty directory-specific override from silently
    falling through to the shared SSO credentials."""
    for secret_name in secret_names:
        value = get_secret_str(secret_name)
        if value is not None:
            return value or None
    return None


def _get_microsoft_directory_tenant() -> str | None:
    return _first_configured_secret("MICROSOFT_DIRECTORY_TENANT", "MICROSOFT_TENANT")


def _get_microsoft_directory_client_id() -> str | None:
    return _first_configured_secret("MICROSOFT_DIRECTORY_CLIENT_ID", "MICROSOFT_CLIENT_ID")


def _get_microsoft_directory_client_secret() -> str | None:
    return _first_configured_secret("MICROSOFT_DIRECTORY_CLIENT_SECRET", "MICROSOFT_CLIENT_SECRET")


def _get_microsoft_graph_endpoint() -> str:
    return (get_secret_str("MICROSOFT_GRAPH_ENDPOINT") or MICROSOFT_GRAPH_DEFAULT_ENDPOINT).rstrip("/")


def _escape_microsoft_graph_filter_value(value: str) -> str:
    return value.replace("'", "''")


def _clear_microsoft_graph_token_cache() -> None:
    _microsoft_graph_token_cache.pop("access_token", None)
    _microsoft_graph_token_cache.pop("expires_at", None)
    _microsoft_graph_token_cache.pop("credentials_fingerprint", None)


def _credentials_fingerprint(tenant: str, client_id: str, client_secret: str) -> str:
    """Identifies which tenant/client_id/client_secret a cached token was
    issued for, so a runtime credential change (secret rotation, switching
    between a dedicated app registration and the shared SSO one) can't
    return a token fetched under the previous configuration."""
    return hashlib.sha256(f"{tenant}:{client_id}:{client_secret}".encode()).hexdigest()


def _get_cached_token_if_fresh(credentials_fingerprint: str) -> str | None:
    cached_token = _microsoft_graph_token_cache.get("access_token")
    expires_at = _microsoft_graph_token_cache.get("expires_at")
    cached_fingerprint = _microsoft_graph_token_cache.get("credentials_fingerprint")
    if (
        isinstance(cached_token, str)
        and isinstance(expires_at, (float, int))
        and cached_fingerprint == credentials_fingerprint
        and expires_at > time.time() + MICROSOFT_GRAPH_TOKEN_EXPIRY_SKEW_SECONDS
    ):
        return cached_token
    return None


async def _get_microsoft_graph_access_token(force_refresh: bool = False) -> str:
    tenant = _get_microsoft_directory_tenant()
    client_id = _get_microsoft_directory_client_id()
    client_secret = _get_microsoft_directory_client_secret()
    if not tenant or not client_id or not client_secret:
        raise HTTPException(
            status_code=500,
            detail="Microsoft directory search is missing tenant, client id, or client secret.",
        )
    credentials_fingerprint = _credentials_fingerprint(tenant, client_id, client_secret)

    if not force_refresh:
        cached_token = _get_cached_token_if_fresh(credentials_fingerprint)
        if cached_token is not None:
            return cached_token

    # Serialize refreshes so concurrent callers on a cold/expired cache
    # converge on a single Graph token request instead of a stampede.
    async with _microsoft_graph_token_lock:
        if not force_refresh:
            cached_token = _get_cached_token_if_fresh(credentials_fingerprint)
            if cached_token is not None:
                return cached_token

        token_url = f"https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.SSO_HANDLER)
        try:
            response = await client.post(
                token_url,
                data=MappingProxyType(
                    {
                        "client_id": client_id,
                        "client_secret": client_secret,
                        "scope": MICROSOFT_GRAPH_DEFAULT_SCOPE,
                        "grant_type": "client_credentials",
                    }
                ),
            )
        except httpx.HTTPStatusError as e:
            # Azure's error body (e.g. invalid_client, AADSTS7000215) is the
            # actionable part - log it before the generic 502 handling upstream.
            verbose_proxy_logger.error(
                "Microsoft Graph token request failed (%s): %s",
                e.response.status_code,
                e.response.text,
            )
            raise
        token_response = response.json()

        access_token = token_response.get("access_token")
        if not isinstance(access_token, str):
            raise HTTPException(
                status_code=500,
                detail="Microsoft Graph token response did not include an access token.",
            )

        expires_in = token_response.get("expires_in", 3600)
        try:
            expires_in_seconds = int(expires_in)
        except (TypeError, ValueError):
            expires_in_seconds = 3600
        _microsoft_graph_token_cache["access_token"] = access_token
        _microsoft_graph_token_cache["expires_at"] = time.time() + expires_in_seconds
        _microsoft_graph_token_cache["credentials_fingerprint"] = credentials_fingerprint
        return access_token


def _parse_one_microsoft_directory_user(raw_user: dict) -> DirectoryUser | None:
    # `mail` is often empty for guest/unlicensed accounts; userPrincipalName
    # is the more reliable fallback (still guarded below in case both are missing).
    email = raw_user.get("mail") or raw_user.get("userPrincipalName")
    user_id = raw_user.get("id")
    if not isinstance(email, str) or not isinstance(user_id, str):
        # Log which fields were missing, not their values - the values
        # are directory identity data (names/emails) that shouldn't
        # land in general proxy logs.
        verbose_proxy_logger.warning(
            "directory_user_search: skipping AD record - missing id or email "
            "(has_id=%s, has_display_name=%s, has_mail=%s, has_user_principal_name=%s)",
            isinstance(user_id, str),
            isinstance(raw_user.get("displayName"), str),
            isinstance(raw_user.get("mail"), str),
            isinstance(raw_user.get("userPrincipalName"), str),
        )
        return None
    return DirectoryUser(id=user_id, display_name=raw_user.get("displayName"), email=email)


def _parse_microsoft_directory_users(
    directory_response: dict,
) -> tuple[DirectoryUser, ...]:
    parsed = (
        _parse_one_microsoft_directory_user(raw_user)
        for raw_user in directory_response.get("value", ())
        if isinstance(raw_user, dict)
    )
    return tuple(user for user in parsed if user is not None)


async def _fetch_microsoft_directory_users(query: str, access_token: str) -> httpx.Response:
    escaped_query = _escape_microsoft_graph_filter_value(query)
    graph_url = f"{_get_microsoft_graph_endpoint()}/users"
    client = get_async_httpx_client(llm_provider=httpxSpecialProvider.SSO_HANDLER)
    response = await client.get(
        graph_url,
        headers=MappingProxyType(
            {
                "Authorization": f"Bearer {access_token}",
                # Required by Graph for advanced queries like an OR of multiple
                # startswith() clauses - see Microsoft's "advanced query
                # capabilities on Microsoft Entra ID objects" docs.
                "ConsistencyLevel": "eventual",
            }
        ),
        params=MappingProxyType(
            {
                "$filter": (
                    f"startswith(displayName,'{escaped_query}') or "
                    f"startswith(mail,'{escaped_query}') or "
                    f"startswith(userPrincipalName,'{escaped_query}')"
                ),
                "$select": "id,displayName,mail,userPrincipalName",
                "$top": str(MICROSOFT_DIRECTORY_SEARCH_MAX_RESULTS),
                "$count": "true",
            }
        ),
    )
    response.raise_for_status()
    return response


async def _search_microsoft_directory_users(
    query: str,
) -> tuple[DirectoryUser, ...]:
    access_token = await _get_microsoft_graph_access_token()
    try:
        response = await _fetch_microsoft_directory_users(query, access_token)
    except httpx.HTTPStatusError as e:
        if e.response.status_code == 401:
            # Cached token was rejected (e.g. secret rotated) - refresh once
            # and retry rather than failing the whole search.
            _clear_microsoft_graph_token_cache()
            access_token = await _get_microsoft_graph_access_token(force_refresh=True)
            response = await _fetch_microsoft_directory_users(query, access_token)
        else:
            raise

    return _parse_microsoft_directory_users(response.json())


class MicrosoftDirectorySearchProvider:
    """Adapts the module-level Graph functions above to `DirectorySearchProvider`."""

    def is_configured(self) -> bool:
        return is_microsoft_directory_search_configured()

    async def search(self, query: str) -> tuple[DirectoryUser, ...]:
        return await _search_microsoft_directory_users(query)
