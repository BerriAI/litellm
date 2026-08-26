"""
GigaChat OAuth Authenticator

Handles OAuth 2.0 token management for GigaChat API.
Based on official GigaChat SDK authentication flow.
"""

import time
import uuid
from collections.abc import Mapping
from typing import Final

import httpx

from litellm._logging import verbose_logger
from litellm.caching.caching import InMemoryCache
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders

# GigaChat OAuth endpoint
GIGACHAT_AUTH_URL: Final = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

# Default scope for personal API access
GIGACHAT_SCOPE: Final = "GIGACHAT_API_PERS"

# Token expiry buffer in milliseconds (refresh token 60s before expiry)
TOKEN_EXPIRY_BUFFER_MS: Final = 60000

# Cache for access tokens
_token_cache: Final = InMemoryCache()


class GigaChatAuthError(BaseLLMException):
    """GigaChat authentication error."""


def _get_credentials() -> str | None:
    """Get GigaChat credentials from environment."""
    return get_secret_str("GIGACHAT_CREDENTIALS") or get_secret_str("GIGACHAT_API_KEY")


def _get_auth_url() -> str:
    """Get GigaChat auth URL from environment or use default."""
    return get_secret_str("GIGACHAT_AUTH_URL") or GIGACHAT_AUTH_URL


def _get_scope() -> str:
    """Get GigaChat scope from environment or use default."""
    return get_secret_str("GIGACHAT_SCOPE") or GIGACHAT_SCOPE


def _get_http_client() -> HTTPHandler:
    """Get cached httpx client with SSL verification disabled."""
    return HTTPHandler(ssl_verify=False)


def get_access_token(
    credentials: str | None = None,
    scope: str | None = None,
    auth_url: str | None = None,
    litellm_params: Mapping[str, object] | None = None,
) -> str:
    """
    Get valid access token, using cache if available.

    Args:
        credentials: Base64-encoded credentials (client_id:client_secret)
        scope: API scope (GIGACHAT_API_PERS, GIGACHAT_API_CORP, etc.)
        auth_url: OAuth endpoint URL

    Returns:
        Access token string

    Raises:
        GigaChatAuthError: If authentication fails
    """
    if not litellm_params:
        litellm_params = {}  # mutable-ok: empty dict default; rebind-ok: provide default

    access_token: Final = litellm_params.get("gigachat_access_token") or get_secret_str("GIGACHAT_ACCESS_TOKEN")
    if access_token:
        return access_token

    effective_credentials: Final = credentials or _get_credentials()
    if not effective_credentials:
        raise GigaChatAuthError(
            status_code=401,
            message="GigaChat credentials not provided. Set GIGACHAT_CREDENTIALS or GIGACHAT_API_KEY environment variable.",
        )

    effective_scope: Final = scope or litellm_params.get("gigachat_scope") or _get_scope()
    effective_auth_url: Final = auth_url or litellm_params.get("gigachat_auth_url") or _get_auth_url()

    # Check cache
    cache_key: Final = f"gigachat_token:{effective_credentials[:16]}"
    cached: Final = _token_cache.get_cache(cache_key)
    if cached:
        _token, _expires_at = cached
        # Check if token is still valid (with buffer)
        if time.time() * 1000 < _expires_at - TOKEN_EXPIRY_BUFFER_MS:
            verbose_logger.debug("Using cached GigaChat access token")
            return _token

    # Request new token
    new_token, new_expires_at = _request_token_sync(effective_credentials, effective_scope, effective_auth_url)  # pyright: ignore[reportArgumentType]  # credential keys may be broader than str

    if new_expires_at:
        # Cache token
        ttl_seconds: Final = max(0, (new_expires_at - TOKEN_EXPIRY_BUFFER_MS - time.time() * 1000) / 1000)
        if ttl_seconds > 0:
            _token_cache.set_cache(cache_key, (new_token, new_expires_at), ttl=ttl_seconds)

    return new_token


async def get_access_token_async(
    credentials: str | None = None,
    scope: str | None = None,
    auth_url: str | None = None,
    litellm_params: Mapping[str, object] | None = None,
) -> str:
    """Async version of get_access_token."""
    if not litellm_params:
        litellm_params = {}  # mutable-ok: empty dict default; rebind-ok: provide default

    access_token: Final = litellm_params.get("gigachat_access_token") or get_secret_str("GIGACHAT_ACCESS_TOKEN")
    if access_token:
        return access_token

    effective_credentials: Final = credentials or _get_credentials()
    if not effective_credentials:
        raise GigaChatAuthError(
            status_code=401,
            message="GigaChat credentials not provided. Set GIGACHAT_CREDENTIALS or GIGACHAT_API_KEY environment variable.",
        )

    effective_scope: Final = scope or litellm_params.get("gigachat_scope") or _get_scope()
    effective_auth_url: Final = auth_url or litellm_params.get("gigachat_auth_url") or _get_auth_url()

    # Check cache
    cache_key: Final = f"gigachat_token:{effective_credentials[:16]}"
    cached: Final = _token_cache.get_cache(cache_key)
    if cached:
        _token, _expires_at = cached
        if time.time() * 1000 < _expires_at - TOKEN_EXPIRY_BUFFER_MS:
            verbose_logger.debug("Using cached GigaChat access token")
            return _token

    # Request new token
    new_token, new_expires_at = await _request_token_async(effective_credentials, effective_scope, effective_auth_url)  # pyright: ignore[reportArgumentType]  # credential keys may be broader than str

    if new_expires_at:
        # Cache token
        ttl_seconds: Final = max(0, (new_expires_at - TOKEN_EXPIRY_BUFFER_MS - time.time() * 1000) / 1000)
        if ttl_seconds > 0:
            _token_cache.set_cache(cache_key, (new_token, new_expires_at), ttl=ttl_seconds)

    return new_token


def _request_token_sync(
    credentials: str,
    scope: str,
    auth_url: str,
) -> tuple[str, int]:
    """
    Request new access token from GigaChat OAuth endpoint (sync).

    Returns:
        tuple of (access_token, expires_at_ms)
    """
    headers: Final = {
        "Authorization": f"Basic {credentials}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data: Final = {"scope": scope}

    verbose_logger.debug("Requesting GigaChat access token from %s", auth_url)

    try:
        client: Final = _get_http_client()
        response: Final = client.post(auth_url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        return _parse_token_response(response)  # pyright: ignore[reportArgumentType]  # httpx Response may be None at type level
    except httpx.HTTPStatusError as e:
        raise GigaChatAuthError(
            status_code=e.response.status_code,
            message=f"GigaChat authentication failed: {e.response.text}",
        )
    except httpx.RequestError as e:
        raise GigaChatAuthError(
            status_code=500,
            message=f"GigaChat authentication request failed: {e}",
        )


async def _request_token_async(
    credentials: str,
    scope: str,
    auth_url: str,
) -> tuple[str, int]:
    """Async version of _request_token_sync."""
    headers: Final = {
        "Authorization": f"Basic {credentials}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data: Final = {"scope": scope}

    verbose_logger.debug("Requesting GigaChat access token from %s", auth_url)

    try:
        client: Final = get_async_httpx_client(
            llm_provider=LlmProviders.GIGACHAT,
            params={"ssl_verify": False},
        )
        response: Final = await client.post(auth_url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        return _parse_token_response(response)  # pyright: ignore[reportArgumentType]  # httpx Response may be None at type level
    except httpx.HTTPStatusError as e:
        raise GigaChatAuthError(
            status_code=e.response.status_code,
            message=f"GigaChat authentication failed: {e.response.text}",
        )
    except httpx.RequestError as e:
        raise GigaChatAuthError(
            status_code=500,
            message=f"GigaChat authentication request failed: {e}",
        )


def _parse_token_response(response: httpx.Response) -> tuple[str, int]:
    """Parse OAuth token response."""
    data: Final = response.json()

    # GigaChat returns either 'tok'/'exp' or 'access_token'/'expires_at'
    access_token: Final = data.get("tok") or data.get("access_token")
    expires_at_raw: Final = data.get("exp") or data.get("expires_at")

    if not access_token:
        raise GigaChatAuthError(
            status_code=500,
            message=f"Invalid token response: {data}",
        )

    # expires_at is in milliseconds
    expires_at: int  # rebind-ok: conditionally assigned from str or int
    if isinstance(expires_at_raw, str):
        expires_at = int(expires_at_raw)  # rebind-ok: conditionally assigned from str or int
    else:
        expires_at = expires_at_raw  # pyright: ignore[reportAssignmentType]  # raw value is int or str; converted above; rebind-ok: conditionally assigned from str or int

    verbose_logger.debug("GigaChat access token obtained successfully")
    return access_token, expires_at
