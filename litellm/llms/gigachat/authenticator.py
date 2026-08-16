"""
GigaChat OAuth Authenticator

Handles OAuth 2.0 token management for GigaChat API.
Based on official GigaChat SDK authentication flow.
"""

import time
import uuid
from typing import Optional, Tuple

import httpx

from litellm._logging import verbose_logger
from litellm.caching.caching import InMemoryCache
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders

# GigaChat OAuth endpoint
GIGACHAT_AUTH_URL = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"

# Default scope for personal API access
GIGACHAT_SCOPE = "GIGACHAT_API_PERS"

# Token expiry buffer in milliseconds (refresh token 60s before expiry)
TOKEN_EXPIRY_BUFFER_MS = 60000

# Cache for access tokens
_token_cache = InMemoryCache()


class GigaChatAuthError(BaseLLMException):
    """GigaChat authentication error."""

    pass


def _get_credentials() -> Optional[str]:
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
    return _get_httpx_client(params={"ssl_verify": False})


def get_access_token(
    credentials: Optional[str] = None,
    scope: Optional[str] = None,
    auth_url: Optional[str] = None,
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
    credentials = credentials or _get_credentials()
    if not credentials:
        raise GigaChatAuthError(
            status_code=401,
            message="GigaChat credentials not provided. Set GIGACHAT_CREDENTIALS or GIGACHAT_API_KEY environment variable.",
        )

    scope = scope or _get_scope()
    auth_url = auth_url or _get_auth_url()

    # Check cache
    cache_key = f"gigachat_token:{credentials[:16]}"
    cached = _token_cache.get_cache(cache_key)
    if cached:
        token, expires_at = cached
        # Check if token is still valid (with buffer)
        if time.time() * 1000 < expires_at - TOKEN_EXPIRY_BUFFER_MS:
            verbose_logger.debug("Using cached GigaChat access token")
            return token

    # Request new token
    token, expires_at = _request_token_sync(credentials, scope, auth_url)

    # Cache token
    ttl_seconds = max(0, (expires_at - TOKEN_EXPIRY_BUFFER_MS - time.time() * 1000) / 1000)
    if ttl_seconds > 0:
        _token_cache.set_cache(cache_key, (token, expires_at), ttl=ttl_seconds)

    return token


async def get_access_token_async(
    credentials: Optional[str] = None,
    scope: Optional[str] = None,
    auth_url: Optional[str] = None,
) -> str:
    """Async version of get_access_token."""
    credentials = credentials or _get_credentials()
    if not credentials:
        raise GigaChatAuthError(
            status_code=401,
            message="GigaChat credentials not provided. Set GIGACHAT_CREDENTIALS or GIGACHAT_API_KEY environment variable.",
        )

    scope = scope or _get_scope()
    auth_url = auth_url or _get_auth_url()

    # Check cache
    cache_key = f"gigachat_token:{credentials[:16]}"
    cached = _token_cache.get_cache(cache_key)
    if cached:
        token, expires_at = cached
        if time.time() * 1000 < expires_at - TOKEN_EXPIRY_BUFFER_MS:
            verbose_logger.debug("Using cached GigaChat access token")
            return token

    # Request new token
    token, expires_at = await _request_token_async(credentials, scope, auth_url)

    # Cache token
    ttl_seconds = max(0, (expires_at - TOKEN_EXPIRY_BUFFER_MS - time.time() * 1000) / 1000)
    if ttl_seconds > 0:
        _token_cache.set_cache(cache_key, (token, expires_at), ttl=ttl_seconds)

    return token


def _request_token_sync(
    credentials: str,
    scope: str,
    auth_url: str,
) -> Tuple[str, int]:
    """
    Request new access token from GigaChat OAuth endpoint (sync).

    Returns:
        Tuple of (access_token, expires_at_ms)
    """
    headers = {
        "Authorization": f"Basic {credentials}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"scope": scope}

    verbose_logger.debug(f"Requesting GigaChat access token from {auth_url}")

    try:
        client = _get_http_client()
        response = client.post(auth_url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        return _parse_token_response(response)
    except httpx.HTTPStatusError as e:
        raise GigaChatAuthError(
            status_code=e.response.status_code,
            message=f"GigaChat authentication failed: {e.response.text}",
        )
    except httpx.RequestError as e:
        raise GigaChatAuthError(
            status_code=500,
            message=f"GigaChat authentication request failed: {str(e)}",
        )


async def _request_token_async(
    credentials: str,
    scope: str,
    auth_url: str,
) -> Tuple[str, int]:
    """Async version of _request_token_sync."""
    headers = {
        "Authorization": f"Basic {credentials}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
    }
    data = {"scope": scope}

    verbose_logger.debug(f"Requesting GigaChat access token from {auth_url}")

    try:
        client = get_async_httpx_client(
            llm_provider=LlmProviders.GIGACHAT,
            params={"ssl_verify": False},
        )
        response = await client.post(auth_url, headers=headers, data=data, timeout=30)
        response.raise_for_status()
        return _parse_token_response(response)
    except httpx.HTTPStatusError as e:
        raise GigaChatAuthError(
            status_code=e.response.status_code,
            message=f"GigaChat authentication failed: {e.response.text}",
        )
    except httpx.RequestError as e:
        raise GigaChatAuthError(
            status_code=500,
            message=f"GigaChat authentication request failed: {str(e)}",
        )


def _parse_token_response(response: httpx.Response) -> Tuple[str, int]:
    """Parse OAuth token response."""
    data = response.json()

    # GigaChat returns either 'tok'/'exp' or 'access_token'/'expires_at'
    access_token = data.get("tok") or data.get("access_token")
    expires_at = data.get("exp") or data.get("expires_at")

    if not access_token:
        raise GigaChatAuthError(
            status_code=500,
            message=f"Invalid token response: {data}",
        )

    # expires_at is in milliseconds
    if isinstance(expires_at, str):
        expires_at = int(expires_at)

    verbose_logger.debug("GigaChat access token obtained successfully")
    return access_token, expires_at
