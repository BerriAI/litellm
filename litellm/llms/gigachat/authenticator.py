"""
GigaChat OAuth Authenticator

Handles OAuth 2.0 token management for GigaChat API.
Based on official GigaChat SDK authentication flow.

Supports two authentication methods:
1. OAuth 2.0 with credentials (Base64-encoded client_id:client_secret)
2. Basic auth with user/password
"""

import base64
import binascii
import time
import uuid
from typing import Optional, Tuple

import httpx

from litellm._logging import verbose_logger
from litellm.caching.caching import InMemoryCache
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.utils import LlmProviders

from .common_utils import (
    GIGACHAT_AUTH_URL,
    GIGACHAT_BASE_URL,
    GIGACHAT_SCOPE,
    TOKEN_EXPIRY_BUFFER_MS,
    USER_AGENT,
    GigaChatAuthError,
    build_url,
)

# Cache for access tokens
_token_cache = InMemoryCache()


def _validate_credentials(credentials: str) -> None:
    """Validate that credentials are properly base64-encoded."""
    try:
        base64.b64decode(credentials, validate=True)
    except (ValueError, binascii.Error):
        verbose_logger.warning(
            "Invalid credentials format. Please use only base64 credentials "
            "(Authorization data, not client secret!)"
        )


def _get_credentials() -> Optional[str]:
    """Get GigaChat credentials from environment."""
    return get_secret_str("GIGACHAT_CREDENTIALS") or get_secret_str("GIGACHAT_API_KEY")


def _get_user_password() -> Tuple[Optional[str], Optional[str]]:
    """Get GigaChat user/password from environment."""
    user = get_secret_str("GIGACHAT_USER")
    password = get_secret_str("GIGACHAT_PASSWORD")
    return user, password


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
    user: Optional[str] = None,
    password: Optional[str] = None
) -> str:
    """
    Get valid access token, using cache if available.

    Supports two authentication methods:
    1. OAuth 2.0 with credentials (Base64-encoded client_id:client_secret)
    2. Basic auth with user/password (returns token from /token endpoint)

    Args:
        credentials: Base64-encoded credentials (client_id:client_secret)
        scope: API scope (GIGACHAT_API_PERS, GIGACHAT_API_CORP, etc.)
        auth_url: OAuth endpoint URL
        user: Username for basic auth
        password: Password for basic auth

    Returns:
        Access token string

    Raises:
        GigaChatAuthError: If authentication fails
    """
    # Try OAuth first
    credentials = credentials or _get_credentials()

    # Fallback to user/password
    if not credentials:
        env_user, env_password = _get_user_password()
        user = user or env_user
        password = password or env_password

    if not credentials and not (user and password):
        raise GigaChatAuthError(
            status_code=401,
            message=(
                "GigaChat credentials not provided. "
                "Set GIGACHAT_CREDENTIALS/GIGACHAT_API_KEY or GIGACHAT_USER/GIGACHAT_PASSWORD environment variables."
            ),
        )

    scope = scope or _get_scope()
    auth_url = auth_url or _get_auth_url()

    # Determine cache key based on auth method
    if credentials:
        cache_key = f"gigachat_token:oauth:{credentials[:16]}"
    else:
        cache_key = f"gigachat_token:basic:{user}"

    # Check cache
    cached = _token_cache.get_cache(cache_key)
    if cached:
        token, expires_at = cached
        # Check if token is still valid (with buffer)
        if time.time() * 1000 < expires_at - TOKEN_EXPIRY_BUFFER_MS:
            verbose_logger.debug("Using cached GigaChat access token")
            return token

    # Request new token
    if credentials:
        _validate_credentials(credentials)
        token, expires_at = _request_oauth_token_sync(credentials, scope, auth_url)
    else:
        token, expires_at = _request_basic_token_sync(user, password)

    # Cache token
    ttl_seconds = max(0, (expires_at - TOKEN_EXPIRY_BUFFER_MS - time.time() * 1000) / 1000)
    if ttl_seconds > 0:
        _token_cache.set_cache(cache_key, (token, expires_at), ttl=ttl_seconds)

    return token


async def get_access_token_async(
    credentials: Optional[str] = None,
    scope: Optional[str] = None,
    auth_url: Optional[str] = None,
    user: Optional[str] = None,
    password: Optional[str] = None,
) -> str:
    """Async version of get_access_token."""
    # Try OAuth first
    credentials = credentials or _get_credentials()

    # Fallback to user/password
    if not credentials:
        env_user, env_password = _get_user_password()
        user = user or env_user
        password = password or env_password

    if not credentials and not (user and password):
        raise GigaChatAuthError(
            status_code=401,
            message=(
                "GigaChat credentials not provided. "
                "Set GIGACHAT_CREDENTIALS/GIGACHAT_API_KEY or GIGACHAT_USER/GIGACHAT_PASSWORD environment variables."
            ),
        )

    scope = scope or _get_scope()
    auth_url = auth_url or _get_auth_url()

    # Determine cache key based on auth method
    if credentials:
        cache_key = f"gigachat_token:oauth:{credentials[:16]}"
    else:
        cache_key = f"gigachat_token:basic:{user}"

    # Check cache
    cached = _token_cache.get_cache(cache_key)
    if cached:
        token, expires_at = cached
        if time.time() * 1000 < expires_at - TOKEN_EXPIRY_BUFFER_MS:
            verbose_logger.debug("Using cached GigaChat access token")
            return token

    # Request new token
    if credentials:
        _validate_credentials(credentials)
        token, expires_at = await _request_oauth_token_async(credentials, scope, auth_url)
    else:
        token, expires_at = await _request_basic_token_async(user, password)

    # Cache token
    ttl_seconds = max(0, (expires_at - TOKEN_EXPIRY_BUFFER_MS - time.time() * 1000) / 1000)
    if ttl_seconds > 0:
        _token_cache.set_cache(cache_key, (token, expires_at), ttl=ttl_seconds)

    return token


def _request_oauth_token_sync(
    credentials: str,
    scope: str,
    auth_url: str,
) -> Tuple[str, int]:
    """
    Request new access token from GigaChat OAuth endpoint (sync).

    Args:
        credentials: Base64-encoded credentials
        scope: API scope
        auth_url: OAuth endpoint URL

    Returns:
        Tuple of (access_token, expires_at_ms)
    """
    headers = {
        "Authorization": f"Basic {credentials}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": USER_AGENT,
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


async def _request_oauth_token_async(
    credentials: str,
    scope: str,
    auth_url: str,
) -> Tuple[str, int]:
    """Async version of _request_oauth_token_sync."""
    headers = {
        "Authorization": f"Basic {credentials}",
        "RqUID": str(uuid.uuid4()),
        "Content-Type": "application/x-www-form-urlencoded",
        "User-Agent": USER_AGENT,
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


def _request_basic_token_sync(
    user: Optional[str] = None,
    password: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Tuple[str, int]:
    """
    Request new access token using basic auth (user/password).

    This uses the /token endpoint instead of OAuth.
    Uses httpx.Client directly with auth parameter (as in official gigachat library).

    Args:
        user: Username
        password: Password
        api_base: API base URL

    Returns:
        Tuple of (access_token, expires_at_ms)
    """
    base_url = api_base or get_secret_str("GIGACHAT_API_BASE") or GIGACHAT_BASE_URL
    token_url = build_url(base_url, "token")

    headers = {
        "User-Agent": USER_AGENT,
    }

    verbose_logger.debug(f"Requesting GigaChat token via basic auth from {token_url}")

    try:
        # Use httpx.Client directly with auth parameter (HTTPHandler doesn't support it)
        with httpx.Client(verify=False) as client:
            response = client.post(
                token_url,
                headers=headers,
                auth=(user, password),
                timeout=30,
            )
            response.raise_for_status()
            return _parse_token_response(response)
    except httpx.HTTPStatusError as e:
        raise GigaChatAuthError(
            status_code=e.response.status_code,
            message=f"GigaChat basic auth failed: {e.response.text}",
        )
    except httpx.RequestError as e:
        raise GigaChatAuthError(
            status_code=500,
            message=f"GigaChat basic auth request failed: {str(e)}",
        )


async def _request_basic_token_async(
    user: Optional[str] = None,
    password: Optional[str] = None,
    api_base: Optional[str] = None,
) -> Tuple[str, int]:
    """Async version of _request_basic_token_sync."""
    base_url = api_base or get_secret_str("GIGACHAT_API_BASE") or GIGACHAT_BASE_URL
    token_url = build_url(base_url, "token")

    headers = {
        "User-Agent": USER_AGENT,
    }

    verbose_logger.debug(f"Requesting GigaChat token via basic auth from {token_url}")

    try:
        # Use httpx.AsyncClient directly with auth parameter
        async with httpx.AsyncClient(verify=False) as client:
            response = await client.post(
                token_url,
                headers=headers,
                auth=(user, password),
                timeout=30,
            )
            response.raise_for_status()
            return _parse_token_response(response)
    except httpx.HTTPStatusError as e:
        raise GigaChatAuthError(
            status_code=e.response.status_code,
            message=f"GigaChat basic auth failed: {e.response.text}",
        )
    except httpx.RequestError as e:
        raise GigaChatAuthError(
            status_code=500,
            message=f"GigaChat basic auth request failed: {str(e)}",
        )


def _parse_token_response(response: httpx.Response) -> Tuple[str, int]:
    """
    Parse OAuth/token response.

    GigaChat returns either:
    - OAuth: {'tok': '...', 'exp': 1234567890}
    - Token: {'access_token': '...', 'expires_at': 1234567890}
    """
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

    # If expires_at is 0 or not provided, set it to far future
    if not expires_at:
        expires_at = int(time.time() * 1000) + 30 * 60 * 1000  # 30 minutes from now

    verbose_logger.debug("GigaChat access token obtained successfully")
    return access_token, expires_at
