"""
Generic OAuth2 client_credentials authenticator for OpenAI-compatible providers.

Backs the ``custom_oauth`` provider: fetches and caches a bearer token from any
standard OAuth2 token endpoint so an OpenAI-compatible endpoint sitting behind an
OAuth2 client-credentials gateway can be driven purely from config.
"""

import hashlib
from typing import Optional

import httpx
from pydantic import BaseModel, ValidationError

from litellm._logging import verbose_logger
from litellm.caching.caching import InMemoryCache
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import HTTPHandler, _get_httpx_client

DEFAULT_TOKEN_EXPIRY_SECONDS = 3600
TOKEN_EXPIRY_BUFFER_SECONDS = 60

_token_cache = InMemoryCache()


class OAuthClientCredentialsError(BaseLLMException):
    """Raised when the OAuth2 client_credentials token flow fails."""


class _TokenResponse(BaseModel):
    access_token: str
    expires_in: int = DEFAULT_TOKEN_EXPIRY_SECONDS


def get_client_credentials_token(
    token_url: Optional[str],
    client_id: Optional[str],
    client_secret: Optional[str],
    scope: Optional[str] = None,
    timeout: float = 30.0,
    http_client: Optional[HTTPHandler] = None,
) -> str:
    """
    Fetch (and cache) an OAuth2 access token using the client_credentials grant.

    The token is cached in-process keyed by (token_url, client_id, scope, secret
    hash) until ``TOKEN_EXPIRY_BUFFER_SECONDS`` before the ``expires_in`` returned
    by the IdP. The secret is hashed into the key so distinct credentials never
    share a cache slot. ``http_client`` is injectable so tests can assert the
    outbound request without real network access.
    """
    if not token_url or not client_id or not client_secret:
        raise OAuthClientCredentialsError(
            status_code=401,
            message=(
                "custom_oauth requires oauth_token_url, oauth_client_id and "
                "oauth_client_secret (set them under litellm_params or via the "
                "CUSTOM_OAUTH_TOKEN_URL / CUSTOM_OAUTH_CLIENT_ID / "
                "CUSTOM_OAUTH_CLIENT_SECRET environment variables)."
            ),
        )

    secret_hash = hashlib.sha256(client_secret.encode()).hexdigest()[:16]
    cache_key = f"custom_oauth:{token_url}:{client_id}:{scope or ''}:{secret_hash}"
    cached = _token_cache.get_cache(cache_key)
    if isinstance(cached, str):
        return cached

    data = {
        "grant_type": "client_credentials",
        "client_id": client_id,
        "client_secret": client_secret,
    }
    if scope:
        data["scope"] = scope

    client = http_client or _get_httpx_client()
    try:
        response = client.post(token_url, data=data, timeout=timeout)
        if response is None:
            raise OAuthClientCredentialsError(
                status_code=500,
                message="custom_oauth token request returned no response",
            )
        response.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise OAuthClientCredentialsError(
            status_code=e.response.status_code,
            message=f"custom_oauth token request failed: {e.response.text}",
        ) from e
    except httpx.RequestError as e:
        raise OAuthClientCredentialsError(
            status_code=500,
            message=f"custom_oauth token request error: {e}",
        ) from e

    try:
        parsed = _TokenResponse.model_validate(response.json())
    except ValidationError as e:
        raise OAuthClientCredentialsError(
            status_code=500,
            message=f"custom_oauth token response missing a valid access_token: {e}",
        ) from e

    ttl = parsed.expires_in - TOKEN_EXPIRY_BUFFER_SECONDS
    if ttl > 0:
        _token_cache.set_cache(cache_key, parsed.access_token, ttl=ttl)

    verbose_logger.debug("custom_oauth: obtained new access token")
    return parsed.access_token
