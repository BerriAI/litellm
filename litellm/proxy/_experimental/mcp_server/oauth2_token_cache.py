"""
OAuth2 client_credentials token cache for MCP servers.

Automatically fetches and refreshes access tokens for MCP servers configured
with ``client_id``, ``client_secret``, and ``token_url``.
"""

import asyncio
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

import httpx

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
    MCP_OAUTH2_TOKEN_CACHE_MAX_SIZE,
    MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
    MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
    MCP_PER_USER_TOKEN_DEFAULT_TTL,
    MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS,
    MCP_PER_USER_TOKEN_REDIS_KEY_PREFIX,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.proxy.common_utils.encrypt_decrypt_utils import (
    decrypt_value_helper,
    encrypt_value_helper,
)
from litellm.types.llms.custom_http import httpxSpecialProvider

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer


class MCPOAuth2TokenCache(InMemoryCache):
    """
    In-memory cache for OAuth2 client_credentials tokens, keyed by server_id.

    Inherits from ``InMemoryCache`` for TTL-based storage and eviction.
    Adds per-server ``asyncio.Lock`` to prevent duplicate concurrent fetches.
    """

    def __init__(self) -> None:
        super().__init__(
            max_size_in_memory=MCP_OAUTH2_TOKEN_CACHE_MAX_SIZE,
            default_ttl=MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
        )
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, server_id: str) -> asyncio.Lock:
        return self._locks.setdefault(server_id, asyncio.Lock())

    async def async_get_token(self, server: "MCPServer") -> Optional[str]:
        """Return a valid access token, fetching or refreshing as needed.

        Returns ``None`` when the server lacks client credentials config.
        """
        if not server.has_client_credentials:
            return None

        server_id = server.server_id

        # Fast path — cached token is still valid
        cached = self.get_cache(server_id)
        if cached is not None:
            return cached

        # Slow path — acquire per-server lock then double-check
        async with self._get_lock(server_id):
            cached = self.get_cache(server_id)
            if cached is not None:
                return cached

            token, ttl = await self._fetch_token(server)
            self.set_cache(server_id, token, ttl=ttl)
            return token

    async def _fetch_token(self, server: "MCPServer") -> Tuple[str, int]:
        """POST to ``token_url`` with ``grant_type=client_credentials``.

        Returns ``(access_token, ttl_seconds)`` where ttl accounts for the
        expiry buffer so the cache entry expires before the real token does.
        """
        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)

        if not server.client_id or not server.client_secret or not server.token_url:
            raise ValueError(
                f"MCP server '{server.server_id}' missing required OAuth2 fields: "
                f"client_id={bool(server.client_id)}, "
                f"client_secret={bool(server.client_secret)}, "
                f"token_url={bool(server.token_url)}"
            )

        data: Dict[str, str] = {
            "grant_type": "client_credentials",
            "client_id": server.client_id,
            "client_secret": server.client_secret,
        }
        if server.scopes:
            data["scope"] = " ".join(server.scopes)

        verbose_logger.debug(
            "Fetching OAuth2 client_credentials token for MCP server %s",
            server.server_id,
        )

        try:
            response = await client.post(server.token_url, data=data)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ValueError(
                f"OAuth2 token request for MCP server '{server.server_id}' "
                f"failed with status {exc.response.status_code}"
            ) from exc

        body = response.json()

        if not isinstance(body, dict):
            raise ValueError(
                f"OAuth2 token response for MCP server '{server.server_id}' "
                f"returned non-object JSON (got {type(body).__name__})"
            )

        access_token = body.get("access_token")
        if not access_token:
            raise ValueError(
                f"OAuth2 token response for MCP server '{server.server_id}' "
                f"missing 'access_token'"
            )

        # Safely parse expires_in — providers may return null or non-numeric values
        raw_expires_in = body.get("expires_in")
        try:
            expires_in = (
                int(raw_expires_in)
                if raw_expires_in is not None
                else MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL
            )
        except (TypeError, ValueError):
            expires_in = MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL

        ttl = max(
            expires_in - MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
            MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
        )

        verbose_logger.info(
            "Fetched OAuth2 token for MCP server %s (expires in %ds)",
            server.server_id,
            expires_in,
        )
        return access_token, ttl

    def invalidate(self, server_id: str) -> None:
        """Remove a cached token (e.g. after a 401)."""
        self.delete_cache(server_id)


mcp_oauth2_token_cache = MCPOAuth2TokenCache()


def _compute_per_user_token_ttl(server: "MCPServer", expires_in: Optional[int]) -> int:
    """Compute Redis TTL for a per-user token.

    Uses server.token_storage_ttl_seconds when configured; otherwise derives
    TTL from expires_in minus the expiry buffer; falls back to the default TTL.
    """
    if server.token_storage_ttl_seconds is not None:
        return max(server.token_storage_ttl_seconds, 1)
    if expires_in is not None:
        return max(
            expires_in - MCP_PER_USER_TOKEN_EXPIRY_BUFFER_SECONDS,
            1,
        )
    return MCP_PER_USER_TOKEN_DEFAULT_TTL


class MCPPerUserTokenCache:
    """Redis-backed cache for per-user OAuth2 access tokens.

    Uses LiteLLM's existing ``user_api_key_cache`` (DualCache with optional
    Redis backend).  Tokens are NaCl-encrypted with ``encrypt_value_helper``
    before storage so they are safe at rest in Redis.

    Redis key format: ``mcp:per_user_token:{user_id}:{server_id}``
    Redis value: ``encrypt_value_helper(access_token)`` — URL-safe base64
    """

    def _cache_key(self, user_id: str, server_id: str) -> str:
        return f"{MCP_PER_USER_TOKEN_REDIS_KEY_PREFIX}:{user_id}:{server_id}"

    async def get(self, user_id: str, server_id: str) -> Optional[str]:
        """Return the plaintext access_token, or None on miss/error."""
        try:
            from litellm.proxy.proxy_server import user_api_key_cache  # noqa: PLC0415

            key = self._cache_key(user_id, server_id)
            encrypted = await user_api_key_cache.async_get_cache(key)
            if encrypted is None:
                return None
            plaintext = decrypt_value_helper(
                encrypted,
                key="mcp_per_user_token",
                exception_type="debug",
            )
            return plaintext or None
        except Exception as exc:
            verbose_logger.debug(
                "MCPPerUserTokenCache.get failed for user=%s server=%s: %s",
                user_id,
                server_id,
                exc,
            )
            return None

    async def set(
        self,
        user_id: str,
        server_id: str,
        access_token: str,
        ttl: int,
    ) -> None:
        """Store NaCl-encrypted access_token in Redis with the given TTL."""
        try:
            from litellm.proxy.proxy_server import user_api_key_cache  # noqa: PLC0415

            key = self._cache_key(user_id, server_id)
            encrypted = encrypt_value_helper(access_token)
            await user_api_key_cache.async_set_cache(key, encrypted, ttl=ttl)
            verbose_logger.debug(
                "MCPPerUserTokenCache.set: cached token for user=%s server=%s ttl=%ds",
                user_id,
                server_id,
                ttl,
            )
        except Exception as exc:
            verbose_logger.debug(
                "MCPPerUserTokenCache.set failed for user=%s server=%s: %s",
                user_id,
                server_id,
                exc,
            )

    async def delete(self, user_id: str, server_id: str) -> None:
        """Invalidate the cached token (removes from both in-memory and Redis layers)."""
        try:
            from litellm.proxy.proxy_server import user_api_key_cache  # noqa: PLC0415

            key = self._cache_key(user_id, server_id)
            await user_api_key_cache.async_delete_cache(key)
        except Exception as exc:
            verbose_logger.debug(
                "MCPPerUserTokenCache.delete failed for user=%s server=%s: %s",
                user_id,
                server_id,
                exc,
            )


mcp_per_user_token_cache = MCPPerUserTokenCache()


async def resolve_mcp_auth(
    server: "MCPServer",
    mcp_auth_header: Optional[Union[str, Dict[str, str]]] = None,
) -> Optional[Union[str, Dict[str, str]]]:
    """Resolve the auth value for an MCP server.

    Priority:
    1. ``mcp_auth_header`` — per-request/per-user override
    2. OAuth2 client_credentials token — auto-fetched and cached
    3. ``server.authentication_token`` — static token from config/DB
    """
    if mcp_auth_header:
        return mcp_auth_header
    if server.has_client_credentials:
        return await mcp_oauth2_token_cache.async_get_token(server)
    return server.authentication_token
