"""
OAuth2 client_credentials token cache for MCP servers.

Automatically fetches and refreshes access tokens for MCP servers configured
with ``client_id``, ``client_secret``, and ``token_url``.
"""

import asyncio
from typing import TYPE_CHECKING, Dict, Optional, Tuple, Union

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
    MCP_OAUTH2_TOKEN_CACHE_MAX_SIZE,
    MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
    MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
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
        if server_id not in self._locks:
            self._locks[server_id] = asyncio.Lock()
        return self._locks[server_id]

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

        data: Dict[str, str] = {
            "grant_type": "client_credentials",
            "client_id": server.client_id,  # type: ignore[arg-type]
            "client_secret": server.client_secret,  # type: ignore[arg-type]
        }
        if server.scopes:
            data["scope"] = " ".join(server.scopes)

        verbose_logger.debug(
            "Fetching OAuth2 client_credentials token for MCP server %s from %s",
            server.server_id,
            server.token_url,
        )

        response = await client.post(server.token_url, data=data)  # type: ignore[arg-type]
        response.raise_for_status()
        body = response.json()

        access_token = body.get("access_token")
        if not access_token:
            raise ValueError(
                f"OAuth2 token response for MCP server '{server.server_id}' "
                f"missing 'access_token': {body}"
            )

        expires_in = int(body.get("expires_in", 3600))
        ttl = max(expires_in - MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS, MCP_OAUTH2_TOKEN_CACHE_MIN_TTL)

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
