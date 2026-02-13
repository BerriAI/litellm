"""
OAuth 2.0 Token Exchange (RFC 8693) handler for MCP servers.

Exchanges a user's incoming JWT (subject_token) for a scoped access token
at an IDP's token exchange endpoint.  The exchanged token is then used to
authenticate requests to the upstream MCP server.

See: https://datatracker.ietf.org/doc/html/rfc8693
"""

import asyncio
import hashlib
from typing import TYPE_CHECKING, Dict, Optional, Tuple

import httpx

from litellm._logging import verbose_logger
from litellm.caching.in_memory_cache import InMemoryCache
from litellm.constants import (
    MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
    MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
    MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
    MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE,
)
from litellm.llms.custom_httpx.http_handler import get_async_httpx_client
from litellm.types.llms.custom_http import httpxSpecialProvider

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

# RFC 8693 grant type constant
TOKEN_EXCHANGE_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:token-exchange"

DEFAULT_SUBJECT_TOKEN_TYPE = "urn:ietf:params:oauth:token-type:access_token"


class TokenExchangeHandler:
    """Handles OAuth 2.0 Token Exchange (RFC 8693) for MCP servers.

    Caches exchanged tokens keyed by ``hash(subject_token + server_id)`` so
    repeated calls with the same user token skip the IDP round-trip.
    """

    def __init__(self) -> None:
        self._cache = InMemoryCache(
            max_size_in_memory=MCP_TOKEN_EXCHANGE_CACHE_MAX_SIZE,
            default_ttl=MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL,
        )
        self._locks: Dict[str, asyncio.Lock] = {}

    def _get_lock(self, cache_key: str) -> asyncio.Lock:
        return self._locks.setdefault(cache_key, asyncio.Lock())

    @staticmethod
    def _cache_key(subject_token: str, server_id: str) -> str:
        raw = f"{subject_token}:{server_id}"
        return hashlib.sha256(raw.encode()).hexdigest()

    async def exchange_token(
        self,
        subject_token: str,
        server: "MCPServer",
    ) -> str:
        """Exchange *subject_token* for a scoped access token.

        Returns the exchanged ``access_token`` string (suitable for a
        ``Bearer`` header).

        Raises ``ValueError`` on configuration or IDP errors.
        """
        cache_key = self._cache_key(subject_token, server.server_id)

        # Fast path
        cached = self._cache.get_cache(cache_key)
        if cached is not None:
            return cached

        # Slow path — one exchange at a time per (user, server) pair
        async with self._get_lock(cache_key):
            cached = self._cache.get_cache(cache_key)
            if cached is not None:
                return cached

            token, ttl = await self._do_exchange(subject_token, server)
            self._cache.set_cache(cache_key, token, ttl=ttl)
            return token

    async def _do_exchange(
        self,
        subject_token: str,
        server: "MCPServer",
    ) -> Tuple[str, int]:
        """POST to the token exchange endpoint with RFC 8693 parameters.

        Returns ``(access_token, ttl_seconds)``.
        """
        endpoint = server.token_exchange_endpoint or server.token_url
        if not endpoint:
            raise ValueError(
                f"MCP server '{server.server_id}' has auth_type=oauth2_token_exchange "
                f"but no token_exchange_endpoint or token_url configured"
            )
        if not server.client_id or not server.client_secret:
            raise ValueError(
                f"MCP server '{server.server_id}' has auth_type=oauth2_token_exchange "
                f"but missing client_id or client_secret"
            )

        data: Dict[str, str] = {
            "grant_type": TOKEN_EXCHANGE_GRANT_TYPE,
            "subject_token": subject_token,
            "subject_token_type": server.subject_token_type
            or DEFAULT_SUBJECT_TOKEN_TYPE,
            "client_id": server.client_id,
            "client_secret": server.client_secret,
        }
        if server.audience:
            data["audience"] = server.audience
        if server.scopes:
            data["scope"] = " ".join(server.scopes)

        verbose_logger.debug(
            "Exchanging token for MCP server %s at %s (audience=%s)",
            server.server_id,
            endpoint,
            server.audience,
        )

        client = get_async_httpx_client(llm_provider=httpxSpecialProvider.MCP)
        try:
            response = await client.post(endpoint, data=data)
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise ValueError(
                f"Token exchange for MCP server '{server.server_id}' "
                f"failed with status {exc.response.status_code}: "
                f"{exc.response.text}"
            ) from exc

        body = response.json()
        if not isinstance(body, dict):
            raise ValueError(
                f"Token exchange response for MCP server '{server.server_id}' "
                f"returned non-object JSON (got {type(body).__name__})"
            )

        access_token = body.get("access_token")
        if not access_token:
            raise ValueError(
                f"Token exchange response for MCP server '{server.server_id}' "
                f"missing 'access_token'"
            )

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
            "Token exchange succeeded for MCP server %s (expires in %ds)",
            server.server_id,
            expires_in,
        )
        return access_token, ttl

    def invalidate(self, subject_token: str, server_id: str) -> None:
        """Remove a cached exchanged token (e.g. after a 401)."""
        cache_key = self._cache_key(subject_token, server_id)
        self._cache.delete_cache(cache_key)


# Module-level singleton
mcp_token_exchange_handler = TokenExchangeHandler()
