"""
OAuth 2.0 Token Exchange (RFC 8693) handler for MCP servers.

Exchanges a user's incoming JWT (subject_token) for a scoped access token
at an IDP's token exchange endpoint.  The exchanged token is then used to
authenticate requests to the upstream MCP server.

See: https://datatracker.ietf.org/doc/html/rfc8693
"""

import asyncio
import hashlib
import weakref
from typing import TYPE_CHECKING, Dict, Tuple

import httpx
from mcp.shared.auth import OAuthToken
from pydantic import ValidationError

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


def parse_oauth_token_response(response: httpx.Response, server_id: str) -> OAuthToken:
    """Validate a token-endpoint response per RFC 6749 §5.1 using the SDK model.

    Raises ``ValueError`` with a server-scoped message on malformed payloads so
    callers surface a clear configuration error instead of a pydantic trace.
    """
    try:
        return OAuthToken.model_validate(response.json())
    except ValidationError as exc:
        raise ValueError(
            f"OAuth2 token response for MCP server '{server_id}' is not a valid "
            f"RFC 6749 token payload: {exc.error_count()} validation error(s)"
        ) from exc


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
        # WeakValueDictionary so locks are GC'd once no coroutine holds a reference,
        # preventing unbounded growth with many rotating user tokens.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def _get_lock(self, cache_key: str) -> asyncio.Lock:
        lock = self._locks.get(cache_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[cache_key] = lock
        return lock

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
            # RFC 8693 §2.1: explicit requested_token_type (the spec default)
            "requested_token_type": "urn:ietf:params:oauth:token-type:access_token",
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
            verbose_logger.debug(
                "Token exchange IDP error for MCP server %s (status %d)",
                server.server_id,
                exc.response.status_code,
            )
            raise ValueError(
                f"Token exchange for MCP server '{server.server_id}' "
                f"failed with status {exc.response.status_code}"
            ) from exc

        token = parse_oauth_token_response(response, server.server_id)
        expires_in = (
            token.expires_in
            if token.expires_in is not None
            else MCP_OAUTH2_TOKEN_CACHE_DEFAULT_TTL
        )
        ttl = max(
            expires_in - MCP_OAUTH2_TOKEN_EXPIRY_BUFFER_SECONDS,
            MCP_OAUTH2_TOKEN_CACHE_MIN_TTL,
        )

        verbose_logger.info(
            "Token exchange succeeded for MCP server %s (expires in %ds)",
            server.server_id,
            expires_in,
        )
        return token.access_token, ttl

    def invalidate(self, subject_token: str, server_id: str) -> None:
        """Remove a cached exchanged token (e.g. after a 401)."""
        cache_key = self._cache_key(subject_token, server_id)
        self._cache.delete_cache(cache_key)


# Module-level singleton
mcp_token_exchange_handler = TokenExchangeHandler()
