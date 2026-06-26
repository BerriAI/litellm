"""Cross-replica ``TokenCacheBackend``: stores the token in LiteLLM's shared ``DualCache``.

Plugs into the foundation's ``CachedOAuthTokenStore`` via the ``TokenCacheBackend`` seam. The token is
encrypted + serialized by the injected codec and written under a per-``(user, server)`` key with the
given TTL, so every worker reads one refresh rather than each re-reading and re-refreshing - matching
v1's ``MCPPerUserTokenCache`` (same NaCl encryption and key, so a token cached by either is readable by
the other across the cutover). A missing or undecryptable entry reads as a miss.
"""

from __future__ import annotations

from typing import Protocol

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_cache_codec import (
    OAuthTokenCacheCodec,
)


class AsyncCache(Protocol):
    """The slice of LiteLLM's ``DualCache`` this backend needs (Redis-backed, shared across workers)."""

    async def async_get_cache(self, key: str) -> object | None: ...

    async def async_set_cache(
        self, key: str, value: str, ttl: float | None = None
    ) -> None: ...

    async def async_delete_cache(self, key: str) -> None: ...


class DualCacheTokenCacheBackend:
    def __init__(
        self,
        cache: AsyncCache,
        codec: OAuthTokenCacheCodec,
        *,
        key_prefix: str = "mcp:per_user_token:",
    ) -> None:
        self._cache = cache
        self._codec = codec
        self._key_prefix = key_prefix

    def _key(self, user_id: str, server_id: str) -> str:
        return f"{self._key_prefix}{user_id}:{server_id}"

    async def get(self, user_id: str, server_id: str) -> OAuthToken | None:
        blob = await self._cache.async_get_cache(self._key(user_id, server_id))
        return self._codec.decode(blob) if isinstance(blob, str) else None

    async def set(
        self, user_id: str, server_id: str, token: OAuthToken, ttl_seconds: float
    ) -> None:
        if ttl_seconds <= 0:
            return
        await self._cache.async_set_cache(
            self._key(user_id, server_id),
            self._codec.encode(token),
            ttl=ttl_seconds,
        )

    async def delete(self, user_id: str, server_id: str) -> None:
        await self._cache.async_delete_cache(self._key(user_id, server_id))
