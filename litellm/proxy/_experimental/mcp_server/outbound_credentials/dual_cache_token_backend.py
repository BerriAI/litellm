"""Cross-replica ``TokenCacheBackend``: stores the token in LiteLLM's shared ``DualCache``.

Plugs into the foundation's ``CachedOAuthTokenStore`` via the ``TokenCacheBackend`` seam. The token is
encrypted + serialized by the injected codec and written under a per-``(user, server)`` key with the
given TTL, so every worker reads one refresh rather than each re-reading and re-refreshing - matching
v1's ``MCPPerUserTokenCache`` (same NaCl encryption and key, so a token cached by either is readable by
the other across the cutover). A missing or undecryptable entry reads as a miss.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from typing import Protocol

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_cache_codec import (
    OAuthTokenCacheCodec,
)


class AsyncCache(Protocol):
    """The slice of LiteLLM's ``DualCache`` this backend needs (Redis-backed, shared across workers)."""

    async def async_get_cache(self, key: str) -> object | None: ...

    async def async_set_cache(self, key: str, value: str, ttl: float | None = None) -> None: ...

    async def async_delete_cache(self, key: str) -> None: ...


@dataclass(frozen=True, slots=True)
class DualCacheTokenCacheBackend:
    """Every method degrades a cache or codec failure to its safe value - ``get`` to a miss
    (``None``), ``set``/``delete`` to a no-op - so a Redis outage or an undecryptable entry reads as a
    cache miss rather than a request error, matching v1 and this layer's "boundary failure = miss"
    contract. The guarantee holds here regardless of whether the injected cache/codec also swallow.
    """

    cache: AsyncCache
    codec: OAuthTokenCacheCodec
    _: KW_ONLY
    key_prefix: str = "mcp:per_user_token:"

    def _key(self, user_id: str, server_id: str) -> str:
        return f"{self.key_prefix}{user_id}:{server_id}"

    async def get(self, user_id: str, server_id: str) -> OAuthToken | None:
        try:
            blob = await self.cache.async_get_cache(self._key(user_id, server_id))
            return self.codec.decode(blob) if isinstance(blob, str) else None
        except Exception as exc:  # noqa: BLE001
            verbose_logger.debug("MCP per-user token cache get failed (miss): %s", exc)
            return None

    async def set(self, user_id: str, server_id: str, token: OAuthToken, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            return
        try:
            await self.cache.async_set_cache(
                self._key(user_id, server_id),
                self.codec.encode(token),
                ttl=ttl_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            verbose_logger.debug("MCP per-user token cache set failed (ignored): %s", exc)

    async def delete(self, user_id: str, server_id: str) -> None:
        try:
            await self.cache.async_delete_cache(self._key(user_id, server_id))
        except Exception as exc:  # noqa: BLE001
            verbose_logger.debug("MCP per-user token cache delete failed (ignored): %s", exc)
