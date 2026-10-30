"""Cross-replica ``TokenCacheBackend``: stores the token in LiteLLM's shared ``DualCache``.

Plugs into the foundation's ``CachedOAuthTokenStore`` via the ``TokenCacheBackend`` seam. The token is
encrypted + serialized by the injected codec and written under a per-``(user, server)`` key with the
given TTL, so every worker reads one refresh rather than each re-reading and re-refreshing - matching
v1's ``MCPPerUserTokenCache`` (same NaCl encryption and key, so a token cached by either is readable by
the other across the cutover). A missing or undecryptable entry reads as a miss.
"""

from __future__ import annotations

from dataclasses import KW_ONLY, dataclass
from datetime import timedelta
from typing import Final, Protocol

from litellm._internal_context import with_service_target
from litellm._logging import verbose_logger
from litellm.caching.redis_batch import active_request_redis_batch
from litellm.caching.redis_cache import RedisCache
from litellm.proxy.common_utils.user_api_key_cache import UserApiKeyCache
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy.mcp_server.outbound_credentials.token_cache_codec import (
    OAuthTokenCacheCodec,
)
from litellm.proxy.mcp_server.utils import MCP_OAUTH_TOKENS_TARGET
from litellm.redis.stores.mcp_per_user_tokens import (
    TOKEN_KEY_PREFIX,
    McpPerUserTokenStore,
    TransportMcpPerUserTokenStore,
    per_user_token_key,
)
from litellm.redis.transport.contract import RedisFailure


class TokenMemory(Protocol):
    async def async_get_cache(self, key: str) -> object: ...

    async def async_set_cache(self, key: str, value: str, *, ttl: float | None = None) -> None: ...

    def delete_cache(self, key: str) -> None: ...


class AsyncCache(Protocol):
    """The slice of LiteLLM's ``DualCache`` this backend needs (Redis-backed, shared across workers)."""

    @property
    def redis_cache(self) -> RedisCache | None: ...

    @property
    def in_memory_cache(self) -> TokenMemory: ...

    @property
    def default_in_memory_ttl(self) -> float | None: ...


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
    key_prefix: str = TOKEN_KEY_PREFIX

    def _key(self, user_id: str, server_id: str) -> str:
        return per_user_token_key(user_id, server_id, prefix=self.key_prefix)

    def _store(self) -> McpPerUserTokenStore | None:
        redis: Final = self.cache.redis_cache
        return None if redis is None else TransportMcpPerUserTokenStore(redis.transport(), self.key_prefix)

    async def _read(self, user_id: str, server_id: str) -> object:
        key: Final = self._key(user_id, server_id)
        local: Final = await self.cache.in_memory_cache.async_get_cache(key)
        store: Final = self._store()
        if local is not None or store is None:
            return local
        redis: Final = self.cache.redis_cache
        batch: Final = None if redis is None else active_request_redis_batch(redis)
        if batch is not None and batch.read_as_missing(key):
            return None
        blob: Final = await store.read(user_id, server_id)
        if isinstance(blob, RedisFailure):
            return None
        if blob is not None:
            ttl: Final = self.cache.default_in_memory_ttl
            await self.cache.in_memory_cache.async_set_cache(key, blob, **({"ttl": ttl} if ttl is not None else {}))
        return blob

    @with_service_target(MCP_OAUTH_TOKENS_TARGET)
    async def get(self, user_id: str, server_id: str) -> OAuthToken | None:
        try:
            blob: Final = await self._read(user_id, server_id)
            return self.codec.decode(blob) if isinstance(blob, str) else None
        except Exception as exc:  # noqa: BLE001  # cache and codec failures remain safe misses
            verbose_logger.debug("MCP per-user token cache get failed (miss): %s", exc)
            return None

    @with_service_target(MCP_OAUTH_TOKENS_TARGET)
    async def set(self, user_id: str, server_id: str, token: OAuthToken, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            return
        await self.cache_token(user_id, server_id, token, ttl_seconds)

    async def cache_token(self, user_id: str, server_id: str, token: OAuthToken, ttl_seconds: float) -> None:
        try:
            ciphertext: Final = self.codec.encode(token)
            await self.cache.in_memory_cache.async_set_cache(
                self._key(user_id, server_id),
                ciphertext,
                ttl=ttl_seconds,
            )
            store: Final = self._store()
            if store is not None:
                await store.write(
                    user_id,
                    server_id,
                    ciphertext,
                    ttl=timedelta(seconds=int(ttl_seconds)),
                    defer=isinstance(self.cache, UserApiKeyCache),
                )
        except Exception as exc:  # noqa: BLE001  # cache and codec failures must not abort token resolution
            verbose_logger.debug("MCP per-user token cache set failed (ignored): %s", exc)

    @with_service_target(MCP_OAUTH_TOKENS_TARGET)
    async def delete(self, user_id: str, server_id: str) -> None:
        await self.evict_token(user_id, server_id)

    async def evict_token(self, user_id: str, server_id: str) -> None:
        try:
            self.cache.in_memory_cache.delete_cache(self._key(user_id, server_id))
            store: Final = self._store()
            if store is not None:
                await store.delete(user_id, server_id)
        except Exception as exc:  # noqa: BLE001  # best-effort eviction must not abort token resolution
            verbose_logger.debug("MCP per-user token cache delete failed (ignored): %s", exc)
