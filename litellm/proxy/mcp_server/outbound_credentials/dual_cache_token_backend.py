"""Cross-replica ``TokenCacheBackend``: stores the token in LiteLLM's shared ``DualCache``.

Plugs into the foundation's ``CachedOAuthTokenStore`` via the ``TokenCacheBackend`` seam. The token is
encrypted + serialized by the injected codec and written under a per-``(user, server)`` key with the
given TTL, so every worker reads one refresh rather than each re-reading and re-refreshing - matching
v1's ``MCPPerUserTokenCache`` (same NaCl encryption and key, so a token cached by either is readable by
the other across the cutover). A missing or undecryptable entry reads as a miss.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from litellm._internal_context import with_service_target
from litellm._logging import verbose_logger
from litellm.proxy.common_utils.user_api_key_cache import (
    UserApiKeyCache,
    delete_mcp_per_user_token,
    get_mcp_per_user_token,
    set_mcp_per_user_token,
)
from litellm.proxy.mcp_server.outbound_credentials.oauth_token_store import (
    OAuthToken,
)
from litellm.proxy.mcp_server.outbound_credentials.token_cache_codec import (
    OAuthTokenCacheCodec,
)
from litellm.proxy.mcp_server.utils import MCP_OAUTH_TOKENS_TARGET


@dataclass(frozen=True, slots=True)
class DualCacheTokenCacheBackend:
    """Every method degrades a cache or codec failure to its safe value - ``get`` to a miss
    (``None``), ``set``/``delete`` to a no-op - so a Redis outage or an undecryptable entry reads as a
    cache miss rather than a request error, matching v1 and this layer's "boundary failure = miss"
    contract. The guarantee holds here regardless of whether the injected cache/codec also swallow.
    """

    cache: UserApiKeyCache
    codec: OAuthTokenCacheCodec

    @with_service_target(MCP_OAUTH_TOKENS_TARGET)
    async def get(self, user_id: str, server_id: str) -> OAuthToken | None:
        try:
            blob: Final = await get_mcp_per_user_token(self.cache, user_id, server_id)
            return self.codec.decode(blob) if blob is not None else None
        except Exception as exc:  # noqa: BLE001
            verbose_logger.debug("MCP per-user token cache get failed (miss): %s", exc)
            return None

    @with_service_target(MCP_OAUTH_TOKENS_TARGET)
    async def set(self, user_id: str, server_id: str, token: OAuthToken, ttl_seconds: float) -> None:
        if ttl_seconds <= 0:
            return
        try:
            await set_mcp_per_user_token(self.cache, user_id, server_id, self.codec.encode(token), ttl=ttl_seconds)
        except Exception as exc:  # noqa: BLE001
            verbose_logger.debug("MCP per-user token cache set failed (ignored): %s", exc)

    @with_service_target(MCP_OAUTH_TOKENS_TARGET)
    async def delete(self, user_id: str, server_id: str) -> None:
        try:
            await delete_mcp_per_user_token(self.cache, user_id, server_id)
        except Exception as exc:  # noqa: BLE001
            verbose_logger.debug("MCP per-user token cache delete failed (ignored): %s", exc)
