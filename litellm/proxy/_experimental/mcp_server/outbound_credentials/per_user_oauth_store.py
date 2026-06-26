"""Composition root for the v2-native authorization_code per-user OAuth token store (step 1b).

Assembles ``Cached(Refreshing(V2PerUserTokenStore))`` and replaces ``V1PerUserTokenStore`` in the
resolver. The runtime collaborators (DB, HTTP, the shared cache, Redis) are LiteLLM globals not ready
at import time, so the chain is built lazily on first use. When Redis is wired it uses the
cross-replica path (DualCache-backed cache + ``SET NX PX`` coordinator); otherwise it falls back to
the foundation's in-process defaults (correct for a single replica). The DB read/refresh-grant/persist
collaborators acquire their globals per call, mirroring v1's lazy-import pattern.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from litellm._logging import verbose_logger
from litellm.proxy._experimental.mcp_server.outbound_credentials.authz_code_refresher import (
    AuthorizationCodeRefresher,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.dual_cache_token_backend import (
    AsyncCache,
    DualCacheTokenCacheBackend,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    CachedOAuthTokenStore,
    OAuthToken,
    RefreshCoordinator,
    RefreshingTokenStore,
    TokenCacheBackend,
    TokenStoreUnavailable,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_distributed_lock import (
    RedisDistributedLock,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    RedisRefreshCoordinator,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.token_cache_codec import (
    OAuthTokenCacheCodec,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.v2_token_store import (
    V2PerUserTokenStore,
)

if TYPE_CHECKING:
    from litellm.types.mcp_server.mcp_server_manager import MCPServer

# A token with no declared expiry is cached for this long; one with an expiry is cached until then.
_DEFAULT_TTL_SECONDS = 300.0

ServerLookup = Callable[[str], "MCPServer | None"]


async def _read_credential(user_id: str, server_id: str) -> dict[str, object] | None:
    from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415
        get_user_oauth_credential,
    )
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415

    if prisma_client is None:
        raise TokenStoreUnavailable("Database not connected")
    return await get_user_oauth_credential(prisma_client, user_id, server_id)


async def _persist_credential(
    user_id: str,
    server_id: str,
    access_token: str,
    refresh_token: str | None,
    expires_in: int | None,
    scopes: tuple[str, ...] | None,
) -> None:
    from litellm.proxy._experimental.mcp_server.db import (  # noqa: PLC0415
        store_user_oauth_credential,
    )
    from litellm.proxy.proxy_server import prisma_client  # noqa: PLC0415

    if prisma_client is None:
        return
    await store_user_oauth_credential(
        prisma_client=prisma_client,
        user_id=user_id,
        server_id=server_id,
        access_token=access_token,
        refresh_token=refresh_token,
        expires_in=expires_in,
        scopes=list(scopes) if scopes else None,
        skip_byok_guard=True,
    )


async def _post_token_endpoint(
    url: str, form: dict[str, str]
) -> dict[str, object] | None:
    from litellm.llms.custom_httpx.http_handler import (  # noqa: PLC0415
        get_async_httpx_client,  # pyright: ignore
    )
    from litellm.types.llms.custom_http import httpxSpecialProvider  # noqa: PLC0415

    # litellm's httpx handler and httpx.Response are only partially typed; the IdP returns a JSON
    # object and the refresher validates each field, so the untyped boundary is contained here.
    provider = httpxSpecialProvider.Oauth2Check
    headers = {"Accept": "application/json"}
    # A failed refresh is a miss, not a 500 (matches v1), so any error becomes None.
    try:
        client = get_async_httpx_client(llm_provider=provider)  # pyright: ignore
        response = await client.post(url, headers=headers, data=form)  # pyright: ignore
        response.raise_for_status()  # pyright: ignore
        body: dict[str, object] = response.json()  # pyright: ignore
    except Exception as exc:  # noqa: BLE001
        verbose_logger.warning("MCP OAuth refresh request failed: %s", exc)
        return None
    else:
        return body  # pyright: ignore


def _runtime_backend_and_coordinator() -> (
    tuple[TokenCacheBackend | None, RefreshCoordinator | None]
):
    """The cross-replica cache + coordinator when Redis is wired, else ``(None, None)`` so the
    foundation's in-process defaults are used (a single replica needs no shared cache or lock).
    """
    from litellm.proxy.common_utils.encrypt_decrypt_utils import (  # noqa: PLC0415
        decrypt_value_helper,
        encrypt_value_helper,
    )
    from litellm.proxy.proxy_server import user_api_key_cache  # noqa: PLC0415

    redis_cache = user_api_key_cache.redis_cache
    if redis_cache is None:
        return None, None
    codec = OAuthTokenCacheCodec(
        encrypt_value_helper,
        lambda blob: decrypt_value_helper(blob, "mcp_per_user_token"),
    )
    # user_api_key_cache satisfies the AsyncCache slice (DualCache types ttl via **kwargs) and the
    # Redis client from init_async_client() is partially typed - both are untyped-boundary casts.
    cache: AsyncCache = user_api_key_cache  # pyright: ignore
    redis_client = redis_cache.init_async_client()  # pyright: ignore
    lock = RedisDistributedLock(redis_client)  # pyright: ignore
    backend = DualCacheTokenCacheBackend(cache, codec)
    coordinator = RedisRefreshCoordinator(lock)
    return backend, coordinator


def build_per_user_oauth_token_store(
    server_lookup: ServerLookup,
) -> CachedOAuthTokenStore:
    backend, coordinator = _runtime_backend_and_coordinator()
    refresher = AuthorizationCodeRefresher(
        server_lookup, _post_token_endpoint, _persist_credential
    )
    refreshing = RefreshingTokenStore(
        V2PerUserTokenStore(_read_credential), refresher, coordinator=coordinator
    )
    return CachedOAuthTokenStore(
        refreshing, default_ttl_seconds=_DEFAULT_TTL_SECONDS, backend=backend
    )


class LazyPerUserOAuthTokenStore:
    """``OAuthTokenStore`` that builds the v2-native chain on first ``fetch``.

    The chain's cache/lock collaborators are LiteLLM runtime globals not available when the resolver
    is constructed at import time, so construction is deferred to the first request (by when they are
    wired). Built once, then reused.
    """

    def __init__(self, server_lookup: ServerLookup) -> None:
        self._server_lookup = server_lookup
        self._store: CachedOAuthTokenStore | None = None

    async def fetch(self, user_id: str, server_id: str) -> OAuthToken | None:
        if self._store is None:
            self._store = build_per_user_oauth_token_store(self._server_lookup)
        return await self._store.fetch(user_id, server_id)
