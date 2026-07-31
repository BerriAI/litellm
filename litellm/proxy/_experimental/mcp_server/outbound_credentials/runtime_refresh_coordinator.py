"""The runtime ``RefreshCoordinator``: cross-replica single-flight when Redis is wired.

Builds ``RedisRefreshCoordinator`` over the proxy's shared Redis so one refresh runs per key
across the fleet, or returns ``None`` when Redis is absent so the caller keeps the foundation's
in-process default (correct for a single replica). The proxy globals it reads are not ready at
import time, so this is called per composition rather than held as module state.

Shared by every credential arm that renews a stored grant: a rotating refresh token must be
redeemed once across all workers, so each arm electing its own winner with its own lock shape
would be a bug waiting to differ.
"""

from __future__ import annotations

from litellm.proxy._experimental.mcp_server.outbound_credentials.oauth_token_store import (
    RefreshCoordinator,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_distributed_lock import (
    RedisDistributedLock,
)
from litellm.proxy._experimental.mcp_server.outbound_credentials.redis_refresh_coordinator import (
    RedisRefreshCoordinator,
)


def runtime_refresh_coordinator() -> RefreshCoordinator | None:
    from litellm.proxy.proxy_server import user_api_key_cache  # noqa: PLC0415  # runtime global

    redis_cache = user_api_key_cache.redis_cache
    if redis_cache is None:
        return None
    # The Redis client from init_async_client() is only partially typed; the lock validates every
    # reply it depends on, so the untyped boundary is contained here.
    redis_client = redis_cache.init_async_client()  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]  # litellm redis wrapper is untyped
    lock = RedisDistributedLock(
        redis_client,  # pyright: ignore[reportArgumentType,reportUnknownArgumentType]  # litellm redis wrapper is untyped
        namespace_key=redis_cache.check_and_fix_namespace,
    )
    return RedisRefreshCoordinator(lock)
