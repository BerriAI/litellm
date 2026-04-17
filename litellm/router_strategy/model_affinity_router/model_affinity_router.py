"""
Model Affinity Router — Session Pinning

Prevents mid-session model switching in agentic loops where successive calls
may have different content characteristics that would otherwise cause
content-aware routing to select different models.

How it works
------------
1. Client sends ``X-Model-Affinity: <session-id>`` (any opaque string, typically
   a UUID) in the request header.
2. First request for that session-id routes normally (content-aware routing runs,
   or infrastructure routing picks the model). The selected **model name** is
   stored in the affinity cache keyed by session-id.
3. All subsequent requests carrying the same session-id are pinned to the cached
   model — content-aware routing is bypassed entirely.
4. Pinning is at the **model group** level, not the deployment level, so
   load-balancing and failover within the pinned model group still work normally.
5. Entries expire after a configurable TTL (default 10 min) and the cache uses
   LRU eviction once its capacity is reached.

Storage backends
----------------
- ``local``  — in-process LRU + TTL cache (default, zero extra deps)
- ``redis``  — shared across multiple proxy replicas; requires the ``redis``
               package and a ``redis_url`` in the config
"""
import asyncio
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from litellm._logging import verbose_router_logger

if TYPE_CHECKING:
    from litellm.types.router import ModelAffinityConfig
else:
    ModelAffinityConfig = Any


# ---------------------------------------------------------------------------
# Local LRU cache
# ---------------------------------------------------------------------------


class _LocalAffinityCache:
    """
    Thread-safe, async-friendly LRU cache with per-entry TTL.

    Entries are stored as ``(model_name, expires_at)`` tuples.
    ``get()`` evicts expired entries on access so they cannot be returned.
    The oldest entry is evicted when ``max_size`` is reached.
    """

    def __init__(self, max_size: int, ttl: float) -> None:
        self._max_size = max_size
        self._ttl = ttl
        # OrderedDict: oldest entries at the front (for LRU eviction)
        self._store: OrderedDict[str, Tuple[str, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[str]:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                return None
            model, expires_at = entry
            if time.monotonic() > expires_at:
                del self._store[key]
                return None
            # Promote to most-recently-used
            self._store.move_to_end(key)
            return model

    async def set(self, key: str, value: str) -> None:
        async with self._lock:
            if key in self._store:
                self._store.move_to_end(key)
            self._store[key] = (value, time.monotonic() + self._ttl)
            if len(self._store) > self._max_size:
                # Evict the least-recently-used entry
                self._store.popitem(last=False)

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._store.pop(key, None)

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Redis cache wrapper
# ---------------------------------------------------------------------------


class _RedisAffinityCache:
    """
    Redis-backed affinity cache.  TTL is delegated to Redis SETEX so entries
    expire server-side even if the Python process restarts.
    """

    _KEY_PREFIX = "litellm:model_affinity:"

    def __init__(self, redis_url: str, ttl: int) -> None:
        self._ttl = ttl
        try:
            import redis.asyncio as aioredis

            self._client = aioredis.from_url(redis_url, decode_responses=True)
        except ImportError as exc:
            raise ImportError(
                "redis package is required for ModelAffinityRouter storage='redis'. "
                "Install it with: pip install redis"
            ) from exc

    def _key(self, session_id: str) -> str:
        return f"{self._KEY_PREFIX}{session_id}"

    async def get(self, session_id: str) -> Optional[str]:
        try:
            return await self._client.get(self._key(session_id))
        except Exception as e:
            verbose_router_logger.warning(
                f"ModelAffinityRouter: Redis GET failed ({e}), treating as cache miss"
            )
            return None

    async def set(self, session_id: str, model: str) -> None:
        try:
            await self._client.setex(self._key(session_id), self._ttl, model)
        except Exception as e:
            verbose_router_logger.warning(
                f"ModelAffinityRouter: Redis SETEX failed ({e}), pin will not persist"
            )

    async def delete(self, session_id: str) -> None:
        try:
            await self._client.delete(self._key(session_id))
        except Exception as e:
            verbose_router_logger.warning(
                f"ModelAffinityRouter: Redis DELETE failed ({e})"
            )


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------


class ModelAffinityRouter:
    """
    Session-pin cache used by the Router to enforce model affinity.

    Instantiated by the Router when ``router_settings.model_affinity.enabled=true``.
    The routing logic itself lives in ``Router.async_pre_routing_hook`` — this
    class is only responsible for managing the underlying cache.
    """

    def __init__(self, config: "ModelAffinityConfig") -> None:
        self.config = config
        storage = config.storage or "local"

        if storage == "redis":
            if not config.redis_url:
                raise ValueError(
                    "ModelAffinityConfig.redis_url is required when storage='redis'"
                )
            self._cache: Any = _RedisAffinityCache(
                redis_url=config.redis_url,
                ttl=config.ttl,
            )
        else:
            self._cache = _LocalAffinityCache(
                max_size=config.max_sessions,
                ttl=float(config.ttl),
            )

        verbose_router_logger.info(
            f"ModelAffinityRouter initialized: storage={storage} "
            f"ttl={config.ttl}s max_sessions={config.max_sessions}"
        )

    async def get_pinned_model(self, session_id: str) -> Optional[str]:
        """Return the pinned model name for *session_id*, or None if not pinned."""
        model = await self._cache.get(session_id)
        if model:
            verbose_router_logger.debug(
                f"ModelAffinityRouter: cache HIT session={session_id} -> model={model}"
            )
        else:
            verbose_router_logger.debug(
                f"ModelAffinityRouter: cache MISS session={session_id}"
            )
        return model

    async def pin_model(self, session_id: str, model: str) -> None:
        """Pin *model* for *session_id* (creates or refreshes the TTL)."""
        await self._cache.set(session_id, model)
        verbose_router_logger.info(
            f"ModelAffinityRouter: pinned session={session_id} -> model={model} "
            f"(ttl={self.config.ttl}s)"
        )

    async def clear_session(self, session_id: str) -> None:
        """Explicitly remove a session pin (e.g., on logout or explicit reset)."""
        await self._cache.delete(session_id)
        verbose_router_logger.info(
            f"ModelAffinityRouter: cleared session={session_id}"
        )
