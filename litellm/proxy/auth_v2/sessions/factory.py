from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Optional, cast

from litellm._redis import get_redis_async_client

from litellm.proxy.auth_v2.sessions.base import StateStore, StateValue
from litellm.proxy.auth_v2.sessions.memory import InMemoryStateStore
from litellm.proxy.auth_v2.sessions.redis import RedisStateStore

if TYPE_CHECKING:
    from redis.asyncio import Redis

logger = logging.getLogger("litellm.proxy.auth_v2.sessions")

_REDIS_ENV_SIGNALS = (
    "REDIS_URL",
    "REDIS_HOST",
    "REDIS_CLUSTER_NODES",
    "REDIS_SENTINEL_NODES",
)


async def _reachable(client: "Redis") -> bool:
    try:
        return bool(await client.ping())
    except Exception:
        return False


def _default_redis_client() -> Optional["Redis"]:
    if not any(os.getenv(signal) for signal in _REDIS_ENV_SIGNALS):
        return None
    try:
        return cast("Redis", get_redis_async_client())
    except Exception:
        logger.warning("auth_v2 state layer could not build a Redis client", exc_info=True)
        return None


class StateBackend:
    """Hands out namespaced state stores backed by Redis when reachable, else memory.

    The Redis-vs-memory choice is made once, at ``connect`` time, and held for the
    backend's lifetime. We deliberately do not fail over per operation: silently
    moving a live session from Redis to a local dict would strand it on one worker
    and lose it the moment another worker serves the next request.

    Inject the client for tests or to share the proxy's existing connection; the
    default builder only fires when Redis is configured via the environment.
    """

    def __init__(self, redis_client: Optional["Redis"]) -> None:
        self._redis = redis_client

    @classmethod
    async def connect(cls, redis_client: Optional["Redis"] = None) -> "StateBackend":
        client = redis_client if redis_client is not None else _default_redis_client()
        if client is not None and await _reachable(client):
            logger.info("auth_v2 state layer using Redis backend")
            return cls(client)
        logger.info("auth_v2 state layer using in-memory backend")
        return cls(None)

    @property
    def using_redis(self) -> bool:
        return self._redis is not None

    def store(self, namespace: str, *, default_ttl: int) -> StateStore[StateValue]:
        """Return a typed store for ``namespace``.

        The value schema is taken from the call site's annotation, e.g.
        ``sessions: StateStore[SessionState] = backend.store("sessions", default_ttl=3600)``.
        """
        if self._redis is not None:
            return RedisStateStore(self._redis, namespace, default_ttl)
        return InMemoryStateStore(namespace, default_ttl)
