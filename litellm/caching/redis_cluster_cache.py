"""
Redis Cluster Cache implementation

Key differences:
- RedisClient NEEDs to be re-used across requests, adds 3000ms latency if it's re-created
"""

import asyncio
import contextvars
import os
import weakref
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Final, TypeAlias, TypeVar

from litellm._logging import verbose_logger
from litellm.caching.redis_cache import RedisCache
from litellm.constants import REDIS_CONNECTION_POOL_TIMEOUT

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span
    from redis.asyncio import Redis
    from redis.asyncio.client import Pipeline

    pipeline = Pipeline
    async_redis_client = Redis
    Span = _Span | Any
else:
    pipeline = Any
    async_redis_client = Any
    Span = Any


_T: Final = TypeVar("_T")


def _connection_limit(max_connections: object) -> int | None:
    try:
        limit: Final = int(max_connections)  # pyright: ignore[reportArgumentType]  # config value, str or int
    except (TypeError, ValueError):
        return None
    return limit if limit > 0 else None


_ClientSemaphores: TypeAlias = weakref.WeakValueDictionary[str, asyncio.Semaphore]


@dataclass(slots=True)
class _SemaphoreRegistry:
    """Weak on both layers: a semaphore binds to its loop once a waiter queues on it, so a strong value would keep a
    closed loop alive. In-flight ``ClusterConnectionGate.run`` calls hold the semaphore they acquired."""

    _per_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, _ClientSemaphores]" = field(
        default_factory=weakref.WeakKeyDictionary, repr=False
    )

    def get(self, client_key: str, limit: int) -> asyncio.Semaphore:
        per_loop: Final = self._per_loop.setdefault(asyncio.get_running_loop(), weakref.WeakValueDictionary())
        existing: Final = per_loop.get(client_key)
        if existing is not None:
            return existing
        created: Final = asyncio.Semaphore(limit)
        per_loop[client_key] = created
        return created


_Admission: TypeAlias = tuple[str, "asyncio.Task[object] | None"]
_SEMAPHORES: Final = _SemaphoreRegistry()
_ADMITTED: Final[contextvars.ContextVar[frozenset[_Admission]]] = contextvars.ContextVar(
    "redis_cluster_gate_admitted", default=frozenset()
)


@dataclass(frozen=True, slots=True)
class ClusterConnectionGate:
    """The async ``RedisCluster`` has no blocking pool, so past ``max_connections`` a node raises
    ``MaxConnectionsError`` instead of waiting and the circuit breaker opens on a healthy Redis. This gate makes
    guarded commands wait for a slot the way the standalone ``BlockingConnectionPool`` does. A command that finds no
    slot within ``wait_timeout`` runs anyway, so a genuinely saturated pool still reaches the client's own error
    handling and the breaker."""

    limit: int
    wait_timeout: float = REDIS_CONNECTION_POOL_TIMEOUT

    @classmethod
    def from_max_connections(cls, max_connections: object) -> "ClusterConnectionGate | None":
        limit: Final = _connection_limit(max_connections)
        return None if limit is None else cls(limit=limit)

    async def run(self, call: Callable[[], Awaitable[_T]], client_key: str) -> _T:
        admission: Final[_Admission] = (client_key, asyncio.current_task())
        if admission in _ADMITTED.get():
            return await call()
        semaphore: Final = _SEMAPHORES.get(client_key, self.limit)
        try:
            await asyncio.wait_for(semaphore.acquire(), timeout=self.wait_timeout)
        except asyncio.TimeoutError:
            verbose_logger.debug(
                "Redis Cluster: no free connection slot within %ss (max_connections=%s), sending anyway",
                self.wait_timeout,
                self.limit,
            )
            return await call()
        token: Final = _ADMITTED.set(_ADMITTED.get() | {admission})
        try:
            return await call()
        finally:
            _ADMITTED.reset(token)
            semaphore.release()


class RedisClusterCache(RedisCache):
    def __init__(self, *args, max_connections: object = None, connection_gate: object = None, **kwargs):
        pool_kwargs: Final = {} if max_connections is None else {"max_connections": max_connections}
        super().__init__(*args, **pool_kwargs, **kwargs)
        self._connection_gate: Final = (
            connection_gate
            if isinstance(connection_gate, ClusterConnectionGate)
            else ClusterConnectionGate.from_max_connections(
                os.environ.get("REDIS_MAX_CONNECTIONS") if max_connections is None else max_connections
            )
        )

    def _admit_async_call(self, call: Callable[[], Awaitable[_T]]) -> Awaitable[_T]:
        if self._connection_gate is None:
            return call()
        return self._connection_gate.run(call, self._get_async_client_cache_key())

    def _run_redis_mget_operation(self, keys: list[str]) -> list[Any]:
        """
        Overrides `_run_redis_mget_operation` in redis_cache.py
        """
        return self.redis_client.mget_nonatomic(keys=keys)

    async def _async_run_redis_mget_operation(self, keys: list[str]) -> list[Any]:
        """
        Overrides `_async_run_redis_mget_operation` in redis_cache.py
        """
        async_redis_cluster_client: Final = self.init_async_client()
        return await async_redis_cluster_client.mget_nonatomic(keys=keys)

    async def test_connection(self) -> dict:
        """
        Test the Redis Cluster connection.

        Returns:
            dict: {"status": "success" | "failed", "message": str, "error": Optional[str]}
        """
        try:
            from .._redis import get_redis_async_client

            redis_client: Final = get_redis_async_client(**self.redis_kwargs)

            # Test the connection
            ping_result: Final = await redis_client.ping()

            # Close the connection
            await redis_client.aclose()

            if ping_result:
                return {
                    "status": "success",
                    "message": "Redis Cluster connection test successful",
                }
            else:
                return {
                    "status": "failed",
                    "message": "Redis Cluster ping returned False",
                }
        except Exception as e:
            from litellm._logging import verbose_logger

            verbose_logger.error("Redis Cluster connection test failed: %s", e)
            return {
                "status": "failed",
                "message": f"Redis Cluster connection failed: {e}",
                "error": str(e),
            }
