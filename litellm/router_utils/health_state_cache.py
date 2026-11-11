"""
Wrapper around router cache for health-check-driven routing.

Stores per-deployment health state from background health checks
and exposes it for router candidate filtering.
"""

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from pydantic import TypeAdapter
from typing_extensions import TypedDict

from litellm import verbose_logger
from litellm._internal_context import with_service_target
from litellm.caching.caching import DualCache
from litellm.caching.redis_cache import RedisCircuitBreakerOpenError
from litellm.constants import HEALTH_CHECKS_TARGET

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


_CACHED_HEALTH_SNAPSHOT: Final = TypeAdapter(object)


class DeploymentHealthStateValue(TypedDict):
    is_healthy: bool
    timestamp: float
    reason: str


@with_service_target(HEALTH_CHECKS_TARGET)
def _read_shared_health_snapshot(cache: DualCache, key: str) -> object:
    redis_cache: Final = cache.redis_cache
    if redis_cache is None:
        return None
    try:
        return redis_cache.get_cache(key)
    except RedisCircuitBreakerOpenError:
        return None


class DeploymentHealthCache:
    """
    Cache for deployment health states produced by background health checks.

    Stores a single dict mapping deployment_id -> DeploymentHealthStateValue.
    Staleness is enforced at read time: entries older than staleness_threshold
    are treated as healthy (unknown).
    """

    CACHE_KEY = "litellm:health_check:deployment_health_state"

    def __init__(self, cache: DualCache, staleness_threshold: float):
        self.cache = cache
        self.staleness_threshold = staleness_threshold

    @with_service_target(HEALTH_CHECKS_TARGET)
    def set_deployment_health_states(self, states: dict[str, DeploymentHealthStateValue]) -> None:
        """Merge the given states into the shared cache entry, pruning expired ones.

        Merging instead of replacing lets writers probing different deployment
        scopes (e.g. pods with different background health check allowlists)
        coexist on the one shared entry without erasing each other's results.
        The snapshot is read from Redis when available, since a pod-local read
        would only ever see this writer's own previous merge. When the Redis
        read comes back empty (a miss, a swallowed connection error, or a read
        refused by the open circuit breaker), the pod-local copy of the last
        merge is used so peers are not erased.
        """
        try:
            redis_raw: Final = _read_shared_health_snapshot(self.cache, self.CACHE_KEY)
            raw: Final = redis_raw if isinstance(redis_raw, dict) else self.cache.get_cache(key=self.CACHE_KEY)
            existing: Final = raw if isinstance(raw, dict) else {}
            expiry_seconds: Final = self.staleness_threshold * 1.5
            now: Final = time.time()
            merged: Final = {
                model_id: state
                for model_id, state in {**existing, **states}.items()
                if isinstance(state, dict) and (now - _timestamp_of(state)) < expiry_seconds  # pyright: ignore[reportUnknownArgumentType]  # a cached record
            }
            self.cache.set_cache(
                key=self.CACHE_KEY,
                value=merged,
                ttl=int(expiry_seconds),
            )
        except Exception as e:
            verbose_logger.error(
                "DeploymentHealthCache::set_deployment_health_states - Exception: %s",
                str(e),
            )

    def _extract_unhealthy_ids(self, raw: object) -> set[str]:
        """Given raw cache value, return set of non-stale unhealthy deployment IDs."""
        if not raw or not isinstance(raw, dict):
            return set()
        now: Final = time.time()
        return {
            model_id
            for model_id, state in raw.items()
            if isinstance(state, dict)
            and not state.get("is_healthy", True)
            and (now - _timestamp_of(state)) < self.staleness_threshold  # pyright: ignore[reportUnknownArgumentType]  # a cached record
        }

    @with_service_target(HEALTH_CHECKS_TARGET)
    async def async_get_unhealthy_deployment_ids(self, parent_otel_span: Span | None = None) -> set[str]:
        """Return set of deployment IDs currently marked unhealthy and not stale."""
        try:
            raw: Final = _CACHED_HEALTH_SNAPSHOT.validate_python(await self.cache.async_get_cache(key=self.CACHE_KEY))
            return self._extract_unhealthy_ids(raw)
        except Exception as e:
            verbose_logger.debug(
                "DeploymentHealthCache::async_get_unhealthy_deployment_ids - Exception: %s",
                str(e),
            )
            return set()

    @with_service_target(HEALTH_CHECKS_TARGET)
    def get_unhealthy_deployment_ids(self, parent_otel_span: Span | None = None) -> set[str]:
        """Sync version: return set of deployment IDs currently marked unhealthy and not stale."""
        try:
            raw: Final = _CACHED_HEALTH_SNAPSHOT.validate_python(self.cache.get_cache(key=self.CACHE_KEY))
            return self._extract_unhealthy_ids(raw)
        except Exception as e:
            verbose_logger.debug(
                "DeploymentHealthCache::get_unhealthy_deployment_ids - Exception: %s",
                str(e),
            )
            return set()


def _timestamp_of(state: Mapping[str, object]) -> float:
    """A health record's timestamp; one that is no number reads as 0 (stale), so a bad record
    drops out alone instead of turning health routing off for every deployment."""
    timestamp: Final = state.get("timestamp", 0)
    return float(timestamp) if isinstance(timestamp, int | float) and not isinstance(timestamp, bool) else 0.0
