"""
Health-check-driven routing state, kept in the native ``DeploymentHealthStore``.

Stores per-deployment health state from background health checks
and exposes it for router candidate filtering.
"""

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from typing_extensions import TypedDict

from litellm import verbose_logger
from litellm.redis.stores.deployment_health import DeploymentHealthStore

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


class DeploymentHealthStateValue(TypedDict):
    is_healthy: bool
    timestamp: float
    reason: str


class DeploymentHealthCache:
    """
    Cache for deployment health states produced by background health checks.

    Stores a single dict mapping deployment_id -> DeploymentHealthStateValue.
    Staleness is enforced at read time: entries older than staleness_threshold
    are treated as healthy (unknown).
    """

    def __init__(self, store: DeploymentHealthStore, staleness_threshold: float):
        self.store: Final = store
        self.staleness_threshold = staleness_threshold

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
            redis_raw: Final = self.store.get_shared_sync()
            raw: Final = redis_raw if isinstance(redis_raw, dict) else self.store.get_sync()
            existing: Final = raw if isinstance(raw, dict) else {}
            expiry_seconds: Final = self.staleness_threshold * 1.5
            now: Final = time.time()
            merged: Final = {
                model_id: state
                for model_id, state in {**existing, **states}.items()
                if isinstance(state, dict) and (now - _timestamp_of(state)) < expiry_seconds  # pyright: ignore[reportUnknownArgumentType]  # a cached record
            }
            self.store.set_sync(merged, int(expiry_seconds))
        except Exception as e:
            verbose_logger.error(
                "DeploymentHealthCache::set_deployment_health_states - Exception: %s",
                str(e),
            )

    def _extract_unhealthy_ids(self, raw: Any) -> set[str]:
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

    async def async_get_states(self) -> object:
        """Every deployment's raw state, as the router core's health filter reads it; a failed read is None."""
        try:
            return await self.store.get()
        except Exception as e:  # noqa: BLE001  # a failed read counts as no health state, as the DualCache read it replaced swallowed errors
            verbose_logger.debug("DeploymentHealthCache::async_get_states - Exception: %s", str(e))
            return None

    def get_states(self) -> object:
        """``async_get_states`` for a sync call."""
        try:
            return self.store.get_sync()
        except Exception as e:  # noqa: BLE001  # a failed read counts as no health state, as the DualCache read it replaced swallowed errors
            verbose_logger.debug("DeploymentHealthCache::get_states - Exception: %s", str(e))
            return None

    async def async_get_unhealthy_deployment_ids(self, parent_otel_span: Span | None = None) -> set[str]:
        """Return set of deployment IDs currently marked unhealthy and not stale."""
        try:
            raw: Final = await self.store.get()
            return self._extract_unhealthy_ids(raw)
        except Exception as e:
            verbose_logger.debug(
                "DeploymentHealthCache::async_get_unhealthy_deployment_ids - Exception: %s",
                str(e),
            )
            return set()

    def get_unhealthy_deployment_ids(self, parent_otel_span: Span | None = None) -> set[str]:
        """Sync version: return set of deployment IDs currently marked unhealthy and not stale."""
        try:
            raw: Final = self.store.get_sync()
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
