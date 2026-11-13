"""
Health-check-driven routing state, kept in the native ``DeploymentHealthStore``.

Stores per-deployment health state from background health checks
and exposes it for router candidate filtering.
"""

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from typing_extensions import ReadOnly, TypedDict

from litellm import verbose_logger
from litellm.redis.stores.deployment_health import DeploymentHealthStates, DeploymentHealthStore

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = _Span
else:
    Span = Any


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

    def set_deployment_health_states(self, states: DeploymentHealthStates) -> None:
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
            shared: Final = self.store.get_shared_sync()
            existing: Final = shared if shared is not None else self.store.get_sync() or {}
            expiry_seconds: Final = self.staleness_threshold * 1.5
            now: Final = time.time()
            self.store.set_sync(_unexpired({**existing, **states}, now - expiry_seconds), int(expiry_seconds))
        except Exception as e:
            verbose_logger.error(
                "DeploymentHealthCache::set_deployment_health_states - Exception: %s",
                str(e),
            )

    def _extract_unhealthy_ids(self, states: DeploymentHealthStates | None) -> set[str]:
        """The non-stale unhealthy deployment IDs."""
        now: Final = time.time()
        return {
            model_id
            for model_id, state in (states or {}).items()
            if not state["is_healthy"] and now - state["timestamp"] < self.staleness_threshold
        }

    async def async_get_states(self) -> DeploymentHealthStates | None:
        """Every deployment's state, as the router core's health filter reads it; a failed read is None."""
        try:
            return await self.store.get()
        except Exception as e:  # noqa: BLE001  # a failed read counts as no health state, as the DualCache read it replaced swallowed errors
            verbose_logger.debug("DeploymentHealthCache::async_get_states - Exception: %s", str(e))
            return None

    def get_states(self) -> DeploymentHealthStates | None:
        """``async_get_states`` for a sync call."""
        try:
            return self.store.get_sync()
        except Exception as e:  # noqa: BLE001  # a failed read counts as no health state, as the DualCache read it replaced swallowed errors
            verbose_logger.debug("DeploymentHealthCache::get_states - Exception: %s", str(e))
            return None

    async def async_get_unhealthy_deployment_ids(self, parent_otel_span: Span | None = None) -> set[str]:
        """Return set of deployment IDs currently marked unhealthy and not stale."""
        try:
            return self._extract_unhealthy_ids(await self.store.get())
        except Exception as e:
            verbose_logger.debug(
                "DeploymentHealthCache::async_get_unhealthy_deployment_ids - Exception: %s",
                str(e),
            )
            return set()

    def get_unhealthy_deployment_ids(self, parent_otel_span: Span | None = None) -> set[str]:
        """Sync version: return set of deployment IDs currently marked unhealthy and not stale."""
        try:
            return self._extract_unhealthy_ids(self.store.get_sync())
        except Exception as e:
            verbose_logger.debug(
                "DeploymentHealthCache::get_unhealthy_deployment_ids - Exception: %s",
                str(e),
            )
            return set()


class DeploymentHealthCheck(TypedDict):
    is_healthy: ReadOnly[bool]
    timestamp: ReadOnly[float]


def health_states_of(states: DeploymentHealthStates | None) -> Mapping[str, DeploymentHealthCheck]:
    """The cached states as router-core reads them, which is everything but the reason."""
    return {
        model_id: {"is_healthy": state["is_healthy"], "timestamp": float(state["timestamp"])}
        for model_id, state in (states or {}).items()
    }


def _unexpired(states: DeploymentHealthStates, cutoff: float) -> DeploymentHealthStates:
    return {model_id: state for model_id, state in states.items() if state["timestamp"] > cutoff}
