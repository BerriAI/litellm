"""
Health-check-driven routing state, kept in the native ``DeploymentHealthStore``.

Stores per-deployment health state from background health checks
and exposes it for router candidate filtering.
"""

import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final

from pydantic import TypeAdapter
from typing_extensions import ReadOnly, TypedDict

from litellm import verbose_logger
from litellm.litellm_core_utils.core_helpers import as_str_mapping
from litellm.redis.stores.deployment_health import DeploymentHealthStore

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
            raw: Final = _CACHED_HEALTH_SNAPSHOT.validate_python(
                redis_raw if isinstance(redis_raw, dict) else self.store.get_sync()
            )
            existing: Final = as_str_mapping(raw) or {}
            expiry_seconds: Final = self.staleness_threshold * 1.5
            now: Final = time.time()
            self.store.set_sync(_unexpired({**existing, **states}, now - expiry_seconds), int(expiry_seconds))
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
            raw: Final = _CACHED_HEALTH_SNAPSHOT.validate_python(await self.store.get())
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
            raw: Final = _CACHED_HEALTH_SNAPSHOT.validate_python(self.store.get_sync())
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


class DeploymentHealthCheck(TypedDict):
    is_healthy: ReadOnly[bool]
    timestamp: ReadOnly[float]


def health_states_of(raw: object) -> Mapping[str, DeploymentHealthCheck]:
    """The cached states as ``_extract_unhealthy_ids`` reads them: any falsy ``is_healthy`` is unhealthy, a
    record without one is healthy, and what is no record is dropped."""
    records: Final = ((model_id, as_str_mapping(state)) for model_id, state in (as_str_mapping(raw) or {}).items())
    return {
        model_id: {"is_healthy": bool(state.get("is_healthy", True)), "timestamp": _timestamp_of(state)}
        for model_id, state in records
        if state is not None
    }


def _unexpired(records: Mapping[str, object], cutoff: float) -> Mapping[str, object]:
    return {
        model_id: state
        for model_id, state in records.items()
        if isinstance(state, dict) and _timestamp_of(state) > cutoff  # pyright: ignore[reportUnknownArgumentType]  # a cached record
    }
