"""
Wrapper around router cache for health-check-driven routing.

Stores per-deployment health state from background health checks
and exposes it for router candidate filtering.
"""

import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Set, Union

from typing_extensions import TypedDict

from litellm import verbose_logger
from litellm.caching.caching import DualCache

if TYPE_CHECKING:
    from opentelemetry.trace import Span as _Span

    Span = Union[_Span, Any]
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

    CACHE_KEY = "litellm:health_check:deployment_health_state"

    def __init__(self, cache: DualCache, staleness_threshold: float):
        self.cache = cache
        self.staleness_threshold = staleness_threshold

    def set_deployment_health_states(
        self, states: Dict[str, DeploymentHealthStateValue]
    ) -> None:
        """Bulk-write all deployment health states as a single cache entry."""
        try:
            self.cache.set_cache(
                key=self.CACHE_KEY,
                value=states,
                ttl=int(self.staleness_threshold * 1.5),
            )
        except Exception as e:
            verbose_logger.error(
                "DeploymentHealthCache::set_deployment_health_states - Exception: %s",
                str(e),
            )

    def _extract_unhealthy_ids(self, raw: Any) -> Set[str]:
        """Given raw cache value, return set of non-stale unhealthy deployment IDs."""
        if not raw or not isinstance(raw, dict):
            return set()
        now = time.time()
        return {
            model_id
            for model_id, state in raw.items()
            if isinstance(state, dict)
            and not state.get("is_healthy", True)
            and (now - state.get("timestamp", 0)) < self.staleness_threshold
        }

    async def async_get_unhealthy_deployment_ids(
        self, parent_otel_span: Optional[Span] = None
    ) -> Set[str]:
        """Return set of deployment IDs currently marked unhealthy and not stale."""
        try:
            raw = await self.cache.async_get_cache(key=self.CACHE_KEY)
            return self._extract_unhealthy_ids(raw)
        except Exception as e:
            verbose_logger.debug(
                "DeploymentHealthCache::async_get_unhealthy_deployment_ids - Exception: %s",
                str(e),
            )
            return set()

    def get_unhealthy_deployment_ids(
        self, parent_otel_span: Optional[Span] = None
    ) -> Set[str]:
        """Sync version: return set of deployment IDs currently marked unhealthy and not stale."""
        try:
            raw = self.cache.get_cache(key=self.CACHE_KEY)
            return self._extract_unhealthy_ids(raw)
        except Exception as e:
            verbose_logger.debug(
                "DeploymentHealthCache::get_unhealthy_deployment_ids - Exception: %s",
                str(e),
            )
            return set()
