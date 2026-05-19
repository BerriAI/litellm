"""
Capability-dimension Prometheus counters (S6-02).

Side-channel from the spend-log write path so dashboards can slice
``rate / latency / spend`` by ``(app_id, entity_type)`` without leaning
on the SQL spend_logs table.

Two counters are exposed:
    litellm_capability_requests_total{app_id, entity_type, entity_id}
    litellm_capability_spend_total   {app_id, entity_type, entity_id}

Cardinality is bounded by ``BoundedPrometheusSeriesTracker`` with the
limit calibrated for what we expect in practice:
    app_id: ~50 apps in any single org
    entity_type: 4 enum values
    entity_id: bounded to ~5 per (app, entity_type) — most apps churn
               through < 20 entities/day, but we keep an explicit 1000
               ceiling per metric and a 1h idle-TTL eviction so a runaway
               doesn't OOM the Prometheus client.

This module is best-effort: every public function swallows exceptions
and silently no-ops. A broken metric must NEVER fail a paying request.
"""

from __future__ import annotations

import os
from typing import Optional

from litellm._logging import verbose_logger
from litellm.integrations.prometheus_helpers.bounded_prometheus_series_tracker import (
    BoundedPrometheusSeriesTracker,
)

_LABEL_NAMES = ("app_id", "entity_type", "entity_id")
_MAX_SERIES = int(os.environ.get("LITELLM_CAPABILITY_METRIC_MAX_SERIES", "1000"))
_TTL_SECONDS = int(os.environ.get("LITELLM_CAPABILITY_METRIC_TTL_SECONDS", "3600"))
_CLEANUP_INTERVAL_SECONDS = int(
    os.environ.get("LITELLM_CAPABILITY_METRIC_CLEANUP_INTERVAL", "60")
)

_requests_counter = None
_spend_counter = None
_tracker = BoundedPrometheusSeriesTracker()


_private_registry = None


def _lazy_init() -> bool:
    """Create counters on first use. Returns False when prometheus_client
    isn't importable — caller treats the call as a no-op.

    Counters live in a **private** ``CollectorRegistry`` so test resets and
    inter-module Counter registrations cannot collide with each other.
    The proxy's ``/metrics`` route exposes the default REGISTRY by default;
    callers that want capability metrics scraped should mount this module's
    registry via ``generate_latest(get_capability_registry())``.
    """
    global _requests_counter, _spend_counter, _private_registry
    if _requests_counter is not None and _spend_counter is not None:
        return True
    try:
        from prometheus_client import CollectorRegistry, Counter  # type: ignore
    except Exception:
        return False
    if _private_registry is None:
        _private_registry = CollectorRegistry()
    _requests_counter = Counter(
        "litellm_capability_requests_total",
        "Number of capability invocations grouped by app + entity.",
        labelnames=_LABEL_NAMES,
        registry=_private_registry,
    )
    _spend_counter = Counter(
        "litellm_capability_spend_total",
        "Spend (USD) accrued per capability invocation grouped by app + entity.",
        labelnames=_LABEL_NAMES,
        registry=_private_registry,
    )
    return True


def get_capability_registry():
    """Return the private CollectorRegistry holding the capability counters.

    Used by the ``/metrics`` exposition path so capability counters appear
    alongside the default LiteLLM metrics — mount via
    ``generate_latest(get_capability_registry())`` and concatenate.
    """
    _lazy_init()
    return _private_registry


def record_capability_call(
    *,
    app_id: Optional[str],
    entity_type: Optional[str],
    entity_id: Optional[str],
    spend: float = 0.0,
) -> None:
    """Stamp one capability invocation. Safe to call from anywhere — never raises."""
    if entity_type is None:
        return  # nothing to attribute
    if not _lazy_init():
        return
    label_values = (
        app_id or "none",
        entity_type,
        entity_id or "none",
    )
    try:
        _requests_counter.labels(*label_values).inc()  # type: ignore[union-attr]
        if spend:
            _spend_counter.labels(*label_values).inc(float(spend))  # type: ignore[union-attr]
        # Track for bounded eviction.
        _tracker.track_series(
            metric=_requests_counter,
            metric_name="litellm_capability_requests_total",
            label_values=label_values,
            max_series=_MAX_SERIES,
            ttl_seconds=_TTL_SECONDS,
            cleanup_interval_seconds=_CLEANUP_INTERVAL_SECONDS,
        )
        _tracker.track_series(
            metric=_spend_counter,
            metric_name="litellm_capability_spend_total",
            label_values=label_values,
            max_series=_MAX_SERIES,
            ttl_seconds=_TTL_SECONDS,
            cleanup_interval_seconds=_CLEANUP_INTERVAL_SECONDS,
        )
    except Exception as e:  # pragma: no cover — metrics must never break logging
        verbose_logger.debug("record_capability_call failed: %s", e)


def _reset_for_tests() -> None:
    """Reset internal state — only used by unit tests."""
    global _requests_counter, _spend_counter, _tracker, _private_registry
    _requests_counter = None
    _spend_counter = None
    _private_registry = None
    _tracker = BoundedPrometheusSeriesTracker()
