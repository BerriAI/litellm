from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

from .audit import Decision


@dataclass(frozen=True)
class MetricsSnapshot:
    """A point-in-time read of the auth_v2 authz metrics, for a /metrics export."""

    # "decision:resource:action" -> count
    decisions: Dict[str, int]
    authz_latency_count: int
    authz_latency_sum_seconds: float
    policy_cache_hits: int
    policy_cache_misses: int


class _Metrics:
    """In-process authz metrics. Dependency-free so it never burdens the hot path;
    a Prometheus/OTel exporter reads snapshot() at the edge."""

    def __init__(self) -> None:
        self._decisions: Dict[Tuple[str, str, str], int] = {}
        self._latency_count: int = 0
        self._latency_sum: float = 0.0
        self._cache_hits: int = 0
        self._cache_misses: int = 0

    def observe_decision(self, decision: Decision, resource: str, action: str) -> None:
        key = (decision.value, resource, action)
        self._decisions[key] = self._decisions.get(key, 0) + 1

    def observe_latency(self, seconds: float) -> None:
        self._latency_count += 1
        self._latency_sum += seconds

    def record_cache(self, hit: bool) -> None:
        if hit:
            self._cache_hits += 1
        else:
            self._cache_misses += 1

    def snapshot(self) -> MetricsSnapshot:
        return MetricsSnapshot(
            decisions={
                f"{d}:{r}:{a}": count for (d, r, a), count in self._decisions.items()
            },
            authz_latency_count=self._latency_count,
            authz_latency_sum_seconds=self._latency_sum,
            policy_cache_hits=self._cache_hits,
            policy_cache_misses=self._cache_misses,
        )

    def reset(self) -> None:
        self._decisions.clear()
        self._latency_count = 0
        self._latency_sum = 0.0
        self._cache_hits = 0
        self._cache_misses = 0


metrics = _Metrics()
