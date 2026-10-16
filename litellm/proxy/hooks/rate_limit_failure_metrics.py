"""
Prometheus counter for rate limiter failures.

Lazily registers the counter on first use so proxy deployments without
Prometheus don't pull in the ``prometheus_client`` import at module load.

``limiter_type`` values emitted by callers:

- ``parallel_v3``: max-parallel-requests limiter (v3 Lua script)
- ``v1_dynamic``: v1 model-group dynamic TPM/RPM limiter
- ``v3_dynamic``: v3 saturation-aware dynamic limiter
- ``model_rate_limit_check``: deployment-level static TPM/RPM pre-call check
- ``budget_reservation``: spend-track budget reservation prepare step
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from prometheus_client import Counter

_LIMITER_FAILURE_COUNTER_NAME: Final = "litellm_rate_limiter_failures_total"
_LIMITER_FAILURE_LABELNAMES: Final = ("limiter_type", "error_type")


class _RateLimiterFailureCounterRegistry:
    __slots__ = ("_counter",)

    def __init__(self) -> None:
        self._counter: Counter | None = None  # mutable-ok: Prometheus Counter is inherently mutable metric

    def get_counter(self) -> Counter | None:  # mutable-ok: Prometheus Counter is inherently mutable metric
        if self._counter is not None:
            return self._counter
        try:
            from prometheus_client import Counter as PromCounter
        except ImportError:
            return None
        self._counter = PromCounter(
            _LIMITER_FAILURE_COUNTER_NAME,
            "Total number of rate limiter failures (e.g. Redis unreachable). "
            "Labels let operators distinguish v1 vs v3 and error class.",
            labelnames=_LIMITER_FAILURE_LABELNAMES,
        )
        return self._counter


_registry: Final[_RateLimiterFailureCounterRegistry] = _RateLimiterFailureCounterRegistry()


def increment_rate_limiter_failure_counter(
    limiter_type: str,
    error_type: str,
) -> None:
    counter: Final = _registry.get_counter()
    if counter is None:
        return
    counter.labels(limiter_type=limiter_type, error_type=error_type).inc()
