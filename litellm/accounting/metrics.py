from typing import Final, Literal

from prometheus_client import Counter, Gauge, Histogram

from litellm.repositories.accounting_delivery_repository import DeliveryMetrics

_PENDING: Final = Gauge("litellm_accounting_pending", "Pending SpendLogs deliveries", multiprocess_mode="livemax")
_PARKED: Final = Gauge("litellm_accounting_parked", "Parked SpendLogs deliveries", multiprocess_mode="livemax")
_OLDEST: Final = Gauge(
    "litellm_accounting_oldest_pending_seconds",
    "Age of the oldest pending SpendLogs delivery",
    multiprocess_mode="livemax",
)
_BATCH_TIME: Final = Histogram("litellm_accounting_batch_seconds", "SpendLogs delivery worker batch time")
_POSTPONED: Final = Counter(
    "litellm_accounting_cleanup_postponed", "Cleanup deferred for unfinished accounting deliveries", ("kind",)
)


def record_deliveries(metrics: DeliveryMetrics, seconds: float) -> None:
    _PENDING.set(metrics.pending_count)
    _PARKED.set(metrics.parked_count)
    _OLDEST.set(max(0.0, metrics.oldest_pending_seconds))
    _BATCH_TIME.observe(seconds)


def record_postponed_cleanup(kind: Literal["rows", "partition"]) -> None:
    _POSTPONED.labels(kind=kind).inc()
