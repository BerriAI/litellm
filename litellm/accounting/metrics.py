from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

from litellm.repositories.accounting_delivery_repository import DeliveryMetrics

if TYPE_CHECKING:
    from prometheus_client import Gauge, Histogram


@dataclass(frozen=True, slots=True)
class _Instruments:
    pending: "Gauge"
    parked: "Gauge"
    oldest: "Gauge"
    batch_time: "Histogram"
    batch_count: "Gauge"
    table_bytes: "Gauge"


def _instruments() -> _Instruments | None:
    try:
        # prometheus-client ships only in the proxy-runtime extra, so a plain litellm[proxy] install lands here
        from prometheus_client import Gauge, Histogram
    except ImportError:
        return None
    return _Instruments(
        pending=Gauge("litellm_accounting_pending", "Pending SpendLogs deliveries", multiprocess_mode="livemax"),
        parked=Gauge("litellm_accounting_parked", "Parked SpendLogs deliveries", multiprocess_mode="livemax"),
        oldest=Gauge(
            "litellm_accounting_oldest_pending_seconds",
            "Age of the oldest pending SpendLogs delivery",
            multiprocess_mode="livemax",
        ),
        batch_time=Histogram("litellm_accounting_batch_seconds", "SpendLogs delivery worker batch time"),
        batch_count=Gauge(
            "litellm_accounting_batch_deliveries", "Retained batch deliveries", multiprocess_mode="livemax"
        ),
        table_bytes=Gauge(
            "litellm_accounting_delivery_table_bytes", "Delivery table size in bytes", multiprocess_mode="livemax"
        ),
    )


_INSTRUMENTS: Final = _instruments()


def record_deliveries(metrics: DeliveryMetrics, seconds: float) -> None:
    if _INSTRUMENTS is None:
        return
    _INSTRUMENTS.pending.set(metrics.pending_count)
    _INSTRUMENTS.parked.set(metrics.parked_count)
    _INSTRUMENTS.oldest.set(max(0.0, metrics.oldest_pending_seconds))
    _INSTRUMENTS.batch_time.observe(seconds)
    _INSTRUMENTS.batch_count.set(metrics.batch_count)
    _INSTRUMENTS.table_bytes.set(metrics.table_bytes)
