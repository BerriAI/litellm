from prometheus_client import REGISTRY

from litellm.accounting.metrics import record_deliveries
from litellm.repositories.accounting_delivery_repository import DeliveryMetrics


def test_delivery_metrics_publish_backlog_and_clamp_clock_skew() -> None:
    batches_before = REGISTRY.get_sample_value("litellm_accounting_batch_seconds_count") or 0.0

    record_deliveries(
        DeliveryMetrics(pending_count=4, parked_count=1, oldest_pending_seconds=-3.0, batch_count=7, table_bytes=8192),
        0.5,
    )

    assert {
        name: REGISTRY.get_sample_value(name)
        for name in (
            "litellm_accounting_pending",
            "litellm_accounting_parked",
            "litellm_accounting_oldest_pending_seconds",
            "litellm_accounting_batch_deliveries",
            "litellm_accounting_delivery_table_bytes",
        )
    } == {
        "litellm_accounting_pending": 4.0,
        "litellm_accounting_parked": 1.0,
        "litellm_accounting_oldest_pending_seconds": 0.0,
        "litellm_accounting_batch_deliveries": 7.0,
        "litellm_accounting_delivery_table_bytes": 8192.0,
    }
    assert REGISTRY.get_sample_value("litellm_accounting_batch_seconds_count") == batches_before + 1
