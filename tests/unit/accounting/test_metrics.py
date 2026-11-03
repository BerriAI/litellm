from prometheus_client.metrics import MetricWrapperBase

from litellm.accounting import metrics
from litellm.accounting.metrics import record_deliveries
from litellm.repositories.accounting_delivery_repository import DeliveryMetrics


def _sample(metric: MetricWrapperBase, suffix: str = "") -> float:
    family = metric.collect()[0]
    return next(sample.value for sample in family.samples if sample.name == family.name + suffix)


def test_delivery_metrics_publish_backlog_and_clamp_clock_skew() -> None:
    batches_before = _sample(metrics._BATCH_TIME, "_count")

    record_deliveries(
        DeliveryMetrics(pending_count=4, parked_count=1, oldest_pending_seconds=-3.0, batch_count=7, table_bytes=8192),
        0.5,
    )

    assert (
        _sample(metrics._PENDING),
        _sample(metrics._PARKED),
        _sample(metrics._OLDEST),
        _sample(metrics._BATCH_COUNT),
        _sample(metrics._TABLE_BYTES),
    ) == (4.0, 1.0, 0.0, 7.0, 8192.0)
    assert _sample(metrics._BATCH_TIME, "_count") == batches_before + 1
