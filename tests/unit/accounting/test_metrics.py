import importlib.util
import sys

import pytest
from prometheus_client.metrics import MetricWrapperBase

from litellm.accounting import metrics
from litellm.accounting.metrics import record_deliveries
from litellm.repositories.accounting_delivery_repository import DeliveryMetrics

_DELIVERY_METRICS = DeliveryMetrics(
    pending_count=4, parked_count=1, oldest_pending_seconds=-3.0, batch_count=7, table_bytes=8192
)


def _sample(metric: MetricWrapperBase, suffix: str = "") -> float:
    family = metric.collect()[0]
    return next(sample.value for sample in family.samples if sample.name == family.name + suffix)


def test_delivery_metrics_publish_backlog_and_clamp_clock_skew() -> None:
    instruments = metrics._INSTRUMENTS
    assert instruments is not None
    batches_before = _sample(instruments.batch_time, "_count")

    record_deliveries(_DELIVERY_METRICS, 0.5)

    assert (
        _sample(instruments.pending),
        _sample(instruments.parked),
        _sample(instruments.oldest),
        _sample(instruments.batch_count),
        _sample(instruments.table_bytes),
    ) == (4.0, 1.0, 0.0, 7.0, 8192.0)
    assert _sample(instruments.batch_time, "_count") == batches_before + 1


def test_delivery_metrics_are_skipped_when_prometheus_client_is_not_installed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "prometheus_client", None)
    spec = importlib.util.spec_from_file_location("accounting_metrics_without_prometheus", metrics.__file__)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)

    spec.loader.exec_module(module)

    assert module.record_deliveries(_DELIVERY_METRICS, 0.5) is None
