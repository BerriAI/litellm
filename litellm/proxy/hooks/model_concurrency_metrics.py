"""Prometheus gauge for the model group concurrency queue depth.

``litellm_model_concurrency_queue_size{model_group=...}`` counts requests that
are waiting for a slot at the group's concurrency limit on this worker. A
request that gets a free slot on its first attempt never shows up here.

The gauge is registered lazily, on the first request that has to wait, so
``PROMETHEUS_MULTIPROC_DIR`` is already set when the collector is created.
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Final, Protocol, runtime_checkable

MODEL_CONCURRENCY_QUEUE_SIZE_METRIC: Final = "litellm_model_concurrency_queue_size"


@runtime_checkable
class _GaugeChild(Protocol):
    def inc(self, amount: float = 1) -> None: ...

    def dec(self, amount: float = 1) -> None: ...


@runtime_checkable
class _Gauge(Protocol):
    def labels(self, **labels: str) -> _GaugeChild: ...


@runtime_checkable
class ModelConcurrencyQueueObserver(Protocol):
    def request_queued(self, model_group: str) -> None: ...

    def request_dequeued(self, model_group: str) -> None: ...


@lru_cache(maxsize=1)
def _queue_size_gauge() -> _Gauge | None:
    try:
        from prometheus_client import Gauge
    except ImportError:
        return None
    try:
        if "PROMETHEUS_MULTIPROC_DIR" in os.environ:
            return Gauge(
                MODEL_CONCURRENCY_QUEUE_SIZE_METRIC,
                "Requests waiting for a model group concurrency slot on this worker",
                labelnames=("model_group",),
                multiprocess_mode="livesum",
            )
        return Gauge(
            MODEL_CONCURRENCY_QUEUE_SIZE_METRIC,
            "Requests waiting for a model group concurrency slot on this worker",
            labelnames=("model_group",),
        )
    except ValueError:
        return None


class PrometheusModelConcurrencyQueueObserver:
    def request_queued(self, model_group: str) -> None:
        gauge: Final = _queue_size_gauge()
        if gauge is not None:
            gauge.labels(model_group=model_group).inc()

    def request_dequeued(self, model_group: str) -> None:
        gauge: Final = _queue_size_gauge()
        if gauge is not None:
            gauge.labels(model_group=model_group).dec()


DEFAULT_MODEL_CONCURRENCY_QUEUE_OBSERVER: Final = PrometheusModelConcurrencyQueueObserver()
