"""The Rust gateway declares the metric families `PrometheusLogger` declares with its default
settings from `families.json`, which this checks against the logger as it is now."""

import json
import math
from collections.abc import Generator, Mapping, Sequence
from pathlib import Path
from typing import Final, get_args

import prometheus_client
import pytest
from prometheus_client import REGISTRY, CollectorRegistry

from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS, PrometheusMetricLabels

DEFAULT_BUCKETS: Final = prometheus_client.Histogram.DEFAULT_BUCKETS
FAMILIES: Final = Path(__file__).parents[3] / "litellm/rust/crates/callbacks-prometheus/families.json"


@pytest.fixture(autouse=True)
def fresh_registry() -> Generator[None]:
    for collector in list(REGISTRY._collector_to_names):  # pyright: ignore[reportPrivateUsage]  # the client keeps no public list
        REGISTRY.unregister(collector)
    yield
    for collector in list(REGISTRY._collector_to_names):  # pyright: ignore[reportPrivateUsage]  # the client keeps no public list
        REGISTRY.unregister(collector)


def _declared(monkeypatch: pytest.MonkeyPatch) -> Mapping[str, object]:
    captured: Final[list[Mapping[str, object]]] = []

    def recording(metric_class: type, kind: str) -> type:
        def construct(*args: object, **kwargs: object) -> object:
            labelnames: Final = kwargs.get("labelnames") or ()
            entry: Final[dict[str, object]] = {
                "declared": str(args[0] if args else kwargs.get("name", "")),
                "documentation": str(args[1] if len(args) > 1 else kwargs.get("documentation", "")),
                "kind": kind,
                "labelnames": [str(name) for name in labelnames],  # pyright: ignore[reportGeneralTypeIssues]  # labelnames is iterable
            }
            if kind == "histogram":
                bounds: Final = kwargs.get("buckets", DEFAULT_BUCKETS)
                entry["buckets"] = [float(bound) for bound in bounds if not math.isinf(float(bound))]  # pyright: ignore[reportGeneralTypeIssues]  # buckets is iterable
            captured.append(entry)
            return metric_class(*args, **{**kwargs, "registry": CollectorRegistry()})

        return construct  # pyright: ignore[reportReturnType]  # stands in for the class at its call sites

    for name, kind in (("Counter", "counter"), ("Gauge", "gauge"), ("Histogram", "histogram")):
        monkeypatch.setattr(prometheus_client, name, recording(getattr(prometheus_client, name), kind))
    logger: Final = PrometheusLogger()
    metric_labels: Final[Mapping[str, Sequence[str]]] = {
        metric: list(logger.get_labels_for_metric(metric))
        for metric in get_args(DEFINED_PROMETHEUS_METRICS)
        if hasattr(PrometheusMetricLabels, metric)
    }
    return {"families": captured, "metric_labels": metric_labels}


def test_the_rust_family_table_is_what_the_logger_declares(monkeypatch: pytest.MonkeyPatch) -> None:
    assert json.loads(FAMILIES.read_text()) == _declared(monkeypatch)
