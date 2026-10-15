import math
from typing import Final

import pytest
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

import litellm
from litellm.integrations.prometheus import PrometheusLogger

METRIC: Final = "litellm_spend_capture_rate"


def _clear_prometheus_registry() -> None:
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _samples(metric_name: str) -> list[Sample]:
    return [sample for metric in REGISTRY.collect() for sample in metric.samples if sample.name == metric_name]


def test_capture_rate_gauge_holds_the_latest_rate_per_provider_and_nan_when_there_is_none() -> None:
    _clear_prometheus_registry()
    try:
        logger: Final = PrometheusLogger()
        assert _samples(METRIC) == []

        logger.set_spend_capture_rate(api_provider="openai", capture_rate=0.87)
        logger.set_spend_capture_rate(api_provider="openai", capture_rate=0.91)

        samples: Final = _samples(METRIC)
        assert [(sample.labels, sample.value) for sample in samples] == [({"api_provider": "openai"}, 0.91)]

        logger.set_spend_capture_rate(api_provider="openai", capture_rate=None)

        (unavailable,) = _samples(METRIC)
        assert unavailable.labels == {"api_provider": "openai"} and math.isnan(unavailable.value)
    finally:
        _clear_prometheus_registry()


def test_capture_rate_gauge_still_records_when_api_provider_is_an_excluded_label(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(litellm, "prometheus_exclude_labels", ["api_provider"])
    _clear_prometheus_registry()
    try:
        logger: Final = PrometheusLogger()

        logger.set_spend_capture_rate(api_provider="openai", capture_rate=0.42)

        assert [(sample.labels, sample.value) for sample in _samples(METRIC)] == [({}, 0.42)]

        logger.set_spend_capture_rate(api_provider="openai", capture_rate=None)

        (unavailable,) = _samples(METRIC)
        assert unavailable.labels == {} and math.isnan(unavailable.value)
    finally:
        _clear_prometheus_registry()
