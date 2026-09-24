from typing import Final

from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

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


def test_capture_rate_gauge_holds_the_latest_rate_per_provider() -> None:
    _clear_prometheus_registry()
    try:
        logger: Final = PrometheusLogger()
        assert _samples(METRIC) == []

        logger.set_spend_capture_rate(api_provider="openai", capture_rate=0.87)
        logger.set_spend_capture_rate(api_provider="openai", capture_rate=0.91)

        samples: Final = _samples(METRIC)
        assert [(sample.labels, sample.value) for sample in samples] == [({"api_provider": "openai"}, 0.91)]
    finally:
        _clear_prometheus_registry()
