from datetime import datetime
from typing import get_args

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import DEFINED_PROMETHEUS_METRICS, PrometheusMetricLabels
from tests.unit.integrations.prometheus_native_support import (
    enable_native_prometheus,
    native_success,
    series,
    standard_payload,
)

HITS = "litellm_cache_hits_metric_total"
MISSES = "litellm_cache_misses_metric_total"
CACHED = "litellm_cached_tokens_metric_total"
PROVIDER_READ = "litellm_provider_cache_read_input_tokens_metric_total"
PROVIDER_CREATION = "litellm_provider_cache_creation_input_tokens_metric_total"


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


@pytest.fixture
def logger():
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)
    return PrometheusLogger()


async def _log(logger: PrometheusLogger, cache_hit: object, usage_object: dict, total_tokens: int = 100) -> None:
    now = datetime.now()
    payload = standard_payload(
        metadata={
            "user_api_key_hash": "test-key-hash",
            "user_api_key_alias": "test-key-alias",
            "user_api_key_team_id": "test-team",
            "user_api_key_team_alias": "test-team-alias",
            "user_api_key_user_id": "test-user",
            "usage_object": usage_object,
        },
        model="gpt-3.5-turbo",
        model_group="openai",
        cache_hit=cache_hit,
        total_tokens=total_tokens,
        prompt_tokens=total_tokens // 2,
        completion_tokens=total_tokens - total_tokens // 2,
    )
    kwargs = {
        "model": "gpt-3.5-turbo",
        "litellm_params": {"metadata": {"user_api_key_end_user_id": "test-end-user"}},
        "start_time": now,
        "end_time": now,
        "standard_logging_object": payload,
    }
    await native_success(logger, kwargs, None)


def _values(name: str) -> list[float]:
    return [value for _, value in series(name)]


class TestPrometheusCacheMetrics:
    def test_cache_metrics_defined_in_types(self):
        defined_metrics = get_args(DEFINED_PROMETHEUS_METRICS)

        assert "litellm_cache_hits_metric" in defined_metrics
        assert "litellm_cache_misses_metric" in defined_metrics
        assert "litellm_cached_tokens_metric" in defined_metrics
        assert "litellm_provider_cache_read_input_tokens_metric" in defined_metrics
        assert "litellm_provider_cache_creation_input_tokens_metric" in defined_metrics

    def test_cache_metric_labels_defined(self):
        assert hasattr(PrometheusMetricLabels, "litellm_cache_hits_metric")
        assert hasattr(PrometheusMetricLabels, "litellm_cache_misses_metric")
        assert hasattr(PrometheusMetricLabels, "litellm_cached_tokens_metric")
        assert hasattr(PrometheusMetricLabels, "litellm_provider_cache_read_input_tokens_metric")
        assert hasattr(PrometheusMetricLabels, "litellm_provider_cache_creation_input_tokens_metric")

        expected_labels = ["model", "hashed_api_key", "api_key_alias", "team", "team_alias", "end_user", "user"]
        for label in expected_labels:
            assert label in PrometheusMetricLabels.litellm_cache_hits_metric
            assert label in PrometheusMetricLabels.litellm_cache_misses_metric
            assert label in PrometheusMetricLabels.litellm_cached_tokens_metric
            assert label in PrometheusMetricLabels.litellm_provider_cache_read_input_tokens_metric
            assert label in PrometheusMetricLabels.litellm_provider_cache_creation_input_tokens_metric

    @pytest.mark.asyncio
    async def test_increment_cache_metrics_on_cache_hit(self, logger):
        await _log(logger, True, {"cache_read_input_tokens": 25, "cache_creation_input_tokens": 10})

        assert _values(HITS) == [1.0]
        assert _values(CACHED) == [100.0]
        assert _values(MISSES) == []
        assert _values(PROVIDER_READ) == [25.0]
        assert _values(PROVIDER_CREATION) == [10.0]

    @pytest.mark.asyncio
    async def test_increment_cache_metrics_on_cache_miss(self, logger):
        await _log(logger, False, {"prompt_tokens_details": {"cached_tokens": 20}})

        assert _values(MISSES) == [1.0]
        assert _values(HITS) == []
        assert _values(CACHED) == []
        assert _values(PROVIDER_READ) == [20.0]
        assert _values(PROVIDER_CREATION) == []

    @pytest.mark.asyncio
    async def test_provider_cache_read_does_not_fallback_on_explicit_zero(self, logger):
        await _log(logger, False, {"cache_read_input_tokens": 0, "prompt_tokens_details": {"cached_tokens": 20}})

        assert _values(PROVIDER_READ) == []

    @pytest.mark.asyncio
    async def test_provider_cache_creation_fallback_to_cache_write_tokens(self, logger):
        await _log(
            logger,
            False,
            {"prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 800}},
            total_tokens=12100,
        )

        assert _values(PROVIDER_CREATION) == [800.0]

    @pytest.mark.asyncio
    async def test_provider_cache_creation_fallback_to_cache_creation_tokens(self, logger):
        await _log(logger, False, {"prompt_tokens_details": {"cache_creation_tokens": 42}})

        assert _values(PROVIDER_CREATION) == [42.0]

    @pytest.mark.asyncio
    async def test_provider_cache_creation_does_not_fallback_on_explicit_zero(self, logger):
        await _log(
            logger, False, {"cache_creation_input_tokens": 0, "prompt_tokens_details": {"cache_write_tokens": 800}}
        )

        assert _values(PROVIDER_CREATION) == []

    @pytest.mark.asyncio
    async def test_increment_cache_metrics_when_cache_hit_is_none(self, logger):
        await _log(logger, None, {"cache_read_input_tokens": 25})

        assert _values(HITS) == []
        assert _values(MISSES) == []
        assert _values(CACHED) == []
        assert _values(PROVIDER_READ) == [25.0]
        assert _values(PROVIDER_CREATION) == []
