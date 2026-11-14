from datetime import datetime

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from tests.unit.integrations.prometheus_native_support import (
    enable_native_prometheus,
    native_success,
    series,
    standard_payload,
)


@pytest.fixture(autouse=True)
def native_prometheus(monkeypatch):
    enable_native_prometheus(monkeypatch)


@pytest.fixture
def logger():
    for collector in list(REGISTRY._collector_to_names.keys()):
        REGISTRY.unregister(collector)
    return PrometheusLogger()


CACHED = "litellm_input_cached_tokens_metric_total"
CACHE_CREATION = "litellm_input_cache_creation_tokens_metric_total"
INPUT_AUDIO = "litellm_input_audio_tokens_metric_total"
REASONING = "litellm_output_reasoning_tokens_metric_total"
OUTPUT_AUDIO = "litellm_output_audio_tokens_metric_total"
TOKEN_DETAIL_METRICS = [CACHED, CACHE_CREATION, INPUT_AUDIO, REASONING, OUTPUT_AUDIO]


async def _log(logger: PrometheusLogger, metadata: dict | None) -> None:
    now = datetime.now()
    payload = standard_payload(
        metadata={
            "user_api_key_hash": "test-key-hash",
            "user_api_key_alias": "test-key-alias",
            "user_api_key_team_id": "test-team",
            "user_api_key_team_alias": "test-team-alias",
            "user_api_key_user_id": "test-user",
        }
        | (metadata or {}),
        model="gpt-4o",
    )
    if metadata is None:
        payload["metadata"] = None
    kwargs = {
        "model": "gpt-4o",
        "litellm_params": {"metadata": {"user_api_key_end_user_id": "test-end-user"}},
        "start_time": now,
        "end_time": now,
        "standard_logging_object": payload,
    }
    await native_success(logger, kwargs, None)


def _values(name: str) -> list[float]:
    return [value for _, value in series(name)]


class TestIncrementTokenDetailMetrics:
    @pytest.mark.asyncio
    async def test_increments_all_present_token_types(self, logger):
        await _log(
            logger,
            {
                "usage_object": {
                    "prompt_tokens": 100,
                    "completion_tokens": 80,
                    "total_tokens": 180,
                    "prompt_tokens_details": {
                        "cached_tokens": 40,
                        "cache_creation_tokens": 25,
                        "audio_tokens": 15,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 60,
                        "audio_tokens": 10,
                    },
                }
            },
        )

        assert _values(CACHED) == [40.0]
        assert _values(CACHE_CREATION) == [25.0]
        assert _values(INPUT_AUDIO) == [15.0]
        assert _values(REASONING) == [60.0]
        assert _values(OUTPUT_AUDIO) == [10.0]

    @pytest.mark.asyncio
    async def test_cache_creation_falls_back_to_cache_write_tokens(self, logger):
        await _log(
            logger,
            {
                "usage_object": {
                    "prompt_tokens": 12000,
                    "completion_tokens": 100,
                    "total_tokens": 12100,
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                        "cache_write_tokens": 800,
                    },
                }
            },
        )

        assert _values(CACHE_CREATION) == [800.0]

    @pytest.mark.asyncio
    async def test_cache_write_tokens_takes_precedence_over_cache_creation_tokens(self, logger):
        await _log(
            logger,
            {
                "usage_object": {
                    "prompt_tokens_details": {
                        "cache_creation_tokens": 25,
                        "cache_write_tokens": 800,
                    },
                }
            },
        )

        assert _values(CACHE_CREATION) == [800.0]

    @pytest.mark.asyncio
    async def test_skips_metrics_when_value_is_zero(self, logger):
        await _log(
            logger,
            {
                "usage_object": {
                    "prompt_tokens_details": {
                        "cached_tokens": 0,
                        "cache_creation_tokens": 0,
                        "audio_tokens": 0,
                    },
                    "completion_tokens_details": {
                        "reasoning_tokens": 0,
                        "audio_tokens": 0,
                    },
                }
            },
        )

        for name in TOKEN_DETAIL_METRICS:
            assert series(name) == []

    @pytest.mark.asyncio
    async def test_skips_metrics_when_value_is_none(self, logger):
        await _log(
            logger,
            {
                "usage_object": {
                    "prompt_tokens_details": {
                        "cached_tokens": None,
                        "audio_tokens": 12,
                    },
                    "completion_tokens_details": {},
                }
            },
        )

        assert series(CACHED) == []
        assert series(CACHE_CREATION) == []
        assert _values(INPUT_AUDIO) == [12.0]
        assert series(REASONING) == []
        assert series(OUTPUT_AUDIO) == []

    @pytest.mark.asyncio
    async def test_no_usage_object_is_a_noop(self, logger):
        await _log(logger, {})

        for name in TOKEN_DETAIL_METRICS:
            assert series(name) == []

    @pytest.mark.asyncio
    async def test_missing_metadata_is_a_noop(self, logger):
        with pytest.raises(TypeError):
            await _log(logger, None)

        for name in TOKEN_DETAIL_METRICS:
            assert series(name) == []

    @pytest.mark.asyncio
    async def test_negative_values_are_ignored(self, logger):
        await _log(
            logger,
            {
                "usage_object": {
                    "prompt_tokens_details": {"cached_tokens": -5},
                    "completion_tokens_details": {"reasoning_tokens": -10},
                }
            },
        )

        assert series(CACHED) == []
        assert series(REASONING) == []
