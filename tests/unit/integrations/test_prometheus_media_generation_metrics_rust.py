from datetime import datetime

import pytest
from prometheus_client import REGISTRY

from litellm.integrations.prometheus import PrometheusLogger
from tests.unit.integrations.prometheus_native_support import (
    enable_native_prometheus,
    labelsets,
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


VIDEO = "litellm_video_duration_seconds_metric_total"
IMAGES = "litellm_images_generated_metric_total"
MEDIA_GENERATION_METRICS = [VIDEO, IMAGES]


def _payload(metadata: dict | None) -> dict:
    payload = standard_payload(
        metadata={
            "user_api_key_hash": "test-key-hash",
            "user_api_key_alias": "test-key-alias",
            "user_api_key_team_id": "test-team",
            "user_api_key_team_alias": "test-team-alias",
            "user_api_key_user_id": "test-user",
        }
        | (metadata or {}),
        model="sora-2",
    )
    if metadata is None:
        payload["metadata"] = None
    return payload


async def _log(logger: PrometheusLogger, metadata: dict | None) -> None:
    now = datetime.now()
    kwargs = {
        "model": "sora-2",
        "litellm_params": {"metadata": {"user_api_key_end_user_id": "test-end-user"}},
        "start_time": now,
        "end_time": now,
        "standard_logging_object": _payload(metadata),
    }
    await native_success(logger, kwargs, None)


class TestMediaGenerationMetricsRegistration:
    @pytest.mark.asyncio
    async def test_runtime_label_set_matches_output_tokens_metric(self, logger):
        await _log(logger, {"usage_object": {"duration_seconds": 8.0, "output_image_count": 2}})

        expected = [set(labels) for labels in labelsets("litellm_output_tokens_metric_total")]
        assert len(expected) == 1
        for name in MEDIA_GENERATION_METRICS:
            assert [set(labels) for labels in labelsets(name)] == expected


class TestIncrementMediaGenerationMetrics:
    @pytest.mark.asyncio
    async def test_video_duration_incremented(self, logger):
        await _log(logger, {"usage_object": {"duration_seconds": 8.0}})

        assert [value for _, value in series(VIDEO)] == [8.0]
        assert series(IMAGES) == []

    @pytest.mark.asyncio
    async def test_image_count_incremented(self, logger):
        await _log(
            logger,
            {
                "usage_object": {
                    "prompt_tokens": 18,
                    "completion_tokens": 391,
                    "total_tokens": 409,
                    "output_image_count": 2,
                }
            },
        )

        assert [value for _, value in series(IMAGES)] == [2.0]
        assert series(VIDEO) == []

    @pytest.mark.asyncio
    async def test_token_only_usage_is_a_noop(self, logger):
        await _log(
            logger,
            {
                "usage_object": {
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            },
        )

        for name in MEDIA_GENERATION_METRICS:
            assert series(name) == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("bad_value", [0, 0.0, None, -4.0, "4", True])
    async def test_non_positive_or_non_numeric_values_are_ignored(self, logger, bad_value):
        await _log(
            logger,
            {
                "usage_object": {
                    "duration_seconds": bad_value,
                    "output_image_count": bad_value,
                }
            },
        )

        for name in MEDIA_GENERATION_METRICS:
            assert series(name) == []

    @pytest.mark.asyncio
    async def test_missing_usage_object_is_a_noop(self, logger):
        for metadata in ({}, {"usage_object": "redacted"}):
            await _log(logger, metadata)
        with pytest.raises(TypeError):
            await _log(logger, None)

        for name in MEDIA_GENERATION_METRICS:
            assert series(name) == []
