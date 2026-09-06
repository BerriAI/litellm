import datetime
from collections.abc import Mapping
from typing import Final, cast

import pytest
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.integrations.prometheus import (
    PrometheusMetricLabels,
    UserAPIKeyLabelNames,
    UserAPIKeyLabelValues,
    get_input_sequence_length_bucket,
)
from litellm.types.utils import StandardLoggingPayload

LATENCY_METRICS: Final = (
    "litellm_llm_api_latency_metric",
    "litellm_llm_api_time_to_first_token_metric",
    "litellm_request_total_latency_metric",
)
FLAG: Final = "prometheus_emit_input_sequence_length_label"


def _clear_prometheus_registry() -> None:
    for collector in list(REGISTRY._collector_to_names):  # pyright: ignore[reportPrivateUsage]
        REGISTRY.unregister(collector)


@pytest.fixture(autouse=True)
def isolated_registry(monkeypatch: pytest.MonkeyPatch):
    _clear_prometheus_registry()
    monkeypatch.setattr(litellm, FLAG, False)
    yield
    _clear_prometheus_registry()


@pytest.mark.parametrize("metric", LATENCY_METRICS)
def test_input_sequence_length_label_is_opt_in(monkeypatch: pytest.MonkeyPatch, metric: str):
    assert UserAPIKeyLabelNames.INPUT_SEQUENCE_LENGTH.value not in PrometheusMetricLabels.get_labels(metric)

    monkeypatch.setattr(litellm, FLAG, True)
    assert UserAPIKeyLabelNames.INPUT_SEQUENCE_LENGTH.value in PrometheusMetricLabels.get_labels(metric)


def test_input_sequence_length_label_stays_off_non_latency_metrics(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(litellm, FLAG, True)
    assert UserAPIKeyLabelNames.INPUT_SEQUENCE_LENGTH.value not in PrometheusMetricLabels.get_labels(
        "litellm_proxy_total_requests_metric"
    )


@pytest.mark.parametrize(
    "prompt_tokens, expected",
    [
        (None, "unknown"),
        (0, "0-1k"),
        (999, "0-1k"),
        (1_000, "1k-4k"),
        (3_999, "1k-4k"),
        (4_000, "4k-16k"),
        (15_999, "4k-16k"),
        (16_000, "16k-64k"),
        (63_999, "16k-64k"),
        (64_000, "64k+"),
        (10_000_000, "64k+"),
        (-1, "unknown"),
    ],
)
def test_input_sequence_length_bucket_boundaries(prompt_tokens: int | None, expected: str):
    assert get_input_sequence_length_bucket(prompt_tokens) == expected


def test_user_api_key_label_values_carries_input_sequence_length():
    values: Final = UserAPIKeyLabelValues(input_sequence_length="4k-16k")

    assert values.input_sequence_length == "4k-16k"
    assert values.model_dump()["input_sequence_length"] == "4k-16k"


def _latency_bucket_samples() -> tuple[Sample, ...]:
    return tuple(
        sample
        for metric in REGISTRY.collect()
        for sample in metric.samples
        if sample.name.endswith("_bucket") and any(name in sample.name for name in LATENCY_METRICS)
    )


def _standard_logging_payload(now: datetime.datetime, prompt_tokens: int) -> StandardLoggingPayload:
    return cast(
        StandardLoggingPayload,
        {
            "id": "t",
            "call_type": "completion",
            "response_cost": 0.001,
            "status": "success",
            "total_tokens": prompt_tokens + 20,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": 20,
            "startTime": now - datetime.timedelta(seconds=3),
            "endTime": now,
            "completionStartTime": now - datetime.timedelta(seconds=1),
            "model": "gpt-4o-mini",
            "model_id": "model-123",
            "model_group": "gpt-4o-mini",
            "api_base": "https://api.openai.com",
            "custom_llm_provider": "openai",
            "request_tags": [],
            "stream": True,
            "metadata": {
                "user_api_key_hash": "h",
                "user_api_key_alias": "a",
                "user_api_key_team_id": "t",
                "user_api_key_team_alias": "ta",
                "user_api_key_user_id": "u",
                "user_api_key_user_email": "e@x.com",
                "user_api_key_org_id": None,
                "user_api_key_org_alias": None,
                "requester_metadata": None,
                "user_api_key_end_user_id": None,
                "usage_object": None,
            },
            "hidden_params": {"litellm_overhead_time_ms": None, "additional_headers": None},
        },
    )


def _success_kwargs(now: datetime.datetime, prompt_tokens: int) -> Mapping[str, object]:
    return {
        "model": "gpt-4o-mini",
        "litellm_params": {"metadata": {}},
        "standard_logging_object": _standard_logging_payload(now, prompt_tokens),
        "stream": True,
        "start_time": now - datetime.timedelta(seconds=3),
        "api_call_start_time": now - datetime.timedelta(seconds=2),
        "completion_start_time": now - datetime.timedelta(seconds=1),
        "end_time": now,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("flag_at_request_time", (True, False))
async def test_logger_emits_bucket_from_its_startup_label_set(
    monkeypatch: pytest.MonkeyPatch, flag_at_request_time: bool
):
    now: Final = datetime.datetime.now()
    monkeypatch.setattr(litellm, FLAG, True)
    logger: Final = PrometheusLogger()
    monkeypatch.setattr(litellm, FLAG, flag_at_request_time)

    await logger.async_log_success_event(dict(_success_kwargs(now, prompt_tokens=4_000)), None, now, now)

    samples: Final = _latency_bucket_samples()
    assert samples
    assert all(sample.labels["input_sequence_length"] == "4k-16k" for sample in samples)


@pytest.mark.asyncio
async def test_logger_built_with_flag_off_emits_no_bucket_label(monkeypatch: pytest.MonkeyPatch):
    now: Final = datetime.datetime.now()
    logger: Final = PrometheusLogger()
    monkeypatch.setattr(litellm, FLAG, True)

    await logger.async_log_success_event(dict(_success_kwargs(now, prompt_tokens=4_000)), None, now, now)

    samples: Final = _latency_bucket_samples()
    assert samples
    assert all("input_sequence_length" not in sample.labels for sample in samples)
