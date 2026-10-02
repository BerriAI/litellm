import asyncio
import datetime
from collections.abc import Mapping
from copy import deepcopy
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
    for collector in tuple(REGISTRY._collector_to_names):  # pyright: ignore[reportPrivateUsage]  # test registry reset
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


def _assert_latency_metrics(expected: str | None, stream: bool = True, queue_time: float = 0) -> None:
    samples: Final = tuple(sample for metric in REGISTRY.collect() for sample in metric.samples)
    for metric, duration in zip(LATENCY_METRICS, (2, 1, 3 + queue_time)):
        counts: Final = tuple(sample for sample in samples if sample.name == f"{metric}_count")
        sums: Final = tuple(sample for sample in samples if sample.name == f"{metric}_sum")
        buckets: Final = tuple(sample for sample in samples if sample.name == f"{metric}_bucket")
        if not stream and metric == "litellm_llm_api_time_to_first_token_metric":
            assert not counts and not sums and not buckets
            continue
        assert len(counts) == len(sums) == 1
        assert counts[0].value == 1
        assert sums[0].value == pytest.approx(duration)
        assert buckets and any(sample.labels["le"] == "+Inf" for sample in buckets)
        assert all(sample.value == int(float(sample.labels["le"]) >= duration) for sample in buckets)
        assert all(sample.labels.get("input_sequence_length") == expected for sample in (*counts, *sums, *buckets))


def _non_target_samples() -> tuple[Sample, ...]:
    return tuple(
        sample
        for metric in REGISTRY.collect()
        if metric.name not in LATENCY_METRICS
        for sample in metric.samples
        if "input_sequence_length" in sample.labels and not sample.name.endswith("_created")
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


def _success_kwargs(
    now: datetime.datetime, prompt_tokens: int, requester_metadata: Mapping[str, object] | None = None
) -> Mapping[str, object]:
    payload: Final = _standard_logging_payload(now, prompt_tokens)
    return {
        "model": "gpt-4o-mini",
        "litellm_params": {"metadata": {}},
        "standard_logging_object": {
            **payload,
            "metadata": {**payload["metadata"], "requester_metadata": requester_metadata},
        },
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

    _assert_latency_metrics("4k-16k")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("response", "combined_usage", "expected"),
    (
        ({"id": "moderation", "results": []}, None, "unknown"),
        ({"usage": None}, None, "unknown"),
        ({"usage": {}}, None, "unknown"),
        ({"usage": {"completion_tokens": 3}}, None, "unknown"),
        ({"usage": {"total_tokens": 5}}, None, "unknown"),
        ({"usage": {"prompt_tokens": 0}}, None, "0-1k"),
        ({"usage": {"prompt_tokens": 4_000}}, None, "4k-16k"),
        ({"usage": {"input_tokens": 0, "output_tokens": 3, "total_tokens": 3}}, None, "0-1k"),
        ({"usage": {"input_tokens": 4_000, "output_tokens": 3, "total_tokens": 4_003}}, None, "4k-16k"),
        (litellm.ModelResponse(usage=litellm.Usage(prompt_tokens=0)), None, "0-1k"),
        (None, litellm.Usage(prompt_tokens=0), "0-1k"),
    ),
)
@pytest.mark.parametrize("include_usage_metadata", (True, False))
async def test_logger_distinguishes_missing_usage_from_reported_zero(
    monkeypatch: pytest.MonkeyPatch,
    response: object,
    combined_usage: object,
    expected: str,
    include_usage_metadata: bool,
):
    from litellm.litellm_core_utils.litellm_logging import StandardLoggingPayloadSetup

    now: Final = datetime.datetime.now()
    monkeypatch.setattr(litellm, FLAG, True)
    logger: Final = PrometheusLogger()
    usage: Final = StandardLoggingPayloadSetup.get_usage_as_dict(
        response_obj=response if isinstance(response, dict) else None
    )

    payload: Final = _standard_logging_payload(now, usage.get("prompt_tokens", 0))
    await logger.async_log_success_event(
        {
            **_success_kwargs(now, prompt_tokens=usage.get("prompt_tokens", 0)),
            "combined_usage_object": combined_usage,
            "standard_logging_object": {
                **payload,
                "metadata": {**payload["metadata"], "usage_object": usage if include_usage_metadata else None},
            },
        },
        response,
        now,
        now,
    )

    _assert_latency_metrics(expected)


@pytest.mark.asyncio
@pytest.mark.parametrize("total_tokens", (None, 0, 5_000))
@pytest.mark.parametrize("prompt_tokens", (0, 4_000))
async def test_upstream_total_only_usage_has_unknown_input_length(
    monkeypatch: pytest.MonkeyPatch, total_tokens: int | None, prompt_tokens: int
):
    import httpx

    from litellm.litellm_core_utils.litellm_logging import Logging, StandardLoggingPayloadSetup
    from litellm.proxy.pass_through_endpoints.upstream_usage_headers import apply_upstream_reported_usage

    now: Final = datetime.datetime.now()
    monkeypatch.setattr(litellm, FLAG, True)
    logger: Final = PrometheusLogger()
    logging_obj: Final = Logging(
        model="gpt-4o-mini",
        messages=[],
        stream=True,
        call_type="pass_through_endpoint",
        start_time=now,
        litellm_call_id="test-call-id",
        function_id="1",
    )
    headers: Final = httpx.Headers(
        {
            "x-litellm-response-cost": "0.001",
            **({"x-litellm-total-tokens": str(total_tokens)} if total_tokens is not None else {}),
        }
    )
    reported: Final = apply_upstream_reported_usage(logging_obj=logging_obj, headers=headers)
    assert reported is not None
    combined_usage: Final = logging_obj.model_call_details.get("combined_usage_object")
    response: Final = {"usage": {"prompt_tokens": prompt_tokens}}
    usage: Final = StandardLoggingPayloadSetup.get_usage_as_dict(response, combined_usage)
    payload: Final = _standard_logging_payload(now, usage.get("prompt_tokens", 0))

    await logger.async_log_success_event(
        {
            **logging_obj.model_call_details,
            **_success_kwargs(now, usage.get("prompt_tokens", 0)),
            "standard_logging_object": {**payload, "metadata": {**payload["metadata"], "usage_object": usage}},
        },
        response,
        now,
        now,
    )

    _assert_latency_metrics("unknown" if total_tokens is not None else get_input_sequence_length_bucket(prompt_tokens))


@pytest.mark.asyncio
async def test_logger_built_with_flag_off_emits_no_bucket_label(monkeypatch: pytest.MonkeyPatch):
    now: Final = datetime.datetime.now()
    logger: Final = PrometheusLogger()
    monkeypatch.setattr(litellm, FLAG, True)

    await logger.async_log_success_event(dict(_success_kwargs(now, prompt_tokens=4_000)), None, now, now)

    _assert_latency_metrics(None)


@pytest.mark.asyncio
@pytest.mark.parametrize("flag_at_startup", (True, False))
@pytest.mark.parametrize("stream", (True, False))
@pytest.mark.parametrize(
    "metadata",
    (
        None,
        {},
        {"input_sequence_length": None},
        {"input_sequence_length": False},
        {"input_sequence_length": True},
        {"input_sequence_length": 0},
        {"input_sequence_length": []},
        {"input_sequence_length": {}},
        {"input_sequence_length": ""},
        {"input_sequence_length": "from-metadata"},
    ),
)
async def test_custom_input_length_label_is_scoped_to_target_histograms(
    monkeypatch: pytest.MonkeyPatch, flag_at_startup: bool, stream: bool, metadata: Mapping[str, object] | None
):
    now: Final = datetime.datetime.now()
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["input_sequence_length"])
    kwargs: Final = {
        **_success_kwargs(now, prompt_tokens=4_000, requester_metadata=metadata),
        "stream": stream,
        "litellm_params": {"metadata": {"queue_time_seconds": 0.25}},
    }
    original_kwargs: Final = deepcopy(kwargs)
    baseline_logger: Final = PrometheusLogger()
    await baseline_logger.async_log_success_event(kwargs, None, now, now)
    baseline_samples: Final = _non_target_samples()
    _clear_prometheus_registry()
    monkeypatch.setattr(litellm, FLAG, flag_at_startup)
    logger: Final = PrometheusLogger()
    monkeypatch.setattr(litellm, FLAG, not flag_at_startup)

    await logger.async_log_success_event(kwargs, None, now, now)

    assert kwargs == original_kwargs
    custom_value: Final = (metadata or {}).get("input_sequence_length")
    expected: Final = custom_value if isinstance(custom_value, str) else ("4k-16k" if flag_at_startup else "None")
    _assert_latency_metrics(expected, stream=stream, queue_time=0.25)
    assert all(logger.get_labels_for_metric(metric).count("input_sequence_length") == 1 for metric in LATENCY_METRICS)
    non_target_samples: Final = _non_target_samples()
    assert {
        "litellm_requests_metric_total",
        "litellm_spend_metric_total",
        "litellm_total_tokens_metric_total",
        "litellm_request_queue_time_seconds_count",
        "litellm_deployment_success_responses_total",
    }.issubset({sample.name for sample in non_target_samples})
    assert non_target_samples == baseline_samples
    queue_sum: Final = tuple(
        sample for sample in non_target_samples if sample.name == "litellm_request_queue_time_seconds_sum"
    )
    assert len(queue_sum) == 1 and queue_sum[0].value == 0.25


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (True, False))
async def test_concurrent_requests_keep_independent_buckets(monkeypatch: pytest.MonkeyPatch, enabled: bool):
    now: Final = datetime.datetime.now()
    monkeypatch.setattr(litellm, FLAG, enabled)
    logger: Final = PrometheusLogger()
    cases: Final = (
        (None, "unknown"),
        (0, "0-1k"),
        (1_000, "1k-4k"),
        (4_000, "4k-16k"),
        (16_000, "16k-64k"),
        (64_000, "64k+"),
    )
    calls: Final = tuple(
        (
            {**_success_kwargs(now, prompt_tokens=tokens or 0), "stream": stream},
            {"usage": {"prompt_tokens": tokens}} if tokens is not None else None,
        )
        for tokens, _ in cases
        for stream in (True, False)
        for _ in range(2)
    )
    original_calls: Final = deepcopy(calls)

    await asyncio.gather(*(logger.async_log_success_event(kwargs, response, now, now) for kwargs, response in calls))

    assert calls == original_calls
    samples: Final = tuple(sample for metric in REGISTRY.collect() for sample in metric.samples)
    for metric, duration in zip(LATENCY_METRICS, (2, 1, 3)):
        expected_count: Final = 2 if metric == "litellm_llm_api_time_to_first_token_metric" else 4
        counts: Final = tuple(sample for sample in samples if sample.name == f"{metric}_count")
        sums: Final = tuple(sample for sample in samples if sample.name == f"{metric}_sum")
        buckets: Final = tuple(sample for sample in samples if sample.name == f"{metric}_bucket")
        expected: Final = (
            {bucket: expected_count for _, bucket in cases} if enabled else {None: expected_count * len(cases)}
        )
        assert len(counts) == len(sums) == len(expected)
        assert {sample.labels.get("input_sequence_length"): sample.value for sample in counts} == expected
        assert {sample.labels.get("input_sequence_length"): sample.value for sample in sums} == {
            bucket: count * duration for bucket, count in expected.items()
        }
        assert sum(sample.value for sample in buckets if sample.labels["le"] == "+Inf") == expected_count * len(cases)
        assert all(
            sample.value
            == expected[sample.labels.get("input_sequence_length")] * int(float(sample.labels["le"]) >= duration)
            for sample in buckets
        )


@pytest.mark.asyncio
@pytest.mark.parametrize("enabled", (True, False))
async def test_failed_request_does_not_observe_latency(monkeypatch: pytest.MonkeyPatch, enabled: bool):
    now: Final = datetime.datetime.now()
    monkeypatch.setattr(litellm, FLAG, enabled)
    monkeypatch.setattr(litellm, "custom_prometheus_metadata_labels", ["input_sequence_length"])
    logger: Final = PrometheusLogger()
    kwargs: Final = {
        **_success_kwargs(now, prompt_tokens=4_000),
        "standard_logging_object": {**_standard_logging_payload(now, 4_000), "status": "failure"},
        "exception": RuntimeError("upstream request failed"),
    }

    await logger.async_log_failure_event(kwargs, None, now, now)

    samples: Final = tuple(sample for metric in REGISTRY.collect() for sample in metric.samples)
    assert not any(sample.name.startswith(LATENCY_METRICS) for sample in samples)
    for metric in ("litellm_llm_api_failed_requests_metric_total", "litellm_deployment_failure_responses_total"):
        counts: Final = tuple(sample for sample in samples if sample.name == metric)
        assert len(counts) == 1
        assert counts[0].value == 1
        assert counts[0].labels["input_sequence_length"] == "None"
