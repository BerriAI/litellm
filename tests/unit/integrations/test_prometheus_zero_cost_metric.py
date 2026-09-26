import datetime
from typing import Final

import pytest
from prometheus_client import REGISTRY
from prometheus_client.samples import Sample

import litellm
from litellm.integrations.prometheus import PrometheusLogger
from litellm.types.utils import StandardLoggingZeroCostDiagnostic

METRIC: Final = "litellm_zero_cost_requests_total"
MISSING_KEY_DIAGNOSTIC: Final[StandardLoggingZeroCostDiagnostic] = {
    "reason": "missing_pricing_key",
    "pricing_model": "dep-1",
    "missing_pricing_keys": ("input_cost_per_token", "output_cost_per_token"),
}


def _clear_prometheus_registry() -> None:
    for collector in list(REGISTRY._collector_to_names.keys()):
        try:
            REGISTRY.unregister(collector)
        except Exception:
            pass


def _samples(metric_name: str) -> list[Sample]:
    return [sample for metric in REGISTRY.collect() for sample in metric.samples if sample.name == metric_name]


def _payload(zero_cost_diagnostic: StandardLoggingZeroCostDiagnostic | None) -> dict[str, object]:
    return {
        "id": "t",
        "call_type": "completion",
        "response_cost": 0.0,
        "status": "success",
        "total_tokens": 30,
        "prompt_tokens": 20,
        "completion_tokens": 10,
        "startTime": 1.0,
        "endTime": 2.0,
        "completionStartTime": 1.5,
        "model": "openai/gpt-5.4-nano",
        "model_id": "dep-1",
        "model_group": "per-second-priced-chat",
        "api_base": "https://api.openai.com",
        "custom_llm_provider": "openai",
        "request_tags": [],
        "end_user": None,
        "cache_hit": False,
        "stream": False,
        "response": {"id": "chatcmpl-1"},
        "model_parameters": {},
        "zero_cost_diagnostic": zero_cost_diagnostic,
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
    }


async def _log_success(
    logger: PrometheusLogger, zero_cost_diagnostic: StandardLoggingZeroCostDiagnostic | None
) -> None:
    now: Final = datetime.datetime.now()
    kwargs: Final = {
        "model": "openai/gpt-5.4-nano",
        "litellm_params": {"metadata": {}},
        "standard_logging_object": _payload(zero_cost_diagnostic),
        "stream": False,
        "start_time": now - datetime.timedelta(seconds=3),
        "api_call_start_time": now - datetime.timedelta(seconds=2),
        "completion_start_time": now - datetime.timedelta(seconds=1),
        "end_time": now,
    }
    await logger.async_log_success_event(kwargs, None, now, now)


async def _log_failure(
    logger: PrometheusLogger, zero_cost_diagnostic: StandardLoggingZeroCostDiagnostic | None
) -> None:
    now: Final = datetime.datetime.now()
    kwargs: Final = {
        "model": "openai/gpt-5.4-nano",
        "litellm_params": {"metadata": {}},
        "standard_logging_object": {**_payload(zero_cost_diagnostic), "status": "failure"},
        "exception": Exception("stream cut off after the usage chunk"),
        "stream": True,
        "start_time": now - datetime.timedelta(seconds=3),
        "end_time": now,
    }
    await logger.async_log_failure_event(kwargs, None, now, now)


@pytest.mark.asyncio
async def test_failure_event_counts_a_zero_cost_request_by_model_and_reason() -> None:
    _clear_prometheus_registry()
    try:
        logger: Final = PrometheusLogger()
        await _log_failure(logger, None)
        assert _samples(METRIC) == []

        await _log_failure(logger, MISSING_KEY_DIAGNOSTIC)

        samples: Final = _samples(METRIC)
        assert len(samples) == 1
        assert samples[0].labels == {
            "requested_model": "per-second-priced-chat",
            "model": "openai/gpt-5.4-nano",
            "model_id": "dep-1",
            "api_provider": "openai",
            "reason": "missing_pricing_key",
        }
        assert samples[0].value == 1.0
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_success_event_counts_a_zero_cost_request_by_model_and_reason() -> None:
    _clear_prometheus_registry()
    try:
        logger: Final = PrometheusLogger()
        await _log_success(logger, MISSING_KEY_DIAGNOSTIC)
        await _log_success(logger, MISSING_KEY_DIAGNOSTIC)

        samples: Final = _samples(METRIC)
        assert len(samples) == 1
        assert samples[0].labels == {
            "requested_model": "per-second-priced-chat",
            "model": "openai/gpt-5.4-nano",
            "model_id": "dep-1",
            "api_provider": "openai",
            "reason": "missing_pricing_key",
        }
        assert samples[0].value == 2.0
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_request_without_a_diagnostic_leaves_the_counter_untouched() -> None:
    _clear_prometheus_registry()
    try:
        await _log_success(PrometheusLogger(), None)

        assert _samples(METRIC) == []
    finally:
        _clear_prometheus_registry()


@pytest.mark.asyncio
async def test_label_filter_that_drops_reason_still_counts_the_request() -> None:
    _clear_prometheus_registry()
    previous_config: Final = litellm.prometheus_metrics_config
    litellm.prometheus_metrics_config = [
        {"group": "zero_cost", "metrics": [METRIC], "include_labels": ["requested_model"]}
    ]
    try:
        await _log_success(PrometheusLogger(), MISSING_KEY_DIAGNOSTIC)

        samples: Final = _samples(METRIC)
        assert len(samples) == 1
        assert samples[0].labels == {"requested_model": "per-second-priced-chat"}
        assert samples[0].value == 1.0
    finally:
        litellm.prometheus_metrics_config = previous_config
        _clear_prometheus_registry()
