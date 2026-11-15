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


@pytest.fixture(autouse=True)
def cleanup_prometheus_registry():
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)
    yield
    collectors = list(REGISTRY._collector_to_names.keys())
    for collector in collectors:
        REGISTRY.unregister(collector)


async def _success(
    prometheus_logger: PrometheusLogger, metadata: dict, start_time: datetime, end_time: datetime
) -> None:
    kwargs = {
        "litellm_params": {"metadata": metadata},
        "model": "gpt-3.5-turbo",
        "start_time": start_time,
        "end_time": end_time,
        "standard_logging_object": standard_payload(
            metadata={"user_api_key_hash": "test-key", "user_api_key_alias": "test-alias"},
            model="gpt-3.5-turbo",
            model_group="gpt-3.5-turbo",
            model_id="gpt-3.5-turbo",
            api_base="https://api.openai.com",
            custom_llm_provider="openai",
        ),
    }
    await native_success(prometheus_logger, kwargs, None)


def _observed(prometheus_logger: PrometheusLogger, attribute: str) -> list[float]:
    return [value for _, value in series(getattr(prometheus_logger, attribute).name)]


class TestPrometheusQueueTimeMetric:
    @pytest.mark.asyncio
    async def test_queue_time_metric_recorded_in_set_latency_metrics(self):
        prometheus_logger = PrometheusLogger()
        queue_time_seconds = 0.5

        await _success(prometheus_logger, {"queue_time_seconds": queue_time_seconds}, datetime.now(), datetime.now())

        observed = _observed(prometheus_logger, "litellm_request_queue_time_metric")
        assert observed == [queue_time_seconds]
        assert observed[0] >= 0

    @pytest.mark.asyncio
    async def test_queue_time_metric_not_recorded_when_missing(self):
        prometheus_logger = PrometheusLogger()

        await _success(prometheus_logger, {}, datetime.now(), datetime.now())

        assert 0.5 not in _observed(prometheus_logger, "litellm_request_queue_time_metric")

    @pytest.mark.asyncio
    async def test_queue_time_metric_not_recorded_when_negative(self):
        prometheus_logger = PrometheusLogger()

        await _success(prometheus_logger, {"queue_time_seconds": -0.1}, datetime.now(), datetime.now())

        assert -0.1 not in _observed(prometheus_logger, "litellm_request_queue_time_metric")


class TestPrometheusTotalLatencyMetric:
    @pytest.mark.asyncio
    async def test_total_latency_includes_queue_time_when_present(self):
        prometheus_logger = PrometheusLogger()

        await _success(
            prometheus_logger,
            {"queue_time_seconds": 0.5},
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 1, 1, 0, 0, 2),
        )

        [observed_value] = _observed(prometheus_logger, "litellm_request_total_latency_metric")
        assert observed_value == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_total_latency_falls_back_to_start_end_delta_without_queue_time(self):
        prometheus_logger = PrometheusLogger()

        await _success(prometheus_logger, {}, datetime(2024, 1, 1, 0, 0, 0), datetime(2024, 1, 1, 0, 0, 2))

        [observed_value] = _observed(prometheus_logger, "litellm_request_total_latency_metric")
        assert observed_value == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_total_latency_ignores_negative_queue_time(self):
        prometheus_logger = PrometheusLogger()

        await _success(
            prometheus_logger,
            {"queue_time_seconds": -0.1},
            datetime(2024, 1, 1, 0, 0, 0),
            datetime(2024, 1, 1, 0, 0, 2),
        )

        [observed_value] = _observed(prometheus_logger, "litellm_request_total_latency_metric")
        assert observed_value == pytest.approx(2.0)


def _record(prometheus_logger: PrometheusLogger, **values: object) -> None:
    prometheus_logger._record_guardrail_metrics(**values)


class TestPrometheusGuardrailMetrics:
    def test_record_guardrail_metrics_success(self):
        prometheus_logger = PrometheusLogger()
        _record(
            prometheus_logger,
            guardrail_name="test_guardrail",
            latency_seconds=0.15,
            status="success",
            error_type=None,
            hook_type="pre_call",
        )

        assert series("litellm_guardrail_latency_seconds") == [
            (
                {
                    "guardrail_name": "test_guardrail",
                    "status": "success",
                    "error_type": "none",
                    "hook_type": "pre_call",
                },
                0.15,
            )
        ]
        assert series("litellm_guardrail_requests_total") == [
            ({"guardrail_name": "test_guardrail", "status": "success", "hook_type": "pre_call"}, 1.0)
        ]
        assert series("litellm_guardrail_errors_total") == []

    def test_record_guardrail_metrics_error(self):
        prometheus_logger = PrometheusLogger()
        _record(
            prometheus_logger,
            guardrail_name="test_guardrail",
            latency_seconds=0.2,
            status="error",
            error_type="ValueError",
            hook_type="pre_call",
        )

        assert series("litellm_guardrail_latency_seconds") == [
            (
                {
                    "guardrail_name": "test_guardrail",
                    "status": "error",
                    "error_type": "ValueError",
                    "hook_type": "pre_call",
                },
                0.2,
            )
        ]
        assert series("litellm_guardrail_requests_total") == [
            ({"guardrail_name": "test_guardrail", "status": "error", "hook_type": "pre_call"}, 1.0)
        ]
        assert series("litellm_guardrail_errors_total") == [
            ({"guardrail_name": "test_guardrail", "error_type": "ValueError", "hook_type": "pre_call"}, 1.0)
        ]

    def test_record_guardrail_metrics_during_call_hook(self):
        prometheus_logger = PrometheusLogger()
        _record(
            prometheus_logger,
            guardrail_name="moderation_guardrail",
            latency_seconds=0.1,
            status="success",
            error_type=None,
            hook_type="during_call",
        )

        [call_kwargs] = labelsets("litellm_guardrail_latency_seconds")
        assert call_kwargs["hook_type"] == "during_call"

    def test_record_guardrail_metrics_with_guardrail_name_attribute(self):
        prometheus_logger = PrometheusLogger()
        _record(
            prometheus_logger,
            guardrail_name="custom_guardrail_name",
            latency_seconds=0.1,
            status="success",
            error_type=None,
            hook_type="pre_call",
        )

        [call_kwargs] = labelsets("litellm_guardrail_latency_seconds")
        assert call_kwargs["guardrail_name"] == "custom_guardrail_name"
