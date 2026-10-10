"""
Guards that keep batch line-item callback events out of the built-in metering sinks.

With ``litellm.store_batch_line_items_in_callbacks`` on, a completed batch emits
one child callback per JSONL line on top of the aggregate ``aretrieve_batch``
event. Every line carries a real ``response_cost``, so any billing/metering sink
without the ``is_batch_line_item_event`` guard meters aggregate + per-line and
reports roughly 2x the true spend.

These tests put REAL sink instances on the REAL dispatch lists, drive the REAL
``Logging.async_success_handler`` for a completed 2-line batch (1 success line +
1 error line, aggregate cost $1.50, per-line cost $0.03), and assert each sink
meters exactly the aggregate. OTel spans for line items must still be emitted:
the guard belongs on cost/token metrics, not on tracing.
"""

import asyncio
import io
import json
import os
import time
import uuid
from contextlib import redirect_stdout
from datetime import datetime
from types import SimpleNamespace
from typing import Any, Final
from unittest.mock import AsyncMock, patch

import pytest

import litellm
from litellm.batches.batch_line_item_logging import batch_line_item_claim_cache, pending_batch_line_item_tasks
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.types.utils import LiteLLMBatch, Usage

AGGREGATE_COST: Final[float] = 1.5

INPUT_JSONL: Final[bytes] = b"\n".join(
    [
        json.dumps(
            {
                "custom_id": "a",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi a"}]},
            }
        ).encode(),
        json.dumps(
            {
                "custom_id": "b",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi b"}]},
            }
        ).encode(),
    ]
)

OUTPUT_JSONL: Final[bytes] = json.dumps(
    {
        "custom_id": "a",
        "response": {
            "status_code": 200,
            "body": {
                "id": "chatcmpl-line-1",
                "model": "gpt-4o",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            },
        },
    }
).encode()

ERROR_JSONL: Final[bytes] = json.dumps(
    {
        "custom_id": "b",
        "response": {"status_code": 400, "body": {"error": {"message": "boom"}}},
        "error": {"message": "boom"},
    }
).encode()

_FILE_BYTES: Final[dict[str, bytes]] = {
    "input-file-1": INPUT_JSONL,
    "output-file-1": OUTPUT_JSONL,
    "error-file-1": ERROR_JSONL,
}


def _file_content(file_id: str, **_kwargs: Any) -> SimpleNamespace:
    return SimpleNamespace(content=_FILE_BYTES[file_id])


def _batch() -> LiteLLMBatch:
    return LiteLLMBatch(
        id=f"batch_{uuid.uuid4().hex[:8]}",
        object="batch",
        endpoint="/v1/chat/completions",
        input_file_id="input-file-1",
        output_file_id="output-file-1",
        error_file_id="error-file-1",
        status="completed",
        completion_window="24h",
        created_at=1,
    )


def _parent_logging() -> Logging:
    logging_obj = Logging(
        model="gpt-4o",
        messages=[{"role": "user", "content": "<retrieve_batch>"}],
        stream=False,
        call_type="aretrieve_batch",
        start_time=datetime.now(),
        litellm_call_id=str(uuid.uuid4()),
        function_id=str(uuid.uuid4()),
    )
    logging_obj.update_environment_variables(
        litellm_params={
            "metadata": {
                "model_info": {"id": "dep-1"},
                "model_group": "gpt-4o",
                "user_api_key_user_id": "user-77",
                "user_api_key_team_id": "team-7",
                "user_api_key_team_alias": "team-seven",
                "user_api_key_alias": "key-alias-1",
            }
        },
        optional_params={},
        custom_llm_provider="openai",
    )
    return logging_obj


class RecordingHTTP:
    """Stands in for a sink's HTTP egress object only; the sink logic is real."""

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    async def post(self, url: str, data: Any = None, content: Any = None, **_kw: Any) -> SimpleNamespace:
        self.posts.append({"url": url, "body": data if data is not None else content})
        return SimpleNamespace(status_code=200, text="ok", raise_for_status=lambda: None)

    async def put(self, url: str, content: Any = None, **_kw: Any) -> SimpleNamespace:
        return SimpleNamespace(status_code=202, text="ok", raise_for_status=lambda: None)


async def _log_completed_batch(monkeypatch: pytest.MonkeyPatch, loggers: list, flag: bool = True) -> None:
    batch_line_item_claim_cache.in_memory_cache.flush_cache()
    monkeypatch.setattr(litellm, "store_batch_line_items_in_callbacks", flag, raising=False)
    saved_success = list(litellm._async_success_callback)
    saved_failure = list(litellm._async_failure_callback)
    litellm._async_success_callback = list(loggers)  # test-quality-ok: no injection seam; finally restores
    litellm._async_failure_callback = list(loggers)  # test-quality-ok: same dispatch seam
    buf = io.StringIO()
    try:
        with (
            patch("litellm.files.main.afile_content", new_callable=AsyncMock, side_effect=_file_content),
            patch("litellm.cost_calculator.batch_cost_calculator", return_value=(0.01, 0.02)),
            redirect_stdout(buf),
        ):
            await _parent_logging().async_success_handler(
                result=_batch(),
                batch_cost=1.5,
                batch_usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15),
                batch_models=["gpt-4o"],
                batch_successful_requests=1,
                batch_failed_requests=1,
                batch_prompt_cost=1.0,
                batch_completion_cost=0.5,
            )
            await asyncio.gather(*pending_batch_line_item_tasks())
    finally:
        litellm._async_success_callback = saved_success  # test-quality-ok: restores the seam
        litellm._async_failure_callback = saved_failure  # test-quality-ok: restores the seam


def _openmeter_sinks() -> tuple[Any, RecordingHTTP]:
    from litellm.integrations.openmeter import OpenMeterLogger

    recorder = RecordingHTTP()
    logger = OpenMeterLogger()
    logger.async_http_handler = recorder
    return logger, recorder


def _lago_sinks() -> tuple[Any, RecordingHTTP]:
    from litellm.integrations.lago import LagoLogger

    recorder = RecordingHTTP()
    logger = LagoLogger()
    logger.async_http_handler = recorder
    return logger, recorder


@pytest.mark.asyncio
async def test_openmeter_meters_only_the_aggregate_batch_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENMETER_API_KEY", "test-openmeter-key")
    logger, recorder = _openmeter_sinks()

    await _log_completed_batch(monkeypatch, [logger])

    costs = [json.loads(p["body"])["data"]["cost"] for p in recorder.posts]
    assert costs == [AGGREGATE_COST], (
        f"OpenMeter metered per-line costs on top of the aggregate: {costs}; "
        "line items must be billed only by the aggregate aretrieve_batch event"
    )


@pytest.mark.asyncio
async def test_lago_bills_only_the_aggregate_batch_event(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LAGO_API_KEY", "test-lago-key")
    monkeypatch.setenv("LAGO_API_BASE", "http://lago.invalid")
    monkeypatch.setenv("LAGO_API_EVENT_CODE", "litellm-usage")
    monkeypatch.setenv("LAGO_API_CHARGE_BY", "user_id")
    logger, recorder = _lago_sinks()

    await _log_completed_batch(monkeypatch, [logger])

    costs = [json.loads(p["body"])["event"]["properties"]["response_cost"] for p in recorder.posts]
    assert costs == [AGGREGATE_COST], (
        f"Lago billed per-line costs on top of the aggregate: {costs}; "
        "line items must be billed only by the aggregate aretrieve_batch event"
    )


@pytest.mark.asyncio
async def test_datadog_cost_management_bills_only_the_aggregate_batch_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from litellm.integrations.datadog.datadog_cost_management import DatadogCostManagementLogger

    monkeypatch.setenv("DD_API_KEY", "test-dd-key")
    monkeypatch.setenv("DD_APP_KEY", "test-dd-app-key")
    logger = DatadogCostManagementLogger(cost_tag_keys=[])

    await _log_completed_batch(monkeypatch, [logger])

    entries = list(logger.log_queue)
    costs = [e.get("response_cost", 0) for e in entries]
    assert costs == [AGGREGATE_COST], (
        f"Datadog FOCUS BilledCost queued per-line entries on top of the aggregate: {costs}; "
        "line items must be billed only by the aggregate aretrieve_batch event"
    )


@pytest.mark.asyncio
async def test_newrelic_metrics_meters_only_the_aggregate_batch_event(monkeypatch: pytest.MonkeyPatch) -> None:
    from litellm.integrations.newrelic.newrelic_metrics import NewRelicMetricsLogger, build_metric_payload

    logger = NewRelicMetricsLogger(newrelic_api_key="test-nr-key")

    await _log_completed_batch(monkeypatch, [logger])

    records = tuple(logger.log_queue)
    costs = [r.response_cost for r in records]
    assert costs == [AGGREGATE_COST], (
        f"New Relic metered per-line costs on top of the aggregate: {costs}; "
        "line items must be metered only by the aggregate aretrieve_batch event"
    )
    now = time.time()
    envelopes = build_metric_payload(records=records, window_start=now - 1, now=now)
    cost_sum = 0.0
    for envelope in envelopes:
        for metric in envelope["metrics"]:
            if "cost" in metric["name"]:
                value = metric["value"]["sum"] if isinstance(metric["value"], dict) else metric["value"]
                cost_sum += value
    assert cost_sum == pytest.approx(AGGREGATE_COST)


@pytest.mark.asyncio
async def test_prometheus_meters_only_the_aggregate_batch_event(monkeypatch: pytest.MonkeyPatch) -> None:
    prometheus_client = pytest.importorskip("prometheus_client")
    from litellm.integrations.prometheus import PrometheusLogger
    from prometheus_client import REGISTRY

    saved_collectors = list(REGISTRY._collector_to_names.keys())
    for collector in saved_collectors:
        REGISTRY.unregister(collector)
    logger = PrometheusLogger()

    try:
        await _log_completed_batch(monkeypatch, [logger])
        spend_samples = []
        for metric in REGISTRY.collect():
            if metric.name == "litellm_spend_metric":
                spend_samples = [sample.value for sample in metric.samples if sample.name.endswith("_total")]
    finally:
        for collector in list(REGISTRY._collector_to_names.keys()):
            REGISTRY.unregister(collector)
        for collector in saved_collectors:
            try:
                REGISTRY.register(collector)
            except Exception:  # noqa: BLE001  # already re-registered by another holder
                pass

    assert spend_samples, "litellm_spend_metric saw no samples at all"
    assert sum(spend_samples) == pytest.approx(AGGREGATE_COST), (
        f"Prometheus spend metric double-counted batch line items: {spend_samples}; "
        "line items must be metered only by the aggregate aretrieve_batch event"
    )


def _otel_v1(monkeypatch: pytest.MonkeyPatch):
    otel_sdk = pytest.importorskip("opentelemetry.sdk")
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from litellm.integrations.opentelemetry import OpenTelemetry as OTelV1, OpenTelemetryConfig

    reader = InMemoryMetricReader()
    span_exporter = InMemorySpanExporter()
    tracer_provider = TracerProvider()
    tracer_provider.add_span_processor(SimpleSpanProcessor(span_exporter))
    logger = OTelV1(
        config=OpenTelemetryConfig(exporter="console", enable_metrics=True),
        callback_name="batch_sink_guard_v1",
        tracer_provider=tracer_provider,
        meter_provider=MeterProvider(metric_readers=[reader]),
    )
    return logger, reader, span_exporter


def _cost_points(reader: Any) -> list[float]:
    data = reader.get_metrics_data()
    points: list[float] = []
    if data is None:
        return points
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                if not metric.name.endswith("cost"):
                    continue
                for data_point in metric.data.data_points:
                    points.append(getattr(data_point, "sum", getattr(data_point, "value", None)))
    return points


@pytest.mark.asyncio
async def test_otel_v1_meter_cost_skips_line_items_but_spans_stay(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger, reader, span_exporter = _otel_v1(monkeypatch)

    await _log_completed_batch(monkeypatch, [logger])

    costs = _cost_points(reader)
    assert sum(costs) == pytest.approx(AGGREGATE_COST), (
        f"OTel v1 gen_ai.usage.cost double-counted batch line items: {costs}; "
        "line items must be metered only by the aggregate aretrieve_batch event"
    )
    finished = span_exporter.get_finished_spans()
    assert finished, "OTel v1 emitted no spans at all"
    assert any("chatcmpl-line-1" in str(span.attributes) for span in finished), (
        f"OTel v1 dropped the per-line span the feature exists to deliver: {[span.name for span in finished]}"
    )


def _reader_snapshot(reader: Any) -> dict[str, list[float]]:
    """Every metric's datapoint values, keyed by metric name (sums for histograms)."""
    data = reader.get_metrics_data()
    out: dict[str, list[float]] = {}
    if data is None:
        return out
    for resource_metrics in data.resource_metrics:
        for scope_metrics in resource_metrics.scope_metrics:
            for metric in scope_metrics.metrics:
                for data_point in metric.data.data_points:
                    value = getattr(data_point, "sum", None)
                    if value is None:
                        value = getattr(data_point, "value", None)
                    if value is not None:
                        out.setdefault(metric.name, []).append(value)
    return out


async def _run_batch_with_otel_v2(monkeypatch: pytest.MonkeyPatch, flag: bool) -> dict[str, list[float]]:
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import InMemoryMetricReader
    from opentelemetry.sdk.trace import TracerProvider

    from litellm.integrations.otel.logger import OpenTelemetryV2
    from litellm.integrations.otel.model.config import OpenTelemetryV2Config

    reader = InMemoryMetricReader()
    logger = OpenTelemetryV2(
        config=OpenTelemetryV2Config(exporter="console", enable_metrics=True),
        callback_name="batch_sink_guard_v2",
        tracer_provider=TracerProvider(),
        meter_provider=MeterProvider(metric_readers=[reader]),
    )
    await _log_completed_batch(monkeypatch, [logger], flag=flag)
    return _reader_snapshot(reader)


@pytest.mark.asyncio
async def test_otel_v2_metrics_are_identical_with_the_flag_on_and_off(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("opentelemetry.sdk")

    off = await _run_batch_with_otel_v2(monkeypatch, flag=False)
    on = await _run_batch_with_otel_v2(monkeypatch, flag=True)

    for name in ("gen_ai.usage.cost", "gen_ai.client.token.usage"):
        assert on.get(name) == off.get(name), (
            f"OTel v2 {name} changed when the line-item flag was turned on; line items "
            "must be metered only by the aggregate aretrieve_batch event. "
            f"off={off.get(name)} on={on.get(name)}"
        )
    for name, values in off.items():
        on_counts = [len(on.get(name, [])), len(values)]
        assert on_counts[0] == on_counts[1], (
            f"OTel v2 {name} gained synthetic per-line samples with the flag on: "
            f"{on_counts[0]} datapoints on vs {on_counts[1]} off"
        )
    assert sum(off.get("gen_ai.usage.cost", [])) == pytest.approx(AGGREGATE_COST)
