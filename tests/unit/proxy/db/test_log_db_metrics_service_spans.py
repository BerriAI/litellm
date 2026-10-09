import asyncio
import importlib
from collections.abc import Sequence
from datetime import datetime, timedelta
from types import SimpleNamespace
from typing import Final, NotRequired, TypedDict

from typing_extensions import ReadOnly
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import StatusCode
from prisma.errors import ClientNotConnectedError
from pydantic import TypeAdapter

import litellm
from litellm._service_logger import ServiceTypes
from litellm.integrations.datadog.datadog import DataDogLogger
from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig
from litellm.proxy.db.log_db_metrics import log_db_metrics
from litellm.proxy.db.prisma_client import _PrismaDrainTracker, _TrackedPrismaEngine
from litellm.proxy.proxy_server import proxy_logging_obj
from litellm.integrations.prometheus_services import PrometheusServicesLogger
from prometheus_client import REGISTRY


class _ServiceEvent(TypedDict):
    service: ReadOnly[str]
    call_type: ReadOnly[str]
    duration: ReadOnly[float]
    is_error: ReadOnly[bool]
    error: ReadOnly[str | None]
    event_metadata: ReadOnly[dict[str, str] | None]
    table_name: NotRequired[ReadOnly[str]]


class _ServiceSpanExporter(InMemorySpanExporter):
    def __init__(self) -> None:
        super().__init__()
        self.service_span_exported: Final = asyncio.Event()

    def export(self, spans: Sequence[ReadableSpan]) -> SpanExportResult:
        result: Final = super().export(spans)
        self.service_span_exported.set()
        return result


class _Rig:
    def __init__(self, exporter: _ServiceSpanExporter, provider: TracerProvider, datadog: DataDogLogger) -> None:
        self.exporter: Final = exporter
        self.provider: Final = provider
        self.datadog: Final = datadog

    def service_spans(self) -> tuple[ReadableSpan, ...]:
        return tuple(span for span in self.exporter.get_finished_spans() if span.name != "request")

    def events(self) -> tuple[_ServiceEvent, ...]:
        adapter: Final = TypeAdapter(_ServiceEvent)
        return tuple(adapter.validate_json(entry["message"]) for entry in self.datadog.log_queue)


def _discard_periodic_flush(coroutine: object) -> None:
    close: Final = getattr(coroutine, "close")
    close()


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> _Rig:
    monkeypatch.setenv("DD_API_KEY", "test_api_key")
    monkeypatch.setenv("DD_SITE", "test.datadoghq.com")
    monkeypatch.setattr(litellm, "datadog_params", None)
    exporter: Final = _ServiceSpanExporter()
    provider: Final = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    otel: Final = OpenTelemetry(config=OpenTelemetryConfig(exporter=exporter), tracer_provider=provider)
    with patch("asyncio.create_task", side_effect=_discard_periodic_flush):
        datadog: Final = DataDogLogger()
    monkeypatch.setattr(litellm, "service_callback", [otel, datadog, "prometheus_system"])
    monkeypatch.setattr(proxy_logging_obj.service_logging_obj, "dd_logger", datadog, raising=False)
    monkeypatch.setattr(
        proxy_logging_obj.service_logging_obj, "prometheusServicesLogger", PrometheusServicesLogger(), raising=False
    )
    return _Rig(exporter, provider, datadog)


async def _run_prisma_query() -> None:
    engine: Final = _TrackedPrismaEngine(SimpleNamespace(query=AsyncMock(return_value={})), _PrismaDrainTracker())
    await engine.query("{}", tx_id=None)


@log_db_metrics
async def read_spend_rows(**kwargs: object) -> str:
    await _run_prisma_query()
    return "success"


def _logged_db_latency() -> tuple[float, float]:
    labels: Final = {ServiceTypes.DB.value: ServiceTypes.DB.value}
    total: Final = REGISTRY.get_sample_value("litellm_postgres_latency_sum", labels)
    count: Final = REGISTRY.get_sample_value("litellm_postgres_latency_count", labels)
    return (total or 0.0, count or 0.0)


def _ns(moment: datetime) -> int:
    return int(moment.timestamp() * 1e9)


_DB_CALL_START: Final = datetime(2026, 1, 1, 12, 0, 0)
_DB_CALL_DURATION: Final = timedelta(milliseconds=250)


class _ScriptedClock:
    def __init__(self, *moments: datetime) -> None:
        self._moments: Final = iter(moments)

    def now(self) -> datetime:
        return next(self._moments)


@pytest.fixture
def scripted_db_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        importlib.import_module("litellm.proxy.db.log_db_metrics"),
        "datetime",
        _ScriptedClock(_DB_CALL_START, _DB_CALL_START + _DB_CALL_DURATION),
    )


@pytest.mark.asyncio
async def test_a_db_success_is_reported_on_the_parent_span_with_its_duration_and_times(
    rig: _Rig, scripted_db_clock: None
) -> None:
    parent: Final = rig.provider.get_tracer("db-test").start_span("request")
    latency_before: Final = _logged_db_latency()

    result: Final = await read_spend_rows(parent_otel_span=parent)
    await asyncio.wait_for(rig.exporter.service_span_exported.wait(), timeout=10)

    assert result == "success"
    spans: Final = rig.service_spans()
    assert len(spans) == 1
    span: Final = spans[0]
    assert span.parent is not None
    assert span.parent.span_id == parent.get_span_context().span_id
    assert span.attributes is not None
    assert span.attributes["service"] == ServiceTypes.DB.value
    assert span.attributes["call_type"] == "read_spend_rows"
    assert span.status.status_code == StatusCode.OK
    assert span.start_time == _ns(_DB_CALL_START)
    assert span.end_time == _ns(_DB_CALL_START + _DB_CALL_DURATION)
    latency_after: Final = _logged_db_latency()
    assert latency_after[1] - latency_before[1] == 1
    assert latency_after[0] - latency_before[0] == pytest.approx(_DB_CALL_DURATION.total_seconds())
    assert rig.events() == ()


@pytest.mark.asyncio
async def test_db_event_metadata_names_only_the_table_and_never_the_raw_kwargs(rig: _Rig) -> None:
    parent: Final = rig.provider.get_tracer("db-test").start_span("request")

    await read_spend_rows(
        parent_otel_span=parent,
        table_name="LiteLLM_SpendLogs",
        token="sk-secret-should-not-leak",
        prisma_client=object(),
    )
    await asyncio.wait_for(rig.exporter.service_span_exported.wait(), timeout=10)

    span_attributes: Final = rig.service_spans()[0].attributes
    assert span_attributes is not None
    assert span_attributes["table_name"] == "LiteLLM_SpendLogs"
    assert not {"token", "prisma_client", "parent_otel_span"} & set(span_attributes)
    assert all("sk-secret-should-not-leak" not in str(value) for value in span_attributes.values())


@pytest.mark.asyncio
async def test_the_logged_db_duration_is_the_span_wall_clock_of_the_wrapped_call(
    rig: _Rig, scripted_db_clock: None
) -> None:
    parent: Final = rig.provider.get_tracer("db-test").start_span("request")
    latency_before: Final = _logged_db_latency()

    await read_spend_rows(parent_otel_span=parent)
    await asyncio.wait_for(rig.exporter.service_span_exported.wait(), timeout=10)

    span: Final = rig.service_spans()[0]
    assert span.start_time is not None and span.end_time is not None
    latency_after: Final = _logged_db_latency()
    logged_duration: Final = latency_after[0] - latency_before[0]
    assert latency_after[1] - latency_before[1] == 1
    assert logged_duration == pytest.approx((span.end_time - span.start_time) / 1e9, rel=1e-3, abs=2e-6)
    assert logged_duration == pytest.approx(_DB_CALL_DURATION.total_seconds())


@log_db_metrics
async def disconnected_read(**kwargs: object) -> str:
    raise ClientNotConnectedError()


@pytest.mark.asyncio
async def test_a_prisma_error_is_reported_as_a_db_failure_and_reraised(rig: _Rig) -> None:
    parent: Final = rig.provider.get_tracer("db-test").start_span("request")

    with pytest.raises(ClientNotConnectedError, match="Client is not connected to the query engine"):
        await disconnected_read(parent_otel_span=parent)

    spans: Final = rig.service_spans()
    assert len(spans) == 1
    assert spans[0].parent is not None
    assert spans[0].parent.span_id == parent.get_span_context().span_id
    assert spans[0].status.status_code == StatusCode.ERROR
    assert spans[0].attributes is not None
    assert spans[0].attributes["call_type"] == "disconnected_read"
    assert spans[0].attributes["service"] == ServiceTypes.DB.value
    assert "Client is not connected" in str(spans[0].attributes["error"])
    events: Final = rig.events()
    assert len(events) == 1
    assert events[0]["is_error"] is True
    assert events[0]["call_type"] == "disconnected_read"
    assert isinstance(events[0]["duration"], float)
    assert "Client is not connected" in (events[0]["error"] or "")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("error", "is_db_error"),
    [
        (ValueError("Generic error"), False),
        (KeyError("Missing key"), False),
        (TypeError("Type error"), False),
        (httpx.ConnectError("Failed to connect"), True),
        (httpx.TimeoutException("Request timed out"), True),
        (ClientNotConnectedError(), True),
    ],
)
async def test_only_db_errors_are_reported_as_db_failures(rig: _Rig, error: Exception, is_db_error: bool) -> None:
    parent: Final = rig.provider.get_tracer("db-test").start_span("request")

    @log_db_metrics
    async def failing_read(**kwargs: object) -> str:
        raise error

    with pytest.raises(type(error)):
        await failing_read(parent_otel_span=parent)

    spans: Final = rig.service_spans()
    events: Final = rig.events()
    if is_db_error:
        assert [span.status.status_code for span in spans] == [StatusCode.ERROR]
        assert [(event["service"], event["call_type"], event["is_error"]) for event in events] == [
            (ServiceTypes.DB.value, "failing_read", True)
        ]
        assert isinstance(events[0]["duration"], float)
    else:
        assert spans == ()
        assert events == ()
