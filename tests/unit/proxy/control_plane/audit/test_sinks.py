from collections.abc import Mapping, Sequence
from typing import Final
import logging
from datetime import UTC, datetime

import pytest
from opentelemetry.sdk._logs import LogData
from opentelemetry.sdk._logs._internal.export.in_memory_log_exporter import InMemoryLogExporter
from opentelemetry.sdk._logs.export import LogExporter, LogExportResult

from pydantic import TypeAdapter

from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.ocsf import ChainState, Product, chain
from litellm.proxy.control_plane.audit.pipeline import Delivered, Undelivered
from litellm.proxy.control_plane.audit.sinks import LogSink, OtlpLogSink
from litellm.proxy.control_plane.common.tracing import build_resource
from litellm.proxy.control_plane.settings import ControlPlaneSettings

T0 = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
TRACE = "0af7651916cd43dd8448eb211c80319c"
SPAN = "b7ad6b7169203331"


_JSON: Final = TypeAdapter(Mapping[str, object])


def _record():
    event = AuditEvent(
        T0, "alice", None, "eu", "POST", "/key/generate", "denied", 403, None, "no", "api_activity", TRACE, SPAN
    )
    return chain(event, Product("t"), ChainState())[1]


@pytest.mark.asyncio
async def test_the_otlp_sink_emits_one_log_record_per_event_tied_to_the_request_span() -> None:
    exporter = InMemoryLogExporter()
    settings = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "sk-m"})
    sink = OtlpLogSink(exporter, build_resource(settings))
    record = _record()
    assert await sink.deliver((record,)) == Delivered(1)
    (emitted,) = exporter.get_finished_logs()
    log = emitted.log_record
    assert (format(log.trace_id, "032x"), format(log.span_id, "016x")) == (TRACE, SPAN)
    assert isinstance(log.body, str) and _JSON.validate_json(log.body) == record.body
    assert log.severity_text == "WARN"
    assert log.resource.attributes["service.name"] == settings.otel_service_name
    assert log.attributes is not None and log.attributes["event.name"] == "litellm.control_plane.audit"
    assert log.attributes["litellm.control_plane.audit.hash"] == record.hash
    assert log.timestamp == int(T0.timestamp() * 1_000_000_000)


class RefusingExporter(LogExporter):
    def export(self, batch: Sequence[LogData]) -> LogExportResult:
        return LogExportResult.FAILURE

    def shutdown(self) -> None:
        return None


@pytest.mark.asyncio
async def test_the_otlp_sink_reports_a_refused_export_so_the_worker_retries_it() -> None:
    settings = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": "sk-m"})
    assert await OtlpLogSink(RefusingExporter(), build_resource(settings)).deliver((_record(),)) == Undelivered(
        "otlp export failed"
    )


@pytest.mark.asyncio
async def test_the_log_sink_writes_the_ocsf_object_as_one_line_even_when_proxy_logs_are_at_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    record = _record()
    proxy_logger = logging.getLogger("LiteLLM Proxy")
    with caplog.at_level(logging.INFO, logger="LiteLLM Proxy.control_plane.audit"):
        proxy_logger.setLevel(logging.WARNING)
        try:
            assert await LogSink().deliver((record,)) == Delivered(1)
        finally:
            proxy_logger.setLevel(logging.NOTSET)
    (line,) = [m for m in caplog.messages if m.startswith("control_plane.audit ")]
    assert '"class_uid":6003' in line and record.hash in line and "\n" not in line
