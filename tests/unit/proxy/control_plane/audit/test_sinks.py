import asyncio
import hashlib
import io
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from threading import Event, Lock
from typing import Final

import pytest
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import InMemoryLogRecordExporter, LogRecordExporter, LogRecordExportResult
from pydantic import TypeAdapter

from litellm.proxy.control_plane.audit.event import AuditEvent
from litellm.proxy.control_plane.audit.log import build_audit_log
from litellm.proxy.control_plane.audit.ocsf import HASH_PREFIX, ChainState, Product, canonical, chain, section
from litellm.proxy.control_plane.audit.pipeline import Accepted, Delivered, Undelivered, build_pipeline
from litellm.proxy.control_plane.audit.sinks import LogSink, OtlpLogSink
from litellm.proxy.control_plane.common.tracing import build_resource
from litellm.proxy.control_plane.settings import ControlPlaneSettings
from tests.unit.proxy.control_plane.support import MASTER_KEY

T0: Final = datetime(2026, 10, 13, 12, 0, tzinfo=UTC)
TRACE: Final = "0af7651916cd43dd8448eb211c80319c"
SPAN: Final = "b7ad6b7169203331"


_JSON: Final = TypeAdapter(Mapping[str, object])


def _record():
    event: Final = AuditEvent(
        T0, "alice", None, "eu", "POST", "/key/generate", "denied", 403, None, "no", "api_activity", TRACE, SPAN
    )
    return chain(event, Product("t"), ChainState())[1]


@pytest.mark.asyncio
async def test_the_otlp_sink_emits_one_log_record_per_event_tied_to_the_request_span() -> None:
    exporter: Final = InMemoryLogRecordExporter()
    settings: Final = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY})
    sink: Final = OtlpLogSink(exporter, build_resource(settings))
    record: Final = _record()
    assert await sink.deliver((record,)) == Delivered(1)
    (emitted,) = exporter.get_finished_logs()
    log: Final = emitted.log_record
    assert (format(log.trace_id, "032x"), format(log.span_id, "016x")) == (TRACE, SPAN)
    assert isinstance(log.body, str) and _JSON.validate_json(log.body) == record.body
    assert log.severity_text == "WARN"
    assert emitted.resource.attributes["service.name"] == settings.otel_service_name
    assert log.attributes is not None and log.attributes["event.name"] == "litellm.control_plane.audit"
    assert log.attributes["litellm.control_plane.audit.hash"] == record.hash
    assert log.timestamp == int(T0.timestamp() * 1_000_000_000)


class RefusingExporter(LogRecordExporter):
    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        return LogRecordExportResult.FAILURE

    def shutdown(self) -> None:
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@dataclass
class BlockingExporter(LogRecordExporter):  # mutable-ok: coordinates exporter state with its worker thread
    export_started: Event = field(default_factory=Event)
    export_release: Event = field(default_factory=Event)
    export_finished: Event = field(default_factory=Event)
    lock: Lock = field(default_factory=Lock)
    export_calls: int = 0
    shutdown_calls: int = 0
    exporting: bool = False
    shutdown_overlapped_export: bool = False

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogRecordExportResult:
        with self.lock:
            self.export_calls += 1
            self.exporting = True
        self.export_started.set()
        self.export_release.wait()
        with self.lock:
            self.exporting = False
            self.export_finished.set()
        return LogRecordExportResult.SUCCESS

    def shutdown(self) -> None:
        with self.lock:
            self.shutdown_calls += 1
            self.shutdown_overlapped_export = self.shutdown_overlapped_export or self.exporting

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


@pytest.mark.asyncio
async def test_the_otlp_sink_reports_a_refused_export_so_the_worker_retries_it() -> None:
    settings: Final = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY})
    assert await OtlpLogSink(RefusingExporter(), build_resource(settings)).deliver((_record(),)) == Undelivered(
        "otlp export failed"
    )


@pytest.mark.asyncio
async def test_worker_close_waits_for_an_inflight_otlp_export_before_shutdown() -> None:
    settings: Final = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY})
    exporter: Final = BlockingExporter()
    sink: Final = OtlpLogSink(exporter, build_resource(settings))
    pipeline: Final = build_pipeline((sink,), queue_size=2, batch_size=1, strict=False)
    pipeline.start()
    assert pipeline.publish(_record()) == (Accepted(),)
    (worker,) = pipeline.workers

    try:
        assert await asyncio.to_thread(exporter.export_started.wait, 5)
        close_task: Final = asyncio.create_task(worker.close(drain_seconds=5))
        await asyncio.sleep(0)
        exporter.export_release.set()
        await close_task
    finally:
        exporter.export_release.set()
        if exporter.export_started.is_set():
            assert await asyncio.to_thread(exporter.export_finished.wait, 5)

    assert exporter.export_calls == 1
    assert exporter.shutdown_calls == 1
    assert not exporter.shutdown_overlapped_export


@pytest.mark.asyncio
async def test_worker_close_stops_waiting_for_an_inflight_export_at_the_drain_deadline() -> None:
    settings: Final = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY})
    exporter: Final = BlockingExporter()
    sink: Final = OtlpLogSink(exporter, build_resource(settings))
    pipeline: Final = build_pipeline((sink,), queue_size=2, batch_size=1, strict=False)
    pipeline.start()
    assert pipeline.publish(_record()) == (Accepted(),)
    (worker,) = pipeline.workers
    assert await asyncio.to_thread(exporter.export_started.wait, 5)

    try:
        await asyncio.wait_for(worker.close(drain_seconds=0.1), timeout=1)
        assert not exporter.export_release.is_set()
        assert exporter.shutdown_calls == 1
    finally:
        exporter.export_release.set()
        assert await asyncio.to_thread(exporter.export_finished.wait, 5)


@pytest.mark.asyncio
async def test_the_otlp_sink_retries_when_batch_construction_raises() -> None:
    settings: Final = ControlPlaneSettings.model_validate({"LITELLM_MASTER_KEY": MASTER_KEY})
    event: Final = AuditEvent(
        T0,
        "alice",
        None,
        "eu",
        "POST",
        "/key/generate",
        "denied",
        403,
        None,
        "no",
        "api_activity",
        "not-hex",
        SPAN,
    )
    record: Final = chain(event, Product("t"), ChainState())[1]
    result: Final = await OtlpLogSink(InMemoryLogRecordExporter(), build_resource(settings)).deliver((record,))
    assert result == Undelivered("otlp export raised ValueError")


@pytest.mark.asyncio
async def test_the_log_sink_writes_the_ocsf_object_as_one_line_even_when_proxy_logs_are_at_warning() -> None:
    record: Final = _record()
    proxy_logger: Final = logging.getLogger("LiteLLM Proxy")
    stream: Final = io.StringIO()
    previous_level: Final = proxy_logger.level
    proxy_logger.setLevel(logging.WARNING)
    try:
        assert await LogSink(stream=stream).deliver((record,)) == Delivered(1)
    finally:
        proxy_logger.setLevel(previous_level)
    line: Final = stream.getvalue().strip()
    assert '"class_uid":6003' in line and record.hash in line and "\n" not in line


@pytest.mark.asyncio
async def test_the_log_sink_preserves_the_seal_and_registered_secret_looking_dataplane_id() -> None:
    dataplane_id: Final = "sk-production-dataplane"
    audit: Final = build_audit_log(
        now=lambda: T0,
        product=Product("t"),
        trace=lambda: (TRACE, SPAN),
        known_dataplane_ids=frozenset({dataplane_id}),
    )
    audit.record("alice", None, dataplane_id, "POST", "/key/list", "allowed", 200, credential="service_token")
    record: Final = audit.recent_records(1)[0]
    stream: Final = io.StringIO()
    assert await LogSink(stream=stream).deliver((record,)) == Delivered(1)

    line: Final = stream.getvalue().strip()
    assert line.startswith("control_plane.audit ")
    body: Final = _JSON.validate_json(line.removeprefix("control_plane.audit "))
    litellm_block: Final = section(body, "unmapped", "litellm")
    printed_hash: Final = litellm_block["hash"]
    hashless_body: Final = {
        **body,
        "unmapped": {
            "litellm": {key: value for key, value in litellm_block.items() if key != "hash"},
        },
    }

    assert printed_hash == record.hash
    assert printed_hash == HASH_PREFIX + hashlib.sha256(canonical(hashless_body)).hexdigest()
    assert section(body, "api", "service")["name"] == dataplane_id


def test_audit_log_redacts_the_http_method_before_chaining() -> None:
    method: Final = "sk-production-dataplane"
    audit: Final = build_audit_log(now=lambda: T0, product=Product("t"), trace=lambda: (TRACE, SPAN))
    event: Final = audit.record("alice", None, "eu", method, "/key/list", "allowed", 200)
    record: Final = audit.recent_records(1)[0]

    assert event.method != method
    assert section(record.body, "http_request")["http_method"] == event.method
