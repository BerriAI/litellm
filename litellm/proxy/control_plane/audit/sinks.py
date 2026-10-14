"""Built-in audit sinks: the stdout log line and an OTel log record over OTLP. Each is one `AuditSink`."""

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Final

from opentelemetry._logs import SeverityNumber
from opentelemetry.sdk._logs import LogData, LogRecord
from opentelemetry.sdk._logs.export import LogExporter, LogExportResult
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.util.instrumentation import InstrumentationScope
from opentelemetry.trace import TraceFlags
from pydantic import TypeAdapter

from litellm._logging import verbose_proxy_logger
from litellm.proxy.control_plane.audit.ocsf import AuditRecord
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, DeliveryResult, Undelivered

EVENT_NAME: Final = "litellm.control_plane.audit"
SCOPE: Final = InstrumentationScope(EVENT_NAME)
_VERDICT: Final = TypeAdapter(LogExportResult)
LOG_PREFIX: Final = "control_plane.audit"
AUDIT_LOGGER: Final = verbose_proxy_logger.getChild(LOG_PREFIX)
AUDIT_LOGGER.setLevel(logging.INFO)


def _line(record: AuditRecord) -> str:
    return json.dumps(record.body, default=str, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class LogSink:
    name: str = "log"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        for record in batch:
            AUDIT_LOGGER.info("%s %s", LOG_PREFIX, _line(record))
        return Delivered(len(batch))

    async def close(self) -> None:
        return None


def _severity(record: AuditRecord) -> tuple[str, SeverityNumber]:
    return ("INFO", SeverityNumber.INFO) if record.event.decision == "allowed" else ("WARN", SeverityNumber.WARN)


def _ids(record: AuditRecord) -> tuple[int, int]:
    trace_id: Final = int(record.event.trace_id, 16) if record.event.trace_id else 0
    span_id: Final = int(record.event.span_id, 16) if record.event.span_id else 0
    return trace_id, span_id


def _attributes(record: AuditRecord) -> Mapping[str, str | int]:
    return {
        "event.name": EVENT_NAME,
        "enduser.id": record.event.actor,
        "http.request.method": record.event.method,
        "url.path": record.event.path,
        "litellm.control_plane.decision": record.event.decision,
        "litellm.control_plane.dataplane": record.event.dataplane_id,
        "litellm.control_plane.audit.sequence": record.sequence,
        "litellm.control_plane.audit.hash": record.hash,
    }


def _log_data(record: AuditRecord, resource: Resource) -> LogData:
    trace_id, span_id = _ids(record)
    severity_text, severity_number = _severity(record)
    return LogData(
        LogRecord(
            timestamp=int(record.event.at.timestamp() * 1_000_000_000),
            observed_timestamp=time.time_ns(),
            resource=resource,
            trace_id=trace_id,
            span_id=span_id,
            trace_flags=TraceFlags(TraceFlags.SAMPLED if trace_id else TraceFlags.DEFAULT),
            severity_text=severity_text,
            severity_number=severity_number,
            body=_line(record),
            attributes=dict(_attributes(record)),
        ),
        SCOPE,
    )


@dataclass(frozen=True, slots=True)
class OtlpLogSink:
    """One OTel log record per audit event, body is the OCSF object, trace and span ids point at the request span.
    The batch goes straight to the exporter, off the event loop, so its verdict is the delivery result and a
    collector that refuses the batch makes the worker retry instead of the SDK dropping it later."""

    exporter: LogExporter
    resource: Resource
    name: str = "otlp"

    async def deliver(self, batch: Batch) -> DeliveryResult:
        data: Final = tuple(_log_data(record, self.resource) for record in batch)
        try:
            result: Final = await asyncio.to_thread(self.exporter.export, data)
        except Exception as error:  # noqa: BLE001  # an exporter that raises instead of returning FAILURE is still a failed delivery
            return Undelivered(f"otlp export raised {type(error).__name__}")
        match _VERDICT.validate_python(result):
            case LogExportResult.SUCCESS:
                return Delivered(len(batch))
            case LogExportResult.FAILURE:
                return Undelivered("otlp export failed")

    async def close(self) -> None:
        await asyncio.to_thread(self.exporter.shutdown)
