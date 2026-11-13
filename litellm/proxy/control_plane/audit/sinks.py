from __future__ import annotations

import asyncio
import json
import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, TextIO

from opentelemetry._logs import LogRecord, SeverityNumber
from opentelemetry.exporter.http.transport._base import BaseHTTPResult, BaseHTTPTransport
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
from opentelemetry.proto.collector.logs.v1.logs_service_pb2 import ExportLogsServiceResponse
from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState, set_span_in_context

from litellm.proxy.control_plane.audit.log import AUDIT_LOGGER, build_audit_logger
from litellm.proxy.control_plane.audit.ocsf import AuditRecord
from litellm.proxy.control_plane.audit.pipeline import Batch, Delivered, DeliveryResult, Rejected, Undelivered

if TYPE_CHECKING:
    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.sdk._logs.export import LogRecordExporter
    from opentelemetry.sdk.resources import Resource

    from litellm.proxy.control_plane.audit.grpc import GrpcResponses

EVENT_NAME: Final = "litellm.control_plane.audit"
LOG_PREFIX: Final = "control_plane.audit"


def _line(record: AuditRecord) -> str:
    return json.dumps(record.body, default=str, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class LogSink:
    name: str = "log"
    stream: TextIO | None = None
    _logger: logging.Logger = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "_logger", AUDIT_LOGGER if self.stream is None else build_audit_logger(self.stream))

    async def deliver(self, batch: Batch) -> DeliveryResult:
        for record in batch:
            self._logger.info("%s %s", LOG_PREFIX, _line(record))
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


def _log_data(record: AuditRecord, resource: Resource) -> ReadableLogRecord:
    from opentelemetry.sdk._logs import ReadableLogRecord
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

    trace_id, span_id = _ids(record)
    severity_text, severity_number = _severity(record)
    trace_flags: Final = TraceFlags(TraceFlags.SAMPLED if trace_id else TraceFlags.DEFAULT)
    return ReadableLogRecord(
        LogRecord(
            timestamp=int(record.event.at.timestamp() * 1_000_000_000),
            observed_timestamp=time.time_ns(),
            context=set_span_in_context(
                NonRecordingSpan(
                    SpanContext(
                        trace_id=trace_id,
                        span_id=span_id,
                        is_remote=False,
                        trace_flags=trace_flags,
                        trace_state=TraceState(),
                    )
                )
            ),
            severity_text=severity_text,
            severity_number=severity_number,
            body=_line(record),
            attributes=dict(_attributes(record)),
        ),
        resource,
        InstrumentationScope(EVENT_NAME),
    )


@dataclass(slots=True)
class OtlpResponses(BaseHTTPTransport):
    transport: BaseHTTPTransport
    rejected: int = 0
    permanent_status: int | None = None

    def request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | None = None,
        data: bytes | None = None,
    ) -> BaseHTTPResult:
        response: Final = self.transport.request(
            method, url, headers=None if headers is None else dict(headers), timeout=timeout, data=data
        )
        status: Final = response.status_code
        if status == 200:
            self.rejected += ExportLogsServiceResponse.FromString(
                response.content()
            ).partial_success.rejected_log_records
        elif isinstance(status, int) and 400 <= status < 500 and status not in (408, 429):
            self.permanent_status = status
        return response

    def is_connection_error(self, exception: Exception | None) -> bool:
        return self.transport.is_connection_error(exception)

    def close(self) -> None:
        self.transport.close()


@dataclass(frozen=True, slots=True)
class OtlpLogSink:
    """One OTel log record per audit event, body is the OCSF object, trace and span ids point at the request span.
    The batch goes straight to the exporter, off the event loop, so its verdict is the delivery result and a
    collector that refuses the batch makes the worker retry instead of the SDK dropping it later."""

    exporter: LogRecordExporter
    resource: Resource
    name: str = "otlp"
    _responses: OtlpResponses | GrpcResponses | None = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if isinstance(self.exporter, OTLPLogExporter):
            client: Final = self.exporter._client  # pyright: ignore[reportPrivateUsage]  # locked SDK exposes HTTP responses only through its transport
            responses: Final = OtlpResponses(client._transport)  # pyright: ignore[reportPrivateUsage]  # preserve the SDK-configured transport and TLS
            client._transport = responses  # pyright: ignore[reportPrivateUsage]  # intercept responses without replacing SDK retry or session configuration
            object.__setattr__(self, "_responses", responses)
        else:
            try:
                from litellm.proxy.control_plane.audit.grpc import capture_grpc_responses
            except ImportError:
                object.__setattr__(self, "_responses", None)
            else:
                object.__setattr__(self, "_responses", capture_grpc_responses(self.exporter))

    async def deliver(self, batch: Batch) -> DeliveryResult:
        from opentelemetry.sdk._logs.export import LogRecordExportResult
        from pydantic import TypeAdapter

        try:
            if self._responses is not None:
                self._responses.rejected = 0
                self._responses.permanent_status = None
            data: Final = tuple(_log_data(record, self.resource) for record in batch)
            result: Final = await asyncio.to_thread(self.exporter.export, data)
        except Exception as error:  # noqa: BLE001  # an exporter that raises instead of returning FAILURE is still a failed delivery
            return Undelivered(f"otlp export raised {type(error).__name__}")
        if self._responses is not None:
            if self._responses.permanent_status is not None:
                status: Final = self._responses.permanent_status
                reason: Final = f"HTTP {status}" if isinstance(status, int) else status
                return Rejected(f"otlp collector rejected {reason}", len(batch))
            if self._responses.rejected > 0:
                return Rejected("otlp collector rejected records", min(len(batch), self._responses.rejected))
        try:
            verdict: Final = TypeAdapter(LogRecordExportResult).validate_python(result)
        except Exception as error:  # noqa: BLE001  # invalid exporter verdicts must be retried, not kill the worker
            return Undelivered(f"otlp export verdict raised {type(error).__name__}")
        match verdict:
            case LogRecordExportResult.SUCCESS:
                return Delivered(len(batch))
            case LogRecordExportResult.FAILURE:
                return Undelivered("otlp export failed")

    async def close(self) -> None:
        await asyncio.to_thread(self.exporter.shutdown)
