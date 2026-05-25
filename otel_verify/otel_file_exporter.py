"""Custom OTEL logger that writes every finished span as one compact JSON line.

Registered as a litellm callback (``otel_file_exporter.otel_logger``). Because it
subclasses litellm's ``OpenTelemetry`` integration, its ``__init__`` registers
itself as ``proxy_server.open_telemetry_logger`` (first-registered-wins), so the
proxy's SERVER-span lifecycle flows through it exactly like ``callbacks: ["otel"]``
would — but spans land in a file we can parse instead of pretty-printed stdout.
"""

import os
import threading

from opentelemetry.sdk.trace.export import SpanExporter, SpanExportResult

from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig

_SPAN_FILE = os.getenv("OTEL_SPAN_FILE", "/tmp/otel_spans.jsonl")
_lock = threading.Lock()


class FileSpanExporter(SpanExporter):
    def export(self, spans):
        with _lock, open(_SPAN_FILE, "a") as f:
            for span in spans:
                f.write(span.to_json(indent=None) + "\n")
        return SpanExportResult.SUCCESS

    def shutdown(self):
        return None

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        return True


# SimpleSpanProcessor is used for non-string exporters, so spans flush on .end().
otel_logger = OpenTelemetry(config=OpenTelemetryConfig(exporter=FileSpanExporter()))
