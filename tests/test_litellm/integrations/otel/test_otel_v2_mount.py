"""V2 entrypoint: the FastAPI instrumentation pattern proxy_server mounts at
startup (gated by LITELLM_OTEL_V2). The mount logic itself lives inline in
``proxy_server.proxy_startup_event``; this exercises the same pattern in
isolation so the server-span + shared-provider behavior stays covered.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath("../../../.."))

pytest.importorskip("opentelemetry")
pytest.importorskip("opentelemetry.instrumentation.fastapi")
fastapi = pytest.importorskip("fastapi")

from fastapi.testclient import TestClient  # noqa: E402
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.trace import SpanKind  # noqa: E402

from litellm.integrations.otel.config import (  # noqa: E402
    OpenTelemetryV2Config,
    is_otel_v2_enabled,
)
from litellm.integrations.otel.logger import OpenTelemetryV2  # noqa: E402


def _instrumented_app():
    """Mirror proxy_server's startup mount: a logger builds the shared provider,
    and the FastAPI instrumentor is attached to it."""
    app = fastapi.FastAPI()

    @app.get("/ping")
    def ping():
        return {"ok": True}

    logger = OpenTelemetryV2(config=OpenTelemetryV2Config(exporter="in_memory"))
    FastAPIInstrumentor.instrument_app(app, tracer_provider=logger._tracer_provider)
    return app, logger


def test_gate_toggles_with_env(monkeypatch):
    """The startup mount is guarded by this flag."""
    monkeypatch.delenv("LITELLM_OTEL_V2", raising=False)
    assert is_otel_v2_enabled() is False
    monkeypatch.setenv("LITELLM_OTEL_V2", "1")
    assert is_otel_v2_enabled() is True


def test_instrumented_app_emits_server_span():
    app, logger = _instrumented_app()
    exporter = InMemorySpanExporter()
    logger._tracer_provider.add_span_processor(SimpleSpanProcessor(exporter))

    TestClient(app).get("/ping")

    server_spans = [
        s for s in exporter.get_finished_spans() if s.kind is SpanKind.SERVER
    ]
    assert server_spans, "FastAPI instrumentor should emit a SERVER span per request"
    attrs = server_spans[0].attributes or {}
    assert any("route" in k or "method" in k for k in attrs)


def test_logger_and_instrumentor_share_provider():
    """Gen-ai spans (logger) and server spans (instrumentor) write to one provider."""
    _, logger = _instrumented_app()
    assert logger._emitter._tracer is logger.tracer
