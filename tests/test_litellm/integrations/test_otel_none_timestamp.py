"""Regression tests for the OTEL service hooks crashing on None start/end times.

When a service success/failure event is logged without timing information,
``start_time``/``end_time`` arrive as ``None`` and were passed straight into
``_to_ns``, which called ``.timestamp()`` on ``None`` and raised
``AttributeError``. See https://github.com/BerriAI/litellm/issues/30061.
"""

from datetime import datetime
from typing import Optional

import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from litellm.integrations.opentelemetry import OpenTelemetry
from litellm.types.services import ServiceLoggerPayload, ServiceTypes


def _make_otel():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))

    otel = OpenTelemetry()
    otel.tracer = provider.get_tracer(__name__)
    return otel, exporter


def _payload(is_error: bool = False, error: Optional[str] = None):
    return ServiceLoggerPayload(
        is_error=is_error,
        error=error,
        service=ServiceTypes.DB,
        duration=0.0,
        call_type="query",
        event_metadata=None,
    )


def test_to_ns_handles_none():
    """_to_ns should fall back to now() instead of crashing on None."""
    otel, _ = _make_otel()
    assert otel._to_ns(None) > 0

    fixed = datetime(2024, 1, 1)
    assert otel._to_ns(fixed) == int(fixed.timestamp() * 1e9)


@pytest.mark.asyncio
async def test_async_service_success_hook_with_none_times():
    """Success hook must not raise when start_time/end_time are None."""
    otel, exporter = _make_otel()
    parent = otel.tracer.start_span("parent")

    await otel.async_service_success_hook(
        payload=_payload(),
        parent_otel_span=parent,
        start_time=None,
        end_time=None,
    )

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    assert all(span.start_time > 0 for span in spans)


@pytest.mark.asyncio
async def test_async_service_failure_hook_with_none_times():
    """Failure hook must not raise when start_time/end_time are None."""
    otel, exporter = _make_otel()
    parent = otel.tracer.start_span("parent")

    await otel.async_service_failure_hook(
        payload=_payload(is_error=True, error="boom"),
        error="boom",
        parent_otel_span=parent,
        start_time=None,
        end_time=None,
    )

    spans = exporter.get_finished_spans()
    assert len(spans) >= 1
    assert all(span.start_time > 0 for span in spans)
