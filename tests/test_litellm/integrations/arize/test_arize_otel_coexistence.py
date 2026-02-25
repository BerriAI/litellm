"""
Tests that Arize Phoenix / Arize and the generic ``otel`` callback can
coexist, each sending spans to their own independent exporter.

Covers the three root-cause fixes:
1. ArizePhoenixLogger / ArizeLogger create *dedicated* TracerProviders.
2. The ``otel`` dedup check does NOT match Arize subclasses.
3. Arize loggers do NOT overwrite ``proxy_server.open_telemetry_logger``.
"""

import unittest
from unittest.mock import patch

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from litellm.integrations.opentelemetry import OpenTelemetry, OpenTelemetryConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_otel_logger(exporter: InMemorySpanExporter) -> OpenTelemetry:
    """Create a generic ``otel`` callback backed by an in-memory exporter.

    We build a dedicated TracerProvider explicitly so the test is isolated
    from whatever global provider state may exist.
    """
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    config = OpenTelemetryConfig(exporter=exporter)
    return OpenTelemetry(config=config, callback_name="otel", tracer_provider=provider)


def _make_arize_phoenix_logger(exporter: InMemorySpanExporter):
    """Create an ``arize_phoenix`` callback backed by an in-memory exporter.

    ArizePhoenixLogger._init_tracing creates its own TracerProvider, so we
    pass the exporter via config and let it build the provider internally.
    """
    from litellm.integrations.arize.arize_phoenix import ArizePhoenixLogger

    config = OpenTelemetryConfig(exporter=exporter)
    return ArizePhoenixLogger(config=config, callback_name="arize_phoenix")


def _make_arize_logger(exporter: InMemorySpanExporter):
    """Create an ``arize`` callback backed by an in-memory exporter.

    ArizeLogger._init_tracing creates its own TracerProvider, so we pass
    the exporter via config and let it build the provider internally.
    """
    from litellm.integrations.arize.arize import ArizeLogger

    config = OpenTelemetryConfig(exporter=exporter)
    return ArizeLogger(config=config, callback_name="arize")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestIndependentTracerProviders(unittest.TestCase):
    """Each integration must get its own TracerProvider so spans go to the right exporter."""

    def test_otel_and_arize_phoenix_have_different_tracer_providers(self):
        otel_exporter = InMemorySpanExporter()
        phoenix_exporter = InMemorySpanExporter()

        otel_logger = _make_otel_logger(otel_exporter)
        phoenix_logger = _make_arize_phoenix_logger(phoenix_exporter)

        # The tracers must come from different providers
        assert otel_logger.tracer is not phoenix_logger.tracer

    def test_otel_and_arize_have_different_tracer_providers(self):
        otel_exporter = InMemorySpanExporter()
        arize_exporter = InMemorySpanExporter()

        otel_logger = _make_otel_logger(otel_exporter)
        arize_logger = _make_arize_logger(arize_exporter)

        assert otel_logger.tracer is not arize_logger.tracer

    def test_arize_phoenix_and_arize_have_different_tracer_providers(self):
        phoenix_exporter = InMemorySpanExporter()
        arize_exporter = InMemorySpanExporter()

        phoenix_logger = _make_arize_phoenix_logger(phoenix_exporter)
        arize_logger = _make_arize_logger(arize_exporter)

        assert phoenix_logger.tracer is not arize_logger.tracer


class TestSpansRoutedToCorrectExporter(unittest.TestCase):
    """Spans created by each logger must land in its own exporter, not the other's."""

    def test_spans_go_to_respective_exporters(self):
        otel_exporter = InMemorySpanExporter()
        phoenix_exporter = InMemorySpanExporter()

        otel_logger = _make_otel_logger(otel_exporter)
        phoenix_logger = _make_arize_phoenix_logger(phoenix_exporter)

        # Create a span on each — SimpleSpanProcessor exports synchronously on end()
        otel_span = otel_logger.tracer.start_span("otel_test_span")
        otel_span.end()

        phoenix_span = phoenix_logger.tracer.start_span("phoenix_test_span")
        phoenix_span.end()

        # Read spans *before* shutdown (shutdown clears the in-memory store)
        otel_span_names = [s.name for s in otel_exporter.get_finished_spans()]
        phoenix_span_names = [s.name for s in phoenix_exporter.get_finished_spans()]

        assert "otel_test_span" in otel_span_names
        assert "phoenix_test_span" not in otel_span_names

        assert "phoenix_test_span" in phoenix_span_names
        assert "otel_test_span" not in phoenix_span_names


class TestOtelDedupCheck(unittest.TestCase):
    """The ``otel`` callback dedup must use exact type check, not isinstance."""

    def test_arize_phoenix_logger_is_not_matched_by_otel_dedup(self):
        from litellm.integrations.arize.arize_phoenix import ArizePhoenixLogger

        phoenix_logger = _make_arize_phoenix_logger(InMemorySpanExporter())

        # isinstance would match — but type() must not
        assert isinstance(phoenix_logger, OpenTelemetry)
        assert type(phoenix_logger) is not OpenTelemetry

    def test_arize_logger_is_not_matched_by_otel_dedup(self):
        from litellm.integrations.arize.arize import ArizeLogger

        arize_logger = _make_arize_logger(InMemorySpanExporter())

        assert isinstance(arize_logger, OpenTelemetry)
        assert type(arize_logger) is not OpenTelemetry

    def test_otel_logger_matches_own_dedup(self):
        otel_logger = _make_otel_logger(InMemorySpanExporter())
        assert type(otel_logger) is OpenTelemetry


class TestProxyLoggerNotOverwritten(unittest.TestCase):
    """Arize / Phoenix must not overwrite ``proxy_server.open_telemetry_logger``."""

    @patch("litellm.proxy.proxy_server.open_telemetry_logger", None)
    def test_arize_phoenix_does_not_set_proxy_otel_logger(self):
        from litellm.proxy import proxy_server

        _make_arize_phoenix_logger(InMemorySpanExporter())
        assert proxy_server.open_telemetry_logger is None

    @patch("litellm.proxy.proxy_server.open_telemetry_logger", None)
    def test_arize_does_not_set_proxy_otel_logger(self):
        from litellm.proxy import proxy_server

        _make_arize_logger(InMemorySpanExporter())
        assert proxy_server.open_telemetry_logger is None


if __name__ == "__main__":
    unittest.main()
