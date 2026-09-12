"""Multi-backend fan-out: one TracerProvider, *N* SpanProcessors.

V1 needed a separate ``TracerProvider`` per integration to avoid stepping on
the global. V2 attaches a ``SpanProcessor`` per exporter to the *same*
provider, so the same trace ID lights up every backend — no duplicate spans,
no per-integration provider caches.
"""

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)

from litellm.integrations.otel.model.config import ExporterOwner, ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.plumbing.providers import build_tracer_provider, register_exporter_factory


def test_two_exporters_receive_the_same_span():
    """A single ``span.end()`` lands in BOTH exporters with the same span ID."""
    exporter_a = InMemorySpanExporter()
    exporter_b = InMemorySpanExporter()
    cfg = OpenTelemetryV2Config(
        exporters=[
            ExporterSpec(kind="in_memory"),
            ExporterSpec(kind="in_memory"),
        ]
    )
    # Override the auto-built exporters with our test ones by swapping
    # processors after construction (the test's purpose is to exercise the
    # multi-processor wiring, not to negotiate the in-memory pipe).
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    provider = build_tracer_provider(cfg)
    # Clear out any auto-built export processors and attach our pair.
    while provider._active_span_processor._span_processors:
        provider._active_span_processor._span_processors = (
            provider._active_span_processor._span_processors[:-1]
        )
    provider.add_span_processor(SimpleSpanProcessor(exporter_a))
    provider.add_span_processor(SimpleSpanProcessor(exporter_b))

    tracer = provider.get_tracer("test")
    span = tracer.start_span("multi-backend")
    span.set_attribute("test.marker", "yes")
    span.end()

    spans_a = exporter_a.get_finished_spans()
    spans_b = exporter_b.get_finished_spans()
    assert len(spans_a) == 1
    assert len(spans_b) == 1
    assert spans_a[0].context.span_id == spans_b[0].context.span_id


def _capturing_kind(kind):
    exporter = InMemorySpanExporter()
    register_exporter_factory(kind, lambda _spec: exporter)
    return exporter


def _emit_langfuse_stamped_span(provider):
    span = provider.get_tracer("test").start_span("chat proof-model")
    span.set_attribute("gen_ai.request.model", "proof-model")
    span.set_attribute("langfuse.trace.name", "private-langfuse-only-name")
    span.set_attribute("langfuse.observation.input", '[{"role":"user","content":"hi"}]')
    span.end()
    return span.context.span_id


def test_generic_exporter_does_not_receive_langfuse_attributes_the_langfuse_exporter_keeps():
    generic = _capturing_kind("capture_generic_collector")
    langfuse = _capturing_kind("capture_langfuse_sink")
    cfg = OpenTelemetryV2Config(
        exporters=[
            ExporterSpec(kind="capture_generic_collector"),
            ExporterSpec(kind="capture_langfuse_sink", owner=ExporterOwner.LANGFUSE_OTEL),
        ]
    )
    span_id = _emit_langfuse_stamped_span(build_tracer_provider(cfg, tenant_overrides=True))

    (generic_span,) = generic.get_finished_spans()
    (langfuse_span,) = langfuse.get_finished_spans()
    assert generic_span.context.span_id == span_id == langfuse_span.context.span_id
    assert generic_span.attributes["gen_ai.request.model"] == "proof-model"
    assert [k for k in generic_span.attributes if k.startswith("langfuse.")] == []
    assert langfuse_span.attributes["langfuse.trace.name"] == "private-langfuse-only-name"
    assert langfuse_span.attributes["langfuse.observation.input"] == '[{"role":"user","content":"hi"}]'


def test_langfuse_mapper_on_a_plain_otlp_exporter_keeps_langfuse_attributes():
    sink = _capturing_kind("capture_langfuse_mapper_sink")
    cfg = OpenTelemetryV2Config(
        mapper_names=["langfuse"], exporters=[ExporterSpec(kind="capture_langfuse_mapper_sink")]
    )
    _emit_langfuse_stamped_span(build_tracer_provider(cfg))

    (span,) = sink.get_finished_spans()
    assert span.attributes["langfuse.trace.name"] == "private-langfuse-only-name"


def test_resource_attributes_apply_to_all_exporters():
    """``resource_attributes`` flow through the shared TracerProvider."""
    cfg = OpenTelemetryV2Config(
        exporters=[ExporterSpec(kind="in_memory")],
        resource_attributes={"openinference.project.name": "phoenix-test"},
    )
    provider = build_tracer_provider(cfg)
    assert provider.resource.attributes["openinference.project.name"] == "phoenix-test"


def test_config_normalizer_inserts_genai_first():
    """The validator pins ``genai`` at the head + appends ``legacy`` on legacy_compat."""
    cfg = OpenTelemetryV2Config(mapper_names=["openinference", "langfuse"])
    assert cfg.mapper_names[0] == "genai"
    assert "openinference" in cfg.mapper_names
    assert "langfuse" in cfg.mapper_names
    assert cfg.mapper_names[-1] == "legacy"  # legacy_compat=True by default


def test_config_normalizer_no_legacy_when_compat_off():
    cfg = OpenTelemetryV2Config(legacy_compat=False, mapper_names=["openinference"])
    assert "legacy" not in cfg.mapper_names
    assert cfg.mapper_names[0] == "genai"


def test_config_folds_legacy_exporter_triple_into_exporters_list():
    """When ``exporters`` is empty, the validator folds the legacy single triple."""
    cfg = OpenTelemetryV2Config(
        exporter="otlp_http", endpoint="https://api.example.com", headers="k=v"
    )
    assert len(cfg.exporters) == 1
    assert cfg.exporters[0].kind == "otlp_http"
    assert cfg.exporters[0].endpoint == "https://api.example.com"
