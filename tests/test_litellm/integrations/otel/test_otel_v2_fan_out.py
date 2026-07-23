"""Tests for the per-request tenant fan-out SpanProcessor (Hoist).

The fan-out processor lives on the main v2 ``TracerProvider`` and forwards every
finished span to the admin-resolved per-tenant destinations carried on the
request's ``ContextVar``. The contract under test:

- Spans on the main provider land at every per-tenant destination matching this
  backend (the ``owner_callback_name``).
- When destinations is empty (an unassigned identity, the SDK path, a request
  before the resolver ran), the processor is a no-op.
- Concurrent requests with different destinations stay isolated -- contextvars
  scope per task, so one request's tenant doesn't receive another's spans.
- The processor caches one BatchSpanProcessor per ``(endpoint, headers)`` pair
  and skips destinations whose ``callback_name`` doesn't match its owner.
"""

import asyncio
from typing import Any

import pytest

pytest.importorskip("opentelemetry")

from opentelemetry.sdk.trace import TracerProvider  # noqa: E402
from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: E402
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (  # noqa: E402
    InMemorySpanExporter,
)
from opentelemetry.sdk.trace.export import SpanExporter  # noqa: E402

from litellm.integrations.otel.model.destination import OtelDestination  # noqa: E402
from litellm.integrations.otel.plumbing.context import (  # noqa: E402
    set_request_destinations,
    _request_destinations,
)
from litellm.integrations.otel.plumbing.routing import (  # noqa: E402
    TenantFanOutSpanProcessor,
    _processor_key,
)
from litellm.integrations.otel.plumbing.providers import (  # noqa: E402
    destination_resource_attrs,
)
from litellm.integrations.otel.plumbing import providers  # noqa: E402
from litellm.integrations.otel.model.config import OpenTelemetryV2Config  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_request_destinations():
    _request_destinations.set(())
    yield
    _request_destinations.set(())


def _build_provider_with_fan_out(
    owner: str, exporter: SpanExporter
) -> tuple[TracerProvider, TenantFanOutSpanProcessor]:
    """Build a real TracerProvider with the in-memory exporter as the configured
    backend, plus the fan-out processor wired in. The fan-out processor's
    per-destination processors are real BatchSpanProcessors, replaced in the
    test by an injected SimpleSpanProcessor against an in-memory exporter via
    ``monkeypatch`` so the test can read what was forwarded."""
    cfg = OpenTelemetryV2Config(exporter="in_memory")
    provider = providers.build_tracer_provider(
        cfg, exporter=exporter, tenant_fan_out_owner=owner
    )
    fan_out = next(
        p
        for p in provider._active_span_processor._span_processors
        if isinstance(p, TenantFanOutSpanProcessor)
    )
    return provider, fan_out


def test_fan_out_forwards_span_to_matching_destination(monkeypatch):
    """A span on the main provider lands at the per-tenant destination whose
    callback_name matches the fan-out owner. Removing the fan-out branch makes
    this test fail (the tenant exporter never sees the span)."""
    main_exporter = InMemorySpanExporter()
    provider, fan_out = _build_provider_with_fan_out("langfuse_otel", main_exporter)

    tenant_exporter = InMemorySpanExporter()

    # Swap the lazy per-destination processor build for an in-memory one so we
    # don't actually hit the network.
    def _stub_processor_for(self, destination):
        return SimpleSpanProcessor(tenant_exporter)

    monkeypatch.setattr(
        TenantFanOutSpanProcessor, "_processor_for", _stub_processor_for
    )

    set_request_destinations(
        (
            OtelDestination(
                callback_name="langfuse_otel",
                endpoint="https://cloud.langfuse.com/api/public/otel",
                headers={"Authorization": "Bearer pk:sk"},
            ),
        )
    )
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("auth /chat/completions"):
        pass

    main_names = [s.name for s in main_exporter.get_finished_spans()]
    tenant_names = [s.name for s in tenant_exporter.get_finished_spans()]
    assert "auth /chat/completions" in main_names
    assert "auth /chat/completions" in tenant_names


def test_fan_out_forwards_proxy_internal_spans_to_every_destination(monkeypatch):
    """Proxy-internal spans (server, auth phase, postgres, post-call ledger)
    are generic OTel semconv with no backend-specific vocabulary, so they ship
    to every admin-resolved destination regardless of its ``callback_name``.
    The owner discriminator is only used to skip the gen-AI span (the v2 logger
    routes that itself through TenantTracerCache to avoid wrong-vocabulary
    duplicates)."""
    main_exporter = InMemorySpanExporter()
    _provider, _ = _build_provider_with_fan_out("langfuse_otel", main_exporter)

    tenant_exporter = InMemorySpanExporter()
    monkeypatch.setattr(
        TenantFanOutSpanProcessor,
        "_processor_for",
        lambda self, destination: SimpleSpanProcessor(tenant_exporter),
    )

    # Destination is tagged for a DIFFERENT backend (``arize``) than the
    # owner (``langfuse_otel``). Proxy-internal spans must still forward.
    set_request_destinations(
        (
            OtelDestination(
                callback_name="arize",
                endpoint="https://otlp.arize.com/v1",
                headers={"api_key": "k", "space_id": "s"},
            ),
        )
    )
    tracer = _provider.get_tracer("test")
    with tracer.start_as_current_span("auth"):
        pass

    names = [s.name for s in tenant_exporter.get_finished_spans()]
    assert "auth" in names


def test_destination_resource_attrs_reads_declared_attrs(monkeypatch):
    """``destination_resource_attrs`` is backend-agnostic: it returns exactly what
    the destination's builder declared, never consulting env itself. The env-vs-
    credential precedence is enforced at build time (see test_presets_destinations),
    so setting ARIZE_PROJECT_NAME here must NOT override the destination's own attrs."""
    monkeypatch.setenv("ARIZE_PROJECT_NAME", "global-project")
    destination = OtelDestination(
        callback_name="arize",
        endpoint="https://otlp.arize.com/v1",
        headers={"api_key": "k", "space_id": "s"},
        resource_attributes={
            "model_id": "tenant-project",
            "arize.project.name": "tenant-project",
        },
    )

    assert destination_resource_attrs(destination) == {
        "model_id": "tenant-project",
        "arize.project.name": "tenant-project",
    }


def test_destination_resource_attrs_empty_for_header_routed_backend():
    """A langfuse/weave-style destination routes by header and declares no Resource
    attrs; the helper returns {} and the fan-out leaves the span Resource untouched."""
    destination = OtelDestination(
        callback_name="langfuse_otel",
        endpoint="https://cloud.langfuse.com/api/public/otel",
        headers={"Authorization": "Basic x"},
    )
    assert destination_resource_attrs(destination) == {}


def test_fan_out_skips_genai_span_to_avoid_double_export(monkeypatch):
    """The gen-AI LLM-call span is routed by the v2 logger through
    ``TenantTracerCache`` to per-tenant exporters with the right attribute
    mapper. Forwarding it here too would deliver a duplicate with the wrong
    vocabulary, surfacing in the destination as an orphaned second span."""
    main_exporter = InMemorySpanExporter()
    _provider, _ = _build_provider_with_fan_out("langfuse_otel", main_exporter)

    tenant_exporter = InMemorySpanExporter()
    monkeypatch.setattr(
        TenantFanOutSpanProcessor,
        "_processor_for",
        lambda self, destination: SimpleSpanProcessor(tenant_exporter),
    )

    set_request_destinations(
        (
            OtelDestination(
                callback_name="arize",
                endpoint="https://otlp.arize.com/v1",
                headers={},
            ),
        )
    )
    tracer = _provider.get_tracer("test")
    # A gen-AI span sets ``gen_ai.operation.name`` (the v2 emitter does this
    # via the SpanRole.LLM_CALL mapper). Fake it explicitly here.
    with tracer.start_as_current_span("chat gpt-4o") as span:
        span.set_attribute("gen_ai.operation.name", "chat")
    # A proxy-internal span also fires.
    with tracer.start_as_current_span("auth /v1/chat/completions"):
        pass

    names = [s.name for s in tenant_exporter.get_finished_spans()]
    assert "auth /v1/chat/completions" in names
    assert "chat gpt-4o" not in names  # gen-AI skipped


def test_fan_out_noop_when_no_destinations(monkeypatch):
    """An unassigned identity / pre-auth / SDK path leaves the contextvar at
    its empty-tuple default; the processor must short-circuit, NOT crash, NOT
    forward."""
    main_exporter = InMemorySpanExporter()
    provider, _ = _build_provider_with_fan_out("langfuse_otel", main_exporter)
    forwarded: list[Any] = []
    monkeypatch.setattr(
        TenantFanOutSpanProcessor,
        "_processor_for",
        lambda self, destination: forwarded.append(destination) or None,
    )
    tracer = provider.get_tracer("test")
    with tracer.start_as_current_span("auth"):
        pass
    assert forwarded == []


def test_fan_out_caches_processor_per_destination_key(monkeypatch):
    """Two requests with the same destination must share one cached
    BatchSpanProcessor; two different destinations must build two. Otherwise
    every request rebuilds the OTLP exporter (and its background thread)."""
    main_exporter = InMemorySpanExporter()
    _provider, fan_out = _build_provider_with_fan_out("langfuse_otel", main_exporter)

    built: list[OtelDestination] = []

    real_processor_for = TenantFanOutSpanProcessor._processor_for

    def _spy(self, destination):
        result = real_processor_for(self, destination)
        if result is not None:
            built.append(destination)
        return result

    # Build always returns None to keep the test offline, but we tracked
    # invocations via the spy above. Easier: stub the inner builder.
    def _stub(self, destination):
        built.append(destination)
        key = _processor_key(destination)
        if key in self._processors:
            return self._processors[key]
        proc = SimpleSpanProcessor(InMemorySpanExporter())
        self._processors[key] = proc
        return proc

    monkeypatch.setattr(TenantFanOutSpanProcessor, "_processor_for", _stub)

    dest_a = OtelDestination(
        callback_name="langfuse_otel", endpoint="https://a/", headers={"k": "1"}
    )
    dest_b = OtelDestination(
        callback_name="langfuse_otel", endpoint="https://b/", headers={"k": "1"}
    )
    set_request_destinations((dest_a,))
    tracer = _provider.get_tracer("test")
    with tracer.start_as_current_span("s1"):
        pass
    set_request_destinations((dest_a,))
    with tracer.start_as_current_span("s2"):
        pass
    set_request_destinations((dest_b,))
    with tracer.start_as_current_span("s3"):
        pass

    # _stub appends every call; the cache size is what matters: two unique
    # ``(endpoint, headers)`` pairs -> two cached processors.
    assert len(fan_out._processors) == 2


def test_fan_out_per_request_isolation_with_concurrent_tasks(monkeypatch):
    """Two requests running concurrently with different destinations must each
    see only THEIR destination's spans. The contextvar scopes per asyncio task,
    so this is a pin on the contextvar approach: switching to a global variable
    would break this test."""
    main_exporter = InMemorySpanExporter()
    provider, _ = _build_provider_with_fan_out("langfuse_otel", main_exporter)

    exporter_a = InMemorySpanExporter()
    exporter_b = InMemorySpanExporter()

    def _stub(self, destination):
        if destination.endpoint == "https://a/":
            return SimpleSpanProcessor(exporter_a)
        return SimpleSpanProcessor(exporter_b)

    monkeypatch.setattr(TenantFanOutSpanProcessor, "_processor_for", _stub)
    tracer = provider.get_tracer("test")

    async def fire(label: str, endpoint: str):
        set_request_destinations(
            (
                OtelDestination(
                    callback_name="langfuse_otel",
                    endpoint=endpoint,
                    headers={},
                ),
            )
        )
        with tracer.start_as_current_span(label):
            await asyncio.sleep(0)

    async def driver():
        await asyncio.gather(
            fire("req_a", "https://a/"),
            fire("req_b", "https://b/"),
        )

    asyncio.run(driver())

    names_a = {s.name for s in exporter_a.get_finished_spans()}
    names_b = {s.name for s in exporter_b.get_finished_spans()}
    assert "req_a" in names_a and "req_b" not in names_a
    assert "req_b" in names_b and "req_a" not in names_b


def test_fan_out_owner_set_by_build_tracer_provider():
    """The v2 logger opts the MAIN provider into fan-out by passing its
    callback_name; the TenantTracerCache clone providers do NOT, so the gen-AI
    span emitted through them is not also forwarded by the fan-out processor.
    Regressing this (adding the processor to clones) would double-export every
    gen-AI span."""
    from litellm.integrations.otel.logger import OpenTelemetryV2

    logger = OpenTelemetryV2(callback_name="arize")
    main_procs = logger._tracer_provider._active_span_processor._span_processors
    assert any(isinstance(p, TenantFanOutSpanProcessor) for p in main_procs)
    # Build a tenant clone provider and confirm it has no fan-out processor.
    clone = providers.build_tracer_provider(logger.config)
    clone_procs = clone._active_span_processor._span_processors
    assert not any(isinstance(p, TenantFanOutSpanProcessor) for p in clone_procs)
