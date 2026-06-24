"""Per-request span fan-out to admin-resolved destinations.

Attached to the main ``TracerProvider`` so every span on it (the FastAPI server
span, the proxy's ``auth`` phase span, DB lookups, the post-call cost ledger)
ships to every per-tenant destination the admin assigned for this request, on
top of the provider's configured global exporters.

Spans emitted through the per-tenant ``TenantTracerCache`` clone providers (the
gen-AI LLM-call span and its MCP-tool sibling) reach tenant backends through
the clone's own exporters; the clone's provider has its own processor list and
does NOT carry this fan-out processor, so a span is exported once per backend.

Each backend often requires backend-specific Resource attributes (Arize rejects
spans missing ``model_id`` / ``arize.project.name``), so the fan-out wraps each
forwarded span with the destination's expected Resource before handing it to
the per-destination exporter.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

import os

from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import ExporterSpec
from litellm.integrations.otel.plumbing.context import request_destinations

if TYPE_CHECKING:
    from litellm.integrations.otel.model.destination import OtelDestination

# Bound on cached per-destination processors. One processor per
# ``(endpoint, sorted(headers))`` pair, so the working set is one entry per
# admin-resolved tenant credential -- a real-world deployment with hundreds of
# tenants stays well under this. Evicted entries are dropped (not shut down; see
# the eviction site) and reclaimed at process exit.
_MAX_CACHED_PROCESSORS = 256


def _processor_key(destination: "OtelDestination") -> tuple:
    return (destination.endpoint, tuple(sorted(destination.headers.items())))


class TenantFanOutSpanProcessor(SpanProcessor):
    """Forward each finished span to every admin-resolved per-tenant destination.

    The destinations are looked up from a request-scoped contextvar set during
    auth, so the processor is stateless across requests and isolation across
    concurrent requests is guaranteed by Python's contextvars.
    """

    def __init__(self, owner_callback_name: str | None) -> None:
        self._owner = owner_callback_name
        # Built lazily so we avoid importing the providers module at class
        # definition time (which would create a circular import with routing).
        self._processors: "OrderedDict[tuple, SpanProcessor]" = OrderedDict()

    def on_start(self, span: "Span", parent_context: Context | None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        destinations = request_destinations()
        if not destinations:
            return
        # The gen-AI LLM-call span (and the MCP tool-call sibling) is already
        # routed to per-tenant destinations by the per-backend v2 logger via
        # ``TenantTracerCache`` -- the logger picks the right attribute mapper
        # (OpenInference for arize, GenAI semconv for langfuse_otel) and ships
        # through the clone provider's appended exporter. Forwarding it here too
        # would deliver a SECOND copy with the wrong vocabulary and a fresh
        # span_id, surfacing in the destination as an orphaned duplicate. Skip.
        if _is_genai_span(span):
            return
        # Proxy-internal spans (FastAPI server, ``auth`` phase, postgres lookups,
        # post-call cost ledger) are generic OTel semantic-convention spans with
        # no backend-specific vocabulary, so they ship to EVERY admin-resolved
        # destination this request fans out to, regardless of the destination's
        # ``callback_name``. The owner discriminator only matters for the gen-AI
        # span (handled by the skip above).
        for destination in destinations:
            processor = self._processor_for(destination)
            if processor is None:
                continue
            try:
                processor.on_end(_with_destination_resource(span, destination))
            except Exception as exc:
                verbose_logger.debug(
                    "OTel V2 fan-out: forwarding span to %s failed: %s",
                    destination.endpoint,
                    exc,
                )

    def shutdown(self) -> None:
        for processor in self._processors.values():
            try:
                processor.shutdown()
            except Exception as exc:
                verbose_logger.debug(
                    "OTel V2 fan-out: processor shutdown failed: %s", exc
                )
        self._processors.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        all_ok = True
        for processor in self._processors.values():
            try:
                if not processor.force_flush(timeout_millis):
                    all_ok = False
            except Exception:
                all_ok = False
        return all_ok

    def _processor_for(self, destination: "OtelDestination") -> "SpanProcessor | None":
        key = _processor_key(destination)
        cached = self._processors.get(key)
        if cached is not None:
            self._processors.move_to_end(key)
            return cached
        from litellm.integrations.otel.plumbing.providers import (
            _exporter_from_spec,
            _processor_for as _build_processor,
            default_otlp_kind_for_backend,
        )

        try:
            spec = ExporterSpec(
                kind=default_otlp_kind_for_backend(destination.callback_name),
                endpoint=destination.endpoint,
                headers=destination.header_string(),
                owner=None,
            )
            exporter = _exporter_from_spec(spec)
            processor = _build_processor(exporter, use_simple=False)
        except Exception as exc:
            verbose_logger.debug(
                "OTel V2 fan-out: failed to build processor for %s: %s",
                destination.endpoint,
                exc,
            )
            return None
        self._processors[key] = processor
        if len(self._processors) > _MAX_CACHED_PROCESSORS:
            # Evict the LRU entry but do NOT shut it down here: a
            # ``BatchSpanProcessor`` may still hold spans queued on its exporter
            # thread, and calling ``shutdown`` synchronously can drop or raise on
            # those in-flight spans. Dropping the reference lets the worker drain
            # naturally and be reclaimed at process exit. The cache is bounded at
            # ``_MAX_CACHED_PROCESSORS``, so the un-shut-down working set stays
            # bounded.
            self._processors.popitem(last=False)
        return processor


# Attribute set on every gen-AI LLM-call span by the v2 emitter. Used as the
# unambiguous skip signal: only the LLM-call span carries this, and the
# per-backend v2 logger already routes it to per-tenant destinations through
# the TenantTracerCache clone provider's appended exporter.
_GENAI_SPAN_ATTR = "gen_ai.operation.name"


def _is_genai_span(span: ReadableSpan) -> bool:
    attributes = span.attributes or {}
    return _GENAI_SPAN_ATTR in attributes


# Backend-specific Resource attributes required by the destination. Arize
# rejects spans missing ``model_id`` (or the alternative ``arize.project.name``
# span attribute); other backends accept the proxy's default Resource.
def _destination_resource_attrs(destination: "OtelDestination") -> dict[str, str]:
    if destination.callback_name == "arize":
        project = os.environ.get("ARIZE_PROJECT_NAME")
        if project:
            return {"model_id": project, "arize.project.name": project}
    return {}


def _with_destination_resource(
    span: ReadableSpan, destination: "OtelDestination"
) -> ReadableSpan:
    """Return ``span`` with its Resource augmented by the destination's required
    attributes. The original span object is left untouched; a shallow wrapper
    reuses every other field and only swaps the ``resource`` property."""
    extra = _destination_resource_attrs(destination)
    if not extra:
        return span
    merged = Resource.create({**dict(span.resource.attributes), **extra})
    return _ResourceWrappedReadableSpan(span, merged)


class _ResourceWrappedReadableSpan(ReadableSpan):
    """A ``ReadableSpan`` view whose ``resource`` is overridden.

    The OTLP exporter reads each span's ``resource`` when serializing; for
    backend-specific attributes (Arize's ``model_id``) we substitute the
    destination's expected Resource without mutating the underlying span.
    """

    def __init__(self, inner: ReadableSpan, resource: Resource) -> None:
        super().__init__(
            name=inner.name,
            context=inner.context,
            parent=inner.parent,
            resource=resource,
            attributes=inner.attributes,
            events=inner.events,
            links=inner.links,
            kind=inner.kind,
            status=inner.status,
            start_time=inner.start_time,
            end_time=inner.end_time,
            instrumentation_scope=inner.instrumentation_scope,
        )
