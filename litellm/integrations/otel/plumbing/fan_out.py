"""Per-request span fan-out to admin-resolved destinations.

Attached to the main ``TracerProvider`` so every span on it (the FastAPI server
span, the proxy's ``auth`` phase span, DB lookups, the post-call cost ledger)
ships to every per-tenant destination the admin assigned for this request, on
top of the provider's configured global exporters.

Spans emitted through the per-tenant ``TenantTracerCache`` clone providers (the
gen-AI LLM-call span and its MCP-tool sibling) reach tenant backends through
the clone's own exporters; the clone's provider has its own processor list and
does NOT carry this fan-out processor, so a span is exported once per backend.

Identical ``(trace_id, span_id)`` deduplication on the OTLP receiver collapses
the rare overlap into one span per backend (e.g. when the configured global
exporter and a per-tenant destination point at the same vendor account).
"""

from __future__ import annotations

from collections import OrderedDict
from typing import TYPE_CHECKING

from opentelemetry.context import Context
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import ExporterSpec
from litellm.integrations.otel.plumbing.context import request_destinations

if TYPE_CHECKING:
    from litellm.integrations.otel.model.destination import OtelDestination

# Bound on cached per-destination processors. One processor per
# ``(endpoint, sorted(headers))`` pair, so the working set is one entry per
# admin-resolved tenant credential -- a real-world deployment with hundreds of
# tenants stays well under this. The LRU shuts down the evicted processor's
# exporter thread so the working set is reclaimed.
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
        for destination in destinations:
            if destination.callback_name != self._owner:
                continue
            processor = self._processor_for(destination)
            if processor is None:
                continue
            try:
                processor.on_end(span)
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
                verbose_logger.debug("OTel V2 fan-out: processor shutdown failed: %s", exc)
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
            _processor_for,
        )

        try:
            spec = ExporterSpec(
                kind=_resolve_kind(destination),
                endpoint=destination.endpoint,
                headers=destination.header_string(),
                owner=None,
            )
            exporter = _exporter_from_spec(spec)
            processor = _processor_for(exporter, use_simple=False)
        except Exception as exc:
            verbose_logger.debug(
                "OTel V2 fan-out: failed to build processor for %s: %s",
                destination.endpoint,
                exc,
            )
            return None
        self._processors[key] = processor
        if len(self._processors) > _MAX_CACHED_PROCESSORS:
            _, evicted = self._processors.popitem(last=False)
            try:
                evicted.shutdown()
            except Exception:
                pass
        return processor


# Per-backend transport: Arize speaks OTLP/gRPC, every other current preset
# speaks OTLP/HTTP. Kept here so the fan-out picks the same transport the
# preset's own exporter uses, mirroring ``TenantTracerCache._owned_otlp_kind``.
_GRPC_BACKENDS = frozenset({"arize"})


def _resolve_kind(destination: "OtelDestination") -> str:
    return "otlp_grpc" if destination.callback_name in _GRPC_BACKENDS else "otlp_http"
