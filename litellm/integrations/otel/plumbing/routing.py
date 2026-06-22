"""Per-request multi-tenant tracer routing with fan-out.

A call's identity chain is assigned a set of admin-owned OTEL destinations
(``LLMCallEvent.otel_destinations``, resolved server-side from named credentials).
Its spans must export to ALL of them plus the configured/global exporter, so
``TenantTracerCache`` builds and caches one ``TracerProvider`` per distinct
destination SET -- the provider keeps the configured exporters and appends one
``SpanProcessor`` per destination, so a span is emitted once and copied to each
(no duplicate spans). With no destinations it hands back the logger's default
tracer (global only). Destinations are never request-derived, so a caller can
neither redirect a trace nor spawn providers.
"""

from collections import OrderedDict

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.integrations.otel.plumbing.providers import (
    build_tracer_provider,
    get_tracer,
)

# Exporter kinds that ignore endpoint/headers — never rewritten with a destination.
_NON_OTLP_KINDS = ("console", "in_memory", "inmemory", "memory")

# Cap on distinct destination-scoped providers held at once. Destinations are
# admin-owned (one per key/team), so this is resource hygiene rather than an
# anti-abuse bound: it keeps the working set of active tenants resident while
# flushing and shutting down evicted providers so their exporter threads are
# reclaimed.
_MAX_CACHED_PROVIDERS = 256


def _shutdown_provider(provider: TracerProvider) -> None:
    """Flush + stop an evicted provider's processors (reclaims their threads).

    ``TracerProvider.shutdown`` force-flushes each ``SpanProcessor`` before
    stopping it, so any spans already handed to a ``BatchSpanProcessor`` are
    exported rather than dropped. Best-effort: a shutdown failure must not break
    the request that triggered the eviction.
    """
    try:
        provider.shutdown()
    except Exception as e:  # pragma: no cover - defensive
        verbose_logger.debug("OTel V2: error shutting down evicted provider: %s", e)


class TenantTracerCache:
    """Destination-scoped ``TracerProvider`` cache keyed by endpoint + headers."""

    def __init__(
        self,
        config: OpenTelemetryV2Config,
        callback_name: str | None,
        tracer_name: str,
    ) -> None:
        self._config = config
        self._callback_name = callback_name
        self._tracer_name = tracer_name
        self._providers: "OrderedDict[tuple[tuple[str, tuple[tuple[str, str], ...]], ...], TracerProvider]" = (OrderedDict())

    def tracer_for(
        self, default: Tracer, destinations: "tuple[OtelDestination, ...]"
    ) -> Tracer:
        """Return the tracer for this request.

        ``destinations`` are the admin-resolved exporters (for this backend) that the
        request's identity chain is assigned. Empty -> the logger's default tracer
        (the global/configured exporter only = deny). Otherwise build (or reuse) a
        provider that exports to the configured exporters PLUS every destination, so
        one span is emitted once and copied to all (fan-out, no duplicate spans). The
        cache is a bounded LRU keyed on the destination SET.
        """
        if not destinations:
            return default
        cache_key = tuple(
            sorted((d.endpoint, tuple(sorted(d.headers.items()))) for d in destinations)
        )
        provider = self._providers.get(cache_key)
        if provider is not None:
            self._providers.move_to_end(cache_key)
        else:
            provider = build_tracer_provider(
                self._config_with_destinations(destinations)
            )
            self._providers[cache_key] = provider
            if len(self._providers) > _MAX_CACHED_PROVIDERS:
                _, evicted = self._providers.popitem(last=False)
                _shutdown_provider(evicted)
        return get_tracer(provider, self._tracer_name)

    def _owned_otlp_kind(self) -> str:
        """The OTLP transport of this integration's own exporter (langfuse -> http,
        arize -> grpc), used for the destinations appended below."""
        for spec in self._config.exporters:
            if (
                spec.owner == self._callback_name
                and spec.kind.lower() not in _NON_OTLP_KINDS
            ):
                return spec.kind
        return "otlp_http"

    def _config_with_destinations(
        self, destinations: "tuple[OtelDestination, ...]"
    ) -> OpenTelemetryV2Config:
        """Clone the config, KEEPING its exporters (so the global/default still
        receives) and APPENDING one exporter per resolved destination. The shared
        ``TracerProvider`` attaches one ``SpanProcessor`` per spec, so a single span
        is emitted once and exported to the global destination plus every assigned
        one. Each appended exporter's endpoint is the resolved host (the cross-host
        fix) with its own auth headers (per-destination isolation)."""
        kind = self._owned_otlp_kind()
        appended = [
            ExporterSpec(
                kind=kind,
                endpoint=d.endpoint,
                headers=d.header_string(),
                owner=None,
            )
            for d in destinations
        ]
        return self._config.model_copy(
            update={"exporters": [*self._config.exporters, *appended]}
        )
