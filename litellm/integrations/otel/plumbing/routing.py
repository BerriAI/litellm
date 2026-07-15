"""Per-request multi-tenant tracer routing and span fan-out.

This module owns both halves of routing a request's spans to its admin-owned
destinations. ``TenantTracerCache`` handles the gen-AI LLM-call span by building
per-tenant clone providers (grouped by backend Resource attributes).
``TenantFanOutSpanProcessor`` (at the bottom) handles the proxy-internal spans on
the main provider (server, auth, DB, cost ledger) by forwarding each to every
destination. Both read the request's destinations from the same server-only
contextvar, so there is one source of truth for where a request's traces go.

A call's identity chain is assigned a set of admin-owned OTEL destinations
(``LLMCallEvent.otel_destinations``, resolved server-side from named credentials).
Its spans must export to ALL of them plus the configured/global exporter, so
``TenantTracerCache`` builds and caches ``TracerProvider``s that append one
``SpanProcessor`` per destination. The gen-AI span path (``tracers_for``) groups
destinations by their backend-required Resource attributes and builds one provider
per group, because a span carries exactly one Resource (a provider property) and a
backend like Arize selects its project FROM the Resource -- so two Arize projects
each get a correctly-tagged span instead of one last-wins merge. Header-routed
backends declare no Resource attributes, so their destinations stay in one group with
multiple exporters and route by per-exporter auth. With no destinations it hands back
the logger's default tracer (global only). Destinations are never request-derived, so
a caller can neither redirect a trace nor spawn providers.
"""

from collections import OrderedDict

from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import ReadableSpan, Span, SpanProcessor, TracerProvider
from opentelemetry.trace import Tracer

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.model.destination import OtelDestination
from litellm.integrations.otel.plumbing.context import request_destinations
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
        self._providers: OrderedDict[tuple[object, ...], TracerProvider] = OrderedDict()

    def tracers_for(self, default: Tracer, destinations: "tuple[OtelDestination, ...]") -> "tuple[Tracer, ...]":
        """The tracers for this request's gen-AI span, one per distinct Resource group.

        A span carries exactly one Resource (it's a property of the ``TracerProvider``),
        but a backend like Arize selects its project FROM the Resource
        (``arize.project.name`` / ``model_id``), so two Arize destinations with different
        projects need two differently-tagged spans. Group the backend's resolved
        destinations by ``destination_resource_attrs`` and return one tracer per group;
        the caller emits the span once per tracer (mirroring how the fan-out processor
        re-wraps proxy-internal spans per destination).

        Header-routed backends (langfuse, weave) declare no Resource attributes, so all
        their destinations collapse into one empty-Resource group with one exporter each
        and keep routing by per-exporter auth -- unchanged from the single-group path.
        The configured/global exporters ride the FIRST group only, so the global receives
        the span once. Empty ``destinations`` -> the logger's default tracer (deny).
        """
        if not destinations:
            return (default,)
        return tuple(
            self._tracer_for_group(resource_key, group, include_base=index == 0)
            for index, (resource_key, group) in enumerate(self._group_by_resource(destinations))
        )

    def _group_by_resource(
        self, destinations: "tuple[OtelDestination, ...]"
    ) -> "list[tuple[tuple[tuple[str, str], ...], list[OtelDestination]]]":
        """Destinations grouped by their backend-required Resource attributes.

        The key is a stable sorted tuple of ``destination_resource_attrs`` items.
        Groups are returned in a deterministic order (sorted by key), so the
        empty-Resource group (header-routed backends) sorts first and the
        configured/global exporters attach to it.
        """
        from litellm.integrations.otel.plumbing.providers import (
            destination_resource_attrs,
        )

        groups: OrderedDict[tuple[tuple[str, str], ...], list[OtelDestination]] = OrderedDict()
        for destination in destinations:
            key = tuple(sorted(destination_resource_attrs(destination).items()))
            groups.setdefault(key, []).append(destination)
        return sorted(groups.items())

    def _tracer_for_group(
        self,
        resource_key: "tuple[tuple[str, str], ...]",
        group: "list[OtelDestination]",
        *,
        include_base: bool,
    ) -> Tracer:
        cache_key: tuple[object, ...] = (
            resource_key,
            tuple(sorted((d.endpoint, tuple(sorted(d.headers.items()))) for d in group)),
            include_base,
        )
        provider = self._providers.get(cache_key)
        if provider is not None:
            self._providers.move_to_end(cache_key)
        else:
            provider = build_tracer_provider(
                self._config_with_destinations(tuple(group), include_base_exporters=include_base)
            )
            self._providers[cache_key] = provider
            if len(self._providers) > _MAX_CACHED_PROVIDERS:
                _, evicted = self._providers.popitem(last=False)
                _shutdown_provider(evicted)
        return get_tracer(provider, self._tracer_name)

    def tracer_for(self, default: Tracer, destinations: "tuple[OtelDestination, ...]") -> Tracer:
        """Single merged tracer for ``destinations`` (one provider, one Resource).

        The single-group primitive: kept for the destination-set cache mechanics and as
        the building block ``tracers_for`` composes per group. The gen-AI span path uses
        ``tracers_for`` so multiple Resource groups aren't last-wins merged.
        """
        if not destinations:
            return default
        cache_key = tuple(sorted((d.endpoint, tuple(sorted(d.headers.items()))) for d in destinations))
        provider = self._providers.get(cache_key)
        if provider is not None:
            self._providers.move_to_end(cache_key)
        else:
            provider = build_tracer_provider(self._config_with_destinations(destinations))
            self._providers[cache_key] = provider
            if len(self._providers) > _MAX_CACHED_PROVIDERS:
                _, evicted = self._providers.popitem(last=False)
                _shutdown_provider(evicted)
        return get_tracer(provider, self._tracer_name)

    def _owned_otlp_kind(self) -> str:
        """The OTLP transport of this integration's own exporter (langfuse -> http,
        arize -> grpc), used for the destinations appended below.

        Prefer the admin's configured exporter kind for this backend; fall back to
        the backend's intrinsic default (shared with the fan-out processor via
        ``default_otlp_kind_for_backend``) so a lazily-activated backend with no
        owned spec still picks the right transport (e.g. arize -> grpc, not http).
        """
        from litellm.integrations.otel.plumbing.providers import (
            default_otlp_kind_for_backend,
        )

        for spec in self._config.exporters:
            if spec.owner == self._callback_name and spec.kind.lower() not in _NON_OTLP_KINDS:
                return spec.kind
        return default_otlp_kind_for_backend(self._callback_name)

    def _config_with_destinations(
        self,
        destinations: "tuple[OtelDestination, ...]",
        *,
        include_base_exporters: bool = True,
    ) -> OpenTelemetryV2Config:
        """Clone the config and APPEND one exporter per resolved destination. The shared
        ``TracerProvider`` attaches one ``SpanProcessor`` per spec, so a single span is
        emitted once and exported to every appended destination. Each appended exporter's
        endpoint is the resolved host (the cross-host fix) with its own auth headers
        (per-destination isolation).

        ``include_base_exporters`` keeps the configured/global exporters too (so the
        global still receives). ``tracers_for`` sets it only on the first Resource group,
        so when a backend splits into multiple groups the global gets the span once
        rather than once per group.

        The clone's Resource folds in the destinations' backend-required Resource
        attributes (Arize needs ``model_id`` / ``arize.project.name``), via the same
        ``destination_resource_attrs`` the fan-out path uses on proxy-internal spans.
        Callers group destinations by those attributes first, so within one call all
        ``destinations`` share a Resource and the merge is not lossy -- without this the
        gen-AI span would reach Arize with only ``service.name`` while its parents
        (fan-out) carry ``model_id``, orphaning the subtree."""
        from litellm.integrations.otel.plumbing.providers import (
            destination_resource_attrs,
        )

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
        base_exporters = [*self._config.exporters] if include_base_exporters else []
        merged_resource_attrs = {
            **self._config.resource_attributes,
            **{key: value for d in destinations for key, value in destination_resource_attrs(d).items()},
        }
        return self._config.model_copy(
            update={
                "exporters": [*base_exporters, *appended],
                "resource_attributes": merged_resource_attrs,
            }
        )


# --- Proxy-internal span fan-out ------------------------------------------- #
#
# ``TenantTracerCache`` above routes the gen-AI LLM-call span (and its MCP-tool
# sibling) to per-tenant destinations through clone providers. The processor below
# handles the OTHER span class: the proxy-internal spans (FastAPI server span, the
# ``auth`` phase, DB lookups, the post-call cost ledger) emitted on the MAIN
# provider. It forwards each to every admin-resolved destination for the request,
# reading them from the same server-only contextvar the cache's callers set, so both
# routing paths share one source of truth for where a request's traces go.

# Bound on cached per-destination processors. One processor per
# ``(endpoint, sorted(headers))`` pair, so the working set is one entry per
# admin-resolved tenant credential -- a real-world deployment with hundreds of
# tenants stays well under this. Evicted entries are dropped (not shut down; see
# the eviction site) and reclaimed at process exit.
_MAX_CACHED_PROCESSORS = 256

# Attribute set on every gen-AI LLM-call span by the v2 emitter. Used as the
# unambiguous skip signal: only the LLM-call span carries this, and the per-backend
# v2 logger already routes it to per-tenant destinations through the
# TenantTracerCache clone provider's appended exporter.
_GENAI_SPAN_ATTR = "gen_ai.operation.name"


def _processor_key(destination: OtelDestination) -> tuple:
    return (destination.endpoint, tuple(sorted(destination.headers.items())))


def _is_genai_span(span: ReadableSpan) -> bool:
    attributes = span.attributes or {}
    return _GENAI_SPAN_ATTR in attributes


def _with_destination_resource(span: ReadableSpan, destination: OtelDestination) -> ReadableSpan:
    """Return ``span`` with its Resource augmented by the destination's required
    attributes. The original span object is left untouched; a shallow wrapper reuses
    every other field and only swaps the ``resource`` property."""
    from litellm.integrations.otel.plumbing.providers import (
        destination_resource_attrs,
    )

    extra = destination_resource_attrs(destination)
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


class TenantFanOutSpanProcessor(SpanProcessor):
    """Forward each finished proxy-internal span to every admin-resolved destination.

    The destinations are looked up from a request-scoped contextvar set during auth,
    so the processor is stateless across requests and isolation across concurrent
    requests is guaranteed by Python's contextvars.
    """

    def __init__(self, owner_callback_name: str | None) -> None:
        self._owner = owner_callback_name
        self._processors: OrderedDict[tuple, SpanProcessor] = OrderedDict()

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        destinations = request_destinations()
        if not destinations:
            return
        # The gen-AI LLM-call span (and the MCP tool-call sibling) is already routed
        # to per-tenant destinations by the per-backend v2 logger via
        # ``TenantTracerCache`` -- the logger picks the right attribute mapper
        # (OpenInference for arize, GenAI semconv for langfuse_otel) and ships through
        # the clone provider's appended exporter. Forwarding it here too would deliver
        # a SECOND copy with the wrong vocabulary and a fresh span_id, surfacing in the
        # destination as an orphaned duplicate. Skip.
        if _is_genai_span(span):
            return
        # Proxy-internal spans (FastAPI server, ``auth`` phase, postgres lookups,
        # post-call cost ledger) are generic OTel semantic-convention spans with no
        # backend-specific vocabulary, so they ship to EVERY admin-resolved destination
        # this request fans out to, regardless of the destination's ``callback_name``.
        for destination in destinations:
            processor = self._processor_for(destination)
            if processor is None:
                continue
            try:
                processor.on_end(_with_destination_resource(span, destination))
            except Exception as exc:  # noqa: BLE001  # best-effort fan-out; one destination's failure must not break the others or the request
                verbose_logger.debug(
                    "OTel V2 fan-out: forwarding span to %s failed: %s",
                    destination.endpoint,
                    exc,
                )

    def shutdown(self) -> None:
        for processor in self._processors.values():
            try:
                processor.shutdown()
            except Exception as exc:  # noqa: BLE001  # a single processor's shutdown failure must not abort shutting down the rest
                verbose_logger.debug("OTel V2 fan-out: processor shutdown failed: %s", exc)
        self._processors.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        all_ok = True
        for processor in self._processors.values():
            try:
                if not processor.force_flush(timeout_millis):
                    all_ok = False
            except Exception:  # noqa: BLE001  # a single processor's flush failure must not fail the whole force_flush
                all_ok = False
        return all_ok

    def _processor_for(self, destination: OtelDestination) -> SpanProcessor | None:
        key = _processor_key(destination)
        cached = self._processors.get(key)
        if cached is not None:
            self._processors.move_to_end(key)
            return cached
        from litellm.integrations.otel.plumbing.providers import (
            _exporter_from_spec,
            default_otlp_kind_for_backend,
        )
        from litellm.integrations.otel.plumbing.providers import (
            _processor_for as _build_processor,
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
        except Exception as exc:  # noqa: BLE001  # a malformed destination spec must not break fan-out; skip this destination
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
            # thread, and calling ``shutdown`` synchronously can drop or raise on those
            # in-flight spans. Dropping the reference lets the worker drain naturally
            # and be reclaimed at process exit. The cache is bounded, so the
            # un-shut-down working set stays bounded.
            self._processors.popitem(last=False)
        return processor
