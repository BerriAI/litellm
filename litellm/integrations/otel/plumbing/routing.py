"""Per-request multi-tenant tracer routing and span fan-out.

``TenantTracerCache`` routes the gen-AI LLM-call span, building per-tenant clone
``TracerProvider``s that export to the request's admin-owned destinations plus the
configured/global exporter. ``TenantFanOutSpanProcessor`` (at the bottom) forwards the
proxy-internal spans (server, auth, DB, cost) to every destination. Both read the request's
destinations from the same server-only contextvar, so a caller can neither redirect a trace
nor spawn providers.

The gen-AI path (``tracers_for``) groups destinations by their backend-required Resource
attributes and builds one provider per group, because a span carries exactly one Resource and
a backend like Arize selects its project FROM it, so two Arize projects each get a correctly
tagged span instead of a last-wins merge. Empty destinations -> the logger's default (global only).
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

_NON_OTLP_KINDS = ("console", "in_memory", "inmemory", "memory")

_MAX_CACHED_PROVIDERS = 256


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
        self._providers: OrderedDict[tuple[object, ...], TracerProvider] = (
            OrderedDict()
        )  # mutable-ok: bounded LRU tracer-provider cache

    def _evict_if_full(self) -> None:
        """Drop the least-recently-used provider when over capacity (no synchronous
        ``shutdown``; the evicted worker drains on its own and is reclaimed at process exit)."""
        if len(self._providers) > _MAX_CACHED_PROVIDERS:
            self._providers.popitem(last=False)

    def tracers_for(self, default: Tracer, destinations: "tuple[OtelDestination, ...]") -> "tuple[Tracer, ...]":
        """The tracers for this request's gen-AI span, one per distinct Resource group.

        A backend like Arize selects its project from the Resource, so destinations are grouped
        by ``destination_resource_attrs`` and the caller emits the span once per tracer. The
        configured/global exporters ride the FIRST group only, so the global receives the span
        once. Empty ``destinations`` -> the logger's default tracer (deny).
        """
        if not destinations:
            return (default,)
        return tuple(
            self._tracer_for_group(resource_key, group, include_base=index == 0)
            for index, (resource_key, group) in enumerate(self._group_by_resource(destinations))
        )

    def _group_by_resource(
        self, destinations: "tuple[OtelDestination, ...]"
    ) -> "tuple[tuple[tuple[tuple[str, str], ...], tuple[OtelDestination, ...]], ...]":
        """Destinations grouped by their backend-required Resource attributes.

        Groups sort deterministically by key, so the empty-Resource group (header-routed
        backends) sorts first and the configured/global exporters attach to it.
        """
        from litellm.integrations.otel.plumbing.providers import (
            destination_resource_attrs,
        )

        groups: OrderedDict[tuple[tuple[str, str], ...], list[OtelDestination]] = (
            OrderedDict()
        )  # mutable-ok: insertion-order grouping accumulator, frozen before return
        for destination in destinations:
            key = tuple(sorted(destination_resource_attrs(destination).items()))
            groups.setdefault(key, []).append(destination)
        return tuple((key, tuple(group)) for key, group in sorted(groups.items()))

    def _tracer_for_group(
        self,
        resource_key: "tuple[tuple[str, str], ...]",
        group: "tuple[OtelDestination, ...]",
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
            self._evict_if_full()
        return get_tracer(provider, self._tracer_name)

    def tracer_for(self, default: Tracer, destinations: "tuple[OtelDestination, ...]") -> Tracer:
        """Single merged tracer for ``destinations`` (one provider, one Resource).

        The single-group primitive ``tracers_for`` composes per group.
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
            self._evict_if_full()
        return get_tracer(provider, self._tracer_name)

    def _owned_otlp_kind(self) -> str:
        """The OTLP transport of this integration's own exporter (langfuse -> http, arize -> grpc).

        Prefers the admin's configured exporter kind; falls back to the backend's intrinsic
        default so a lazily-activated backend with no owned spec still picks the right transport.
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
        """Clone the config, appending one exporter per resolved destination (its resolved host
        and own auth headers) so one span exports to every destination.

        ``include_base_exporters`` keeps the configured/global exporters; ``tracers_for`` sets it
        only on the first Resource group so the global gets the span once, not once per group. The
        clone's Resource folds in the destinations' ``destination_resource_attrs`` (Arize needs
        ``model_id`` / ``arize.project.name``); callers group by those first so the merge is lossless."""
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


_MAX_CACHED_PROCESSORS = 256

_GENAI_SPAN_ATTR = "gen_ai.operation.name"


def _processor_key(destination: OtelDestination) -> tuple:
    return (destination.endpoint, tuple(sorted(destination.headers.items())))


def _is_genai_span(span: ReadableSpan) -> bool:
    attributes = span.attributes or {}
    return _GENAI_SPAN_ATTR in attributes


def _with_destination_resource(span: ReadableSpan, destination: OtelDestination) -> ReadableSpan:
    """Return ``span`` with its Resource augmented by the destination's required attributes,
    via a shallow wrapper that leaves the original span untouched."""
    from litellm.integrations.otel.plumbing.providers import (
        destination_resource_attrs,
    )

    extra = destination_resource_attrs(destination)
    if not extra:
        return span
    merged = Resource.create({**dict(span.resource.attributes), **extra})
    return _ResourceWrappedReadableSpan(span, merged)


class _ResourceWrappedReadableSpan(ReadableSpan):
    """A ``ReadableSpan`` view whose ``resource`` is overridden (for backend-specific attributes
    like Arize's ``model_id``) without mutating the underlying span."""

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

    Destinations come from a request-scoped contextvar set during auth, so the processor is
    stateless across requests and concurrent requests are isolated by contextvars.
    """

    def __init__(self, owner_callback_name: str | None) -> None:
        self._owner = owner_callback_name
        self._processors: OrderedDict[tuple, SpanProcessor] = (
            OrderedDict()
        )  # mutable-ok: bounded LRU span-processor cache

    def on_start(self, span: Span, parent_context: Context | None = None) -> None:
        return None

    def on_end(self, span: ReadableSpan) -> None:
        destinations = request_destinations()
        if not destinations:
            return
        if _is_genai_span(span):
            return
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
            self._processors.popitem(last=False)
        return processor
