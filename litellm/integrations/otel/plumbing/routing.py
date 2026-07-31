"""Per-request multi-tenant tracer routing and span fan-out.

``TenantTracerCache`` routes the gen-AI span to per-tenant/destination providers;
``TenantFanOutSpanProcessor`` forwards proxy-internal spans to every admin-resolved destination.
"""

import threading
from collections import OrderedDict
from typing import TYPE_CHECKING

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
from litellm.integrations.otel.presets import dynamic_otlp_headers

if TYPE_CHECKING:
    from litellm.types.utils import StandardCallbackDynamicParams

# Exporter kinds that ignore headers — never rewritten with dynamic credentials.
_NON_OTLP_KINDS = ("console", "in_memory", "inmemory", "memory")

# Cap on distinct credential-scoped providers held at once. ``dynamic_params``
# can be populated from request metadata, so an unbounded cache lets a caller
# spawn one ``TracerProvider`` (plus its ``BatchSpanProcessor`` background
# thread) per unique credential set and exhaust the proxy. The LRU bound keeps
# the working set of active tenants resident while flushing and shutting down
# evicted providers so their threads are reclaimed.
_MAX_CACHED_PROVIDERS = 256


def _shutdown_in_background(evicted: "TracerProvider | SpanProcessor") -> None:
    """Reclaim an evicted provider/processor's ``BatchSpanProcessor`` worker thread.

    Dropping it without ``shutdown`` leaks the daemon thread; ``shutdown`` force-flushes and can
    do network I/O, so it runs fire-and-forget on a daemon thread rather than the request path.
    """

    def _run() -> None:
        try:
            evicted.shutdown()
        except Exception as exc:  # noqa: BLE001  # a failed shutdown must not surface on the hot path
            verbose_logger.debug("OTel V2: background shutdown of evicted %s failed: %s", type(evicted).__name__, exc)

    threading.Thread(target=_run, name="litellm-otel-evict-shutdown", daemon=True).start()


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
        """Drop the least-recently-used provider when over capacity, shutting it down off
        the hot path so its ``BatchSpanProcessor`` worker thread is reclaimed rather than
        leaked for the life of the process."""
        if len(self._providers) > _MAX_CACHED_PROVIDERS:
            _, evicted = self._providers.popitem(last=False)
            _shutdown_in_background(evicted)

    def tracers_for(
        self,
        default: Tracer,
        destinations: "tuple[OtelDestination, ...]",
        *,
        include_base_on_first: bool = True,
    ) -> "tuple[Tracer, ...]":
        """The tracers for this request's gen-AI span, one per distinct Resource group.

        Destinations are grouped by ``destination_resource_attrs`` (a backend like Arize selects its
        project from the Resource); the configured/global exporters ride the first group only, and
        ``include_base_on_first=False`` drops them when a credential-scoped tracer already carries them.
        """
        if not destinations:
            return (default,)
        return tuple(
            self._tracer_for_group(resource_key, group, include_base=include_base_on_first and index == 0)
            for index, (resource_key, group) in enumerate(self._group_by_resource(destinations))
        )

    def genai_tracers_for(
        self,
        default: Tracer,
        destinations: "tuple[OtelDestination, ...]",
        dynamic_params: "StandardCallbackDynamicParams | None",
    ) -> "tuple[Tracer, ...]":
        """The gen-AI span's tracers, layering per-request credential routing over the destination
        fan-out: with this backend's team/key OTLP credentials the global export rides a
        credential-scoped provider and the destination groups omit the base exporters; else plain fan-out.
        """
        headers = dynamic_otlp_headers(self._callback_name, dynamic_params)
        if not headers:
            return self.tracers_for(default, destinations)
        dynamic = self._credential_scoped_tracer(headers)
        if not destinations:
            return (dynamic,)
        return (dynamic, *self.tracers_for(default, destinations, include_base_on_first=False))

    def dynamic_tracer_for(self, default: Tracer, dynamic_params: "StandardCallbackDynamicParams | None") -> Tracer:
        """The credential-scoped tracer when the request carries this backend's team/key OTLP
        credentials, else ``default``. Distinct from ``tracer_for`` (admin destinations); this
        is the per-request path restored for parity with the pre-v2-refactor behavior."""
        headers = dynamic_otlp_headers(self._callback_name, dynamic_params)
        if not headers:
            return default
        return self._credential_scoped_tracer(headers)

    def _credential_scoped_tracer(self, headers: "dict[str, str]") -> Tracer:
        """A cached provider that keeps the configured exporters and rewrites only this
        backend's owned exporter's headers to ``headers`` (the per-request credentials)."""
        cache_key: tuple[object, ...] = ("dynamic", tuple(sorted(headers.items())))
        provider = self._providers.get(cache_key)
        if provider is not None:
            self._providers.move_to_end(cache_key)
        else:
            provider = build_tracer_provider(self._config_with_headers(headers))
            self._providers[cache_key] = provider
            self._evict_if_full()
        return get_tracer(provider, self._tracer_name)

    def _config_with_headers(self, headers: "dict[str, str]") -> OpenTelemetryV2Config:
        """Clone the config, stamping ``headers`` onto the credential's own exporter.

        ``headers`` are the per-request credentials of ``self._callback_name`` (the
        integration that built this cache), so they apply only to the exporter that
        integration contributed (``spec.owner``). A request that carries one
        tenant's Arize key must never rewrite the headers of a co-configured
        Langfuse or self-hosted collector exporter, which would leak that key to a
        different backend.
        """
        header_str = ",".join(f"{key}={value}" for key, value in headers.items())
        exporters = [
            (
                spec.model_copy(update={"headers": header_str})
                if spec.owner == self._callback_name and spec.kind.lower() not in _NON_OTLP_KINDS
                else spec
            )
            for spec in self._config.exporters
        ]
        return self._config.model_copy(update={"exporters": exporters})

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
        """Clone the config, appending one exporter per destination so one span exports to every one.

        ``include_base_exporters`` keeps the configured/global exporters (``tracers_for`` sets it only on
        the first group); the clone's Resource folds in the destinations' ``destination_resource_attrs``.
        """
        from litellm.integrations.otel.plumbing.providers import (
            destination_resource_attrs,
        )

        kind = self._owned_otlp_kind()
        appended = tuple(
            ExporterSpec(
                kind=kind,
                endpoint=d.endpoint,
                headers=d.header_string(),
                owner=None,
            )
            for d in destinations
        )
        base_exporters = (*self._config.exporters,) if include_base_exporters else ()
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
        # Snapshot before iterating: ``on_end`` mutates ``self._processors`` (insert /
        # move_to_end / popitem) on the span-ending thread and can run concurrently with
        # this SDK-driven shutdown, so iterating the live mapping risks a
        # "mutated during iteration" RuntimeError that the per-item except can't catch.
        for processor in tuple(self._processors.values()):
            try:
                processor.shutdown()
            except Exception as exc:  # noqa: BLE001  # a single processor's shutdown failure must not abort shutting down the rest
                verbose_logger.debug("OTel V2 fan-out: processor shutdown failed: %s", exc)
        self._processors.clear()

    def force_flush(self, timeout_millis: int = 30000) -> bool:
        all_ok = True
        # Snapshot before iterating (see ``shutdown``): a concurrent ``on_end`` mutating
        # the processor cache must not abort the flush and drop the remaining destinations'
        # buffered spans.
        for processor in tuple(self._processors.values()):
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
            _, evicted = self._processors.popitem(last=False)
            _shutdown_in_background(evicted)
        return processor
