"""Per-request multi-tenant tracer routing.

When a request carries team/key vendor credentials in
``standard_callback_dynamic_params``, or the key/team config resolved at auth
names a destination project or a service name, its spans must export through a
``TracerProvider`` whose OTLP headers carry those credentials / that project,
or whose Resource carries that ``service.name``. ``TenantTracerCache`` builds
and caches one provider per distinct (credentials, project, service name)
tuple, and otherwise hands back the logger's default tracer. This lets a
single logger fan requests out to many tenants without needing a logger per
tenant.
"""

import threading
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias
from urllib.parse import quote

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from litellm._logging import verbose_logger
from litellm.constants import OTEL_SERVICE_NAME_METADATA_KEYS
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.plumbing.providers import (
    build_tracer_provider,
    get_tracer,
)
from litellm.integrations.otel.presets import (
    dynamic_otlp_endpoint,
    dynamic_otlp_headers,
    project_routing_headers,
)
from litellm.types.utils import StandardCallbackDynamicParams

# Exporter kinds that ignore headers — never rewritten with dynamic credentials.
_NON_OTLP_KINDS: Final = ("console", "in_memory", "inmemory", "memory")

# gRPC exporters still take dynamic credentials (as gRPC metadata) but not
# project headers: the routing headers backends read (Phoenix's
# ``x-project-name``) are only honored on the OTLP/HTTP endpoint.
_GRPC_KINDS: Final = ("otlp_grpc", "grpc")

# Cap on distinct credential-scoped providers held at once. ``dynamic_params``
# can be populated from request metadata, so an unbounded cache lets a caller
# spawn one ``TracerProvider`` (plus its ``BatchSpanProcessor`` background
# thread) per unique credential set and exhaust the proxy. The LRU bound keeps
# the working set of active tenants resident while flushing and shutting down
# evicted providers so their threads are reclaimed.
_MAX_CACHED_PROVIDERS: Final = 256

# Cap on providers evicted from the cache while still holding open spans, which
# are kept alive to drain instead of being shut down under them. Their only
# other bound is the logger's open-call map (10k), so without this a caller
# cycling unique credential sets across long-lived calls could pin far more
# live providers, and exporter threads, than the cache cap allows. Past this
# many, the stalest retiree is shut down and whatever it was draining is
# dropped (a shut-down ``BatchSpanProcessor`` discards spans handed to it after
# the fact), which by then means a span on a route evicted long ago. A quarter
# of the cache cap: enough that a burst of tenant churn during long-lived calls
# still drains normally, small enough that the worst case is a bounded 320
# providers rather than one per concurrent call.
_MAX_RETIRED_PROVIDERS: Final = 64

_HeaderItems: TypeAlias = tuple[tuple[str, str], ...]

_RouteKey: TypeAlias = tuple[_HeaderItems, _HeaderItems, str | None, str | None]

_NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})

#: Key/team config fields naming the Resource ``service.name``, highest
#: precedence first. Read only from ``user_api_key_auth_metadata`` (the config
#: the proxy resolved at auth), never from client-supplied request metadata:
#: the service name picks the dataset/service traces land in (Honeycomb routes
#: datasets by it), so a caller must not be able to choose one.
_SERVICE_NAME_KEYS: Final = OTEL_SERVICE_NAME_METADATA_KEYS


def tenant_service_name(auth_metadata: Mapping[str, str] | None) -> str | None:
    """The per-request ``service.name`` override for this key/team, if any.

    ``None`` keeps the env-configured default (``OTEL_SERVICE_NAME``).
    """
    if not auth_metadata:
        return None
    return next(
        (stripped for key in _SERVICE_NAME_KEYS if (stripped := (auth_metadata.get(key) or "").strip())),
        None,
    )


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


def _plain_header_string(headers: Mapping[str, str]) -> str:
    return ",".join(f"{key}={value}" for key, value in headers.items())


def _encoded_header_string(headers: Mapping[str, str]) -> str:
    """Percent-encode values so one containing the ``k=v,k=v`` separators (e.g.
    a project name with a comma) survives; ``parse_env_headers`` decodes it back.
    """
    return ",".join(f"{key}={quote(value, safe='')}" for key, value in headers.items())


@dataclass(frozen=True, slots=True)
class TenantRoute:
    """The tracer to create a span on, plus whether it must root its own trace.

    ``detached`` is True when project routing engaged. Phoenix assigns a whole
    trace to one project by whichever of its spans arrives first, so a
    project-routed span parented into the request trace gets dragged into the
    project of the default-exported request spans and the header does nothing.
    The span must therefore start a fresh trace (with a link back to the
    request trace for correlation) — which is also how the v1 Phoenix logger
    behaved, exporting each request under its own Phoenix-local parent span.
    """

    tracer: Tracer
    detached: bool
    #: The provider ``tracer`` came from, or ``None`` on the default route. It
    #: is returned already held (counted as an open span, atomically with the
    #: cache update), so LRU eviction can't shut it down before the caller's
    #: span lands; the caller must ``release`` it exactly once when done.
    provider: TracerProvider | None = None


class TenantTracerCache:
    """Tenant-scoped ``TracerProvider`` cache keyed by routing headers and service name."""

    def __init__(
        self,
        config: OpenTelemetryV2Config,
        callback_name: str | None,
        tracer_name: str,
    ) -> None:
        self._config = config
        self._callback_name = callback_name
        self._tracer_name = tracer_name
        # Guards the three mutable structures below: ``pre_call`` can run on
        # thread-pool workers concurrently with the event loop, so cache
        # updates, span counts, and retirement must be atomic.
        self._lock: Final = threading.Lock()
        self._providers: OrderedDict[_RouteKey, TracerProvider] = (
            OrderedDict()  # mutable-ok: bounded LRU; eviction needs in-place ordered mutation
        )
        self._open_span_counts: dict[TracerProvider, int] = {}  # mutable-ok: live refcount state
        # Oldest-first so an overflow of draining providers sheds the stalest.
        self._retired: OrderedDict[TracerProvider, None] = OrderedDict()  # mutable-ok: draining evicted providers
        self._project_routable = any(
            spec.owner == callback_name and spec.kind.lower() not in (*_NON_OTLP_KINDS, *_GRPC_KINDS)
            for spec in config.exporters
        )
        self._warned_project_unroutable = False

    def release(self, provider: TracerProvider | None) -> None:
        """Drop one open-span count; shut a retired provider down once drained.

        ``None`` (the default route) is a no-op so callers can release a
        ``TenantRoute.provider`` unconditionally. The shutdown itself runs
        outside the lock: it force-flushes over the network and must not stall
        every concurrently routing request.
        """
        if provider is None:
            return
        with self._lock:
            remaining: Final = self._open_span_counts.get(provider, 0) - 1
            if remaining > 0:
                self._open_span_counts[provider] = remaining
                return
            self._open_span_counts.pop(provider, None)
            drained: Final = provider in self._retired
            self._retired.pop(provider, None)
        if drained:
            _shutdown_provider(provider)

    def route_for(
        self,
        default: Tracer,
        dynamic_params: StandardCallbackDynamicParams | None,
        auth_metadata: Mapping[str, str] | None = None,
    ) -> TenantRoute:
        """Return the tracer (and trace-detachment flag) for this request.

        Use ``default`` unless the request's dynamic credentials, its key/team
        project, or its key/team service name require a scoped tracer, in
        which case build (or reuse) one. The cache is a bounded LRU: the
        least-recently-used provider is flushed and shut down on overflow so
        its exporter threads don't accumulate.

        A routed provider is returned already held — its open-span count is
        incremented in the same critical section as the cache update — so a
        concurrent overflow eviction can't shut it down between selection and
        the caller's span start. The caller must ``release`` it exactly once.
        """
        credential_headers: Final = dynamic_otlp_headers(self._callback_name, dynamic_params) or _NO_HEADERS
        project_headers: Final = self._project_headers(auth_metadata)
        service_name: Final = tenant_service_name(auth_metadata)
        if not credential_headers and not project_headers and service_name is None:
            return TenantRoute(tracer=default, detached=False)
        # A fixed per-integration region endpoint (New Relic us/eu), never a
        # caller-supplied host; ``None`` keeps the preset's own endpoint.
        endpoint: Final = dynamic_otlp_endpoint(self._callback_name, dynamic_params)
        cache_key: Final = (
            tuple(sorted(credential_headers.items())),
            tuple(sorted(project_headers.items())),
            endpoint,
            service_name,
        )
        with self._lock:
            provider: Final = self._cached_provider_locked(
                cache_key, credential_headers, project_headers, endpoint, service_name
            )
            self._open_span_counts[provider] = self._open_span_counts.get(provider, 0) + 1
            evicted: Final = self._evicted_on_overflow_locked()
        if evicted is not None:
            _shutdown_provider(evicted)
        return TenantRoute(
            tracer=get_tracer(provider, self._tracer_name),
            detached=bool(project_headers),
            provider=provider,
        )

    def _cached_provider_locked(
        self,
        cache_key: _RouteKey,
        credential_headers: Mapping[str, str],
        project_headers: Mapping[str, str],
        endpoint: str | None,
        service_name: str | None,
    ) -> TracerProvider:
        cached: Final = self._providers.get(cache_key)
        if cached is not None:
            self._providers.move_to_end(cache_key)
            return cached
        built: Final = build_tracer_provider(
            self._routed_config(credential_headers, project_headers, endpoint, service_name)
        )
        self._providers[cache_key] = built
        return built

    def _evicted_on_overflow_locked(self) -> TracerProvider | None:
        """Pop the LRU provider past the cap; return it if the caller must shut it down.

        A provider with open spans is retired to drain instead: stopping its
        processors while a span opened at ``pre_call`` is still live would
        silently drop that span at end instead of exporting it. Retirees are
        themselves capped, so the stalest one is shut down (and its open-span
        count dropped, making its eventual ``release`` a no-op) once too many
        pile up rather than letting them accumulate a thread each.
        """
        if len(self._providers) <= _MAX_CACHED_PROVIDERS:
            return None
        _, evicted = self._providers.popitem(last=False)
        if self._open_span_counts.get(evicted, 0) == 0:
            return evicted
        self._retired[evicted] = None
        if len(self._retired) <= _MAX_RETIRED_PROVIDERS:
            return None
        overflowed, _ = self._retired.popitem(last=False)
        self._open_span_counts.pop(overflowed, None)
        return overflowed

    def _project_headers(self, auth_metadata: Mapping[str, str] | None) -> Mapping[str, str]:
        """The per-request project-routing headers, if this cache can apply them.

        A gRPC-only exporter can't (the project header route is HTTP-only), so
        the request warns once and stays on the env-configured default project.
        """
        requested: Final = project_routing_headers(self._callback_name, auth_metadata)
        if not requested or self._project_routable:
            return requested
        if not self._warned_project_unroutable:
            self._warned_project_unroutable = True
            verbose_logger.warning(
                "OTel V2: %s key/team config names a per-request project, but its exporter "
                "is not OTLP/HTTP and the project header is HTTP-only; spans stay in the "
                "default project.",
                self._callback_name,
            )
        return _NO_HEADERS

    def _routed_config(
        self,
        credential_headers: Mapping[str, str],
        project_headers: Mapping[str, str],
        endpoint: str | None = None,
        service_name: str | None = None,
    ) -> OpenTelemetryV2Config:
        """Clone the config, rewriting headers on the callback's own exporter.

        Both header sets apply only to the exporter ``self._callback_name``
        contributed (``spec.owner``). A request that carries one tenant's Arize
        key must never rewrite the headers of a co-configured Langfuse or
        self-hosted collector exporter, which would leak that key to a
        different backend.

        Dynamic credentials REPLACE the exporter's headers — they are the
        tenant's complete credential set. Project headers APPEND instead: the
        preset's static headers carry the backend auth (Phoenix's
        ``Authorization``), which must survive routing to a project.
        """
        exporters: Final = [
            self._routed_exporter(spec, credential_headers, project_headers, endpoint)
            for spec in self._config.exporters
        ]
        update: Final = (
            {"exporters": exporters} if service_name is None else {"exporters": exporters, "service_name": service_name}
        )
        return self._config.model_copy(update=update)

    def _routed_exporter(
        self,
        spec: ExporterSpec,
        credential_headers: Mapping[str, str],
        project_headers: Mapping[str, str],
        endpoint: str | None = None,
    ) -> ExporterSpec:
        kind: Final = spec.kind.lower()
        if spec.owner != self._callback_name or kind in _NON_OTLP_KINDS:
            return spec
        base: Final = _plain_header_string(credential_headers) if credential_headers else spec.headers
        routed: Final = (
            ",".join(part for part in (base, _encoded_header_string(project_headers)) if part)
            if project_headers and kind not in _GRPC_KINDS
            else base
        )
        update: Final = {  # mutable-ok: model_copy(update=...) requires a plain dict
            field: value
            for field, value in (("headers", routed), ("endpoint", endpoint))
            if (field == "headers" and routed != spec.headers)
            or (field == "endpoint" and endpoint is not None and endpoint != spec.endpoint)
        }
        return spec if not update else spec.model_copy(update=update)
