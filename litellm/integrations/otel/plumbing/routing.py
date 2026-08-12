"""Per-request multi-tenant tracer routing.

When a request carries team/key vendor credentials in
``standard_callback_dynamic_params``, or the key/team config resolved at auth
names a destination project, its spans must export through a
``TracerProvider`` whose OTLP headers carry those credentials / that project.
``TenantTracerCache`` builds and caches one provider per distinct
(credentials, project) pair, and otherwise hands back the logger's default
tracer. This lets a single logger fan requests out to many tenants without
needing a logger per tenant.
"""

from collections import OrderedDict
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Final, TypeAlias
from urllib.parse import quote

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import ExporterSpec, OpenTelemetryV2Config
from litellm.integrations.otel.plumbing.providers import (
    build_tracer_provider,
    get_tracer,
)
from litellm.integrations.otel.presets import (
    dynamic_otlp_headers,
    project_routing_headers,
)

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

_HeaderItems: TypeAlias = tuple[tuple[str, str], ...]

_NO_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})


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


class TenantTracerCache:
    """Credential/project-scoped ``TracerProvider`` cache keyed by the routing headers."""

    def __init__(
        self,
        config: OpenTelemetryV2Config,
        callback_name: str | None,
        tracer_name: str,
    ) -> None:
        self._config = config
        self._callback_name = callback_name
        self._tracer_name = tracer_name
        self._providers: OrderedDict[tuple[_HeaderItems, _HeaderItems], TracerProvider] = OrderedDict()
        self._project_routable = any(
            spec.owner == callback_name and spec.kind.lower() not in (*_NON_OTLP_KINDS, *_GRPC_KINDS)
            for spec in config.exporters
        )
        self._warned_project_unroutable = False

    def tracer_for(
        self,
        default: Tracer,
        dynamic_params: Any,
        auth_metadata: Mapping[str, str] | None = None,
    ) -> Tracer:
        """Return the tracer for this request.

        Use ``default`` unless the request's dynamic credentials or its key/team
        project require a scoped tracer, in which case build (or reuse) one. The
        cache is a bounded LRU: the least-recently-used provider is flushed and
        shut down on overflow so its exporter threads don't accumulate.
        """
        credential_headers: Final = dynamic_otlp_headers(self._callback_name, dynamic_params) or _NO_HEADERS
        project_headers: Final = self._project_headers(auth_metadata)
        if not credential_headers and not project_headers:
            return default
        cache_key: Final = (
            tuple(sorted(credential_headers.items())),
            tuple(sorted(project_headers.items())),
        )
        provider = self._providers.get(cache_key)
        if provider is not None:
            self._providers.move_to_end(cache_key)
        else:
            provider = build_tracer_provider(self._routed_config(credential_headers, project_headers))
            self._providers[cache_key] = provider
            if len(self._providers) > _MAX_CACHED_PROVIDERS:
                _, evicted = self._providers.popitem(last=False)
                _shutdown_provider(evicted)
        return get_tracer(provider, self._tracer_name)

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
            self._routed_exporter(spec, credential_headers, project_headers) for spec in self._config.exporters
        ]
        return self._config.model_copy(update={"exporters": exporters})

    def _routed_exporter(
        self,
        spec: ExporterSpec,
        credential_headers: Mapping[str, str],
        project_headers: Mapping[str, str],
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
        return spec if routed == spec.headers else spec.model_copy(update={"headers": routed})
