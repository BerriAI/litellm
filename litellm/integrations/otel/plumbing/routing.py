"""Per-request multi-tenant tracer routing.

When a request carries team/key vendor credentials in
``standard_callback_dynamic_params``, its spans must export through a
``TracerProvider`` whose OTLP headers carry those credentials.
``TenantTracerCache`` builds and caches one provider per distinct credential
set, and otherwise hands back the logger's default tracer. This lets a single
logger fan requests out to many tenants without needing a logger per tenant.
"""

from collections import OrderedDict
from typing import Any, Mapping

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.trace import Tracer

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import OpenTelemetryV2Config
from litellm.integrations.otel.presets import dynamic_otlp_headers
from litellm.integrations.otel.plumbing.providers import (
    build_tracer_provider,
    get_tracer,
)

# Exporter kinds that ignore headers — never rewritten with dynamic credentials.
_NON_OTLP_KINDS = ("console", "in_memory", "inmemory", "memory")

# Cap on distinct credential-scoped providers held at once. ``dynamic_params``
# can be populated from request metadata, so an unbounded cache lets a caller
# spawn one ``TracerProvider`` (plus its ``BatchSpanProcessor`` background
# thread) per unique credential set and exhaust the proxy. The LRU bound keeps
# the working set of active tenants resident while flushing and shutting down
# evicted providers so their threads are reclaimed.
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
    """Credential-scoped ``TracerProvider`` cache keyed by the dynamic headers."""

    def __init__(
        self,
        config: OpenTelemetryV2Config,
        callback_name: str | None,
        tracer_name: str,
    ) -> None:
        self._config = config
        self._callback_name = callback_name
        self._tracer_name = tracer_name
        self._providers: "OrderedDict[tuple[tuple[str, str], ...], TracerProvider]" = OrderedDict()

    def tracer_for(self, default: Tracer, dynamic_params: Any) -> Tracer:
        """Return the tracer for this request.

        Use ``default`` unless the request's dynamic credentials require a
        credential-scoped tracer, in which case build (or reuse) one. The cache
        is a bounded LRU: the least-recently-used provider is flushed and shut
        down on overflow so its exporter threads don't accumulate.
        """
        headers = dynamic_otlp_headers(self._callback_name, dynamic_params)
        if not headers:
            return default
        cache_key = tuple(sorted(headers.items()))
        provider = self._providers.get(cache_key)
        if provider is not None:
            self._providers.move_to_end(cache_key)
        else:
            provider = build_tracer_provider(self._config_with_headers(headers))
            self._providers[cache_key] = provider
            if len(self._providers) > _MAX_CACHED_PROVIDERS:
                _, evicted = self._providers.popitem(last=False)
                _shutdown_provider(evicted)
        return get_tracer(provider, self._tracer_name)

    def _config_with_headers(self, headers: Mapping[str, str]) -> OpenTelemetryV2Config:
        """Clone the config, stamping ``headers`` onto the credential's own exporter.

        ``headers`` are the per-request credentials of ``self._callback_name`` (the
        integration that built this cache), so they apply only to the exporter that
        integration contributed (``spec.owner``). A request that carries one
        tenant's Arize key must never rewrite the headers of a co-configured
        Langfuse or self-hosted collector exporter, which would leak that key to a
        different backend.
        """
        header_str = ",".join(f"{key}={value}" for key, value in headers.items())
        header_update: dict[str, str] = {"headers": header_str}
        exporters = [
            (
                spec.model_copy(update=header_update)
                if spec.owner == self._callback_name and spec.kind.lower() not in _NON_OTLP_KINDS
                else spec
            )
            for spec in self._config.exporters
        ]
        return self._config.model_copy(update={"exporters": exporters})
