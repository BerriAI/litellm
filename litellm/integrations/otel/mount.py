"""FastAPI server-span instrumentation — the proxy mounts this at app creation.

``opentelemetry-instrumentation-fastapi`` creates the SERVER span for each HTTP
route and extracts inbound ``traceparent`` headers. This module owns the one call
site that attaches it to the proxy app, plus the passthrough span-naming hook, so
``proxy_server`` stays free of OTel details.

The ``FastAPIInstrumentor`` import is kept lazy (inside :func:`instrument_fastapi_app`,
after the gate check) so importing this module never requires the optional
``opentelemetry-instrumentation-fastapi`` package and pulls in nothing OTel-related
when the feature gate is off.
"""

import os
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import is_otel_v2_enabled

if TYPE_CHECKING:
    from fastapi import FastAPI
    from opentelemetry.trace import Span, TracerProvider

# Routes excluded from server-span tracing by default: high-frequency pollers and
# static UI/docs assets, none of which are LLM traffic. Entries are substring-matched
# against the request path (unanchored, so they survive a ``server_root_path`` prefix
# and each entry also covers everything beneath it — e.g. ``/health`` covers
# ``/health/readiness``). Operators override the whole set via the standard
# ``OTEL_PYTHON_FASTAPI_EXCLUDED_URLS`` env var (set "" to trace everything).
_DEFAULT_EXCLUDED_ROUTES: Final = (
    "/health",  # load-balancer liveness/readiness polling
    "/metrics",  # Prometheus scrape (also drops the /model/metrics admin analytics)
    "/litellm-asset-prefix",  # hashed UI asset bundles
    "/_next",  # Next.js static JS/CSS chunks (root-level mount)
    "/ui",  # admin UI single-page app
    "/swagger",  # static Swagger UI assets
    "/docs",  # FastAPI Swagger docs page
    "/redoc",  # FastAPI ReDoc docs page
    "/openapi.json",  # OpenAPI schema
    "favicon",  # /favicon.ico + /get_favicon
    "/.well-known",  # UI config discovery
)
_DEFAULT_EXCLUDED_URLS: Final = ",".join(_DEFAULT_EXCLUDED_ROUTES)

# Passthrough routes are catch-alls (e.g. "/openai/{endpoint:path}"), so the
# default OTel server-span name "{method} {route}" collapses every upstream
# endpoint into "POST /openai/{endpoint:path}". The hook below renames those spans
# to the real request path so each endpoint is distinguishable. Non-catch-all
# routes keep their low-cardinality template name.
PASSTHROUGH_PREFIXES: Final = frozenset(
    {
        "openai",
        "openai_passthrough",
        "anthropic",
        "azure",
        "azure_ai",
        "bedrock",
        "cohere",
        "cursor",
        "gemini",
        "mistral",
        "vllm",
        "vertex_ai",
        "vertex-ai",
        "assemblyai",
        "eu.assemblyai",
        "milvus",
    }
)


class _RenameableSpan(Protocol):
    def is_recording(self) -> bool: ...

    def update_name(self, name: str) -> None: ...

    def set_attribute(self, key: str, value: str) -> None: ...


def _passthrough_span_name_hook(span: "_RenameableSpan | None", scope: Mapping[str, object]) -> None:
    """FastAPI ``server_request_hook``: give passthrough server spans a useful name.

    The instrumentation matches the route at span creation, so both the span name
    and ``http.route`` are set to the catch-all template (``/openai/{endpoint:path}``)
    before this hook runs. Rewrite both to the real request path so each upstream
    endpoint is distinguishable. (The ASGI ``http receive``/``http send`` sub-spans
    can't be renamed from here — their name is captured at creation — so they are
    dropped via ``exclude_spans`` at instrumentation time.)
    """
    try:
        if span is None or not span.is_recording():
            return
        path: Final = str(scope.get("path") or "")
        method: Final = str(scope.get("method") or "")
        first_segment: Final = path.lstrip("/").split("/", 1)[0]
        if first_segment in PASSTHROUGH_PREFIXES:
            span.update_name(f"{method} {path}".strip())
            span.set_attribute("http.route", path)
    except Exception:
        pass


def instrument_fastapi_app(app: "FastAPI", tracer_provider: "TracerProvider | None" = None) -> None:
    """Attach OTel server-span instrumentation to the proxy FastAPI app.

    Safe no-op when the V2 gate is off or ``opentelemetry-instrumentation-fastapi``
    is unavailable. This MUST be called at app-creation time — once the lifespan
    runs, the middleware stack is frozen and middleware added later never runs, so
    a late call is skipped without registering anything.

    No ``TracerProvider`` is passed, so the instrumentation binds to the OTel global
    ``ProxyTracerProvider``; the proxy publishes the real provider as the global
    after config load (see ``proxy_startup_event``), and the proxy delegates to it.
    That way server spans and gen-ai spans share one provider and the same trace.

    Once instrumented, every sampled HTTP response this app sends carries a
    ``traceresponse`` header naming its server span, unless the response already
    relays an upstream's ``traceresponse`` or another response propagator is already
    registered. Other OTel-instrumented apps in the same process get no header. A relayed
    ``traceresponse`` from an upstream that restarted the trace becomes a link on the
    server span. The server span also gets ``litellm.call_id`` from the response's
    ``x-litellm-call-id`` header. ``tracer_provider`` defaults to the global provider.
    """
    try:
        if not is_otel_v2_enabled():
            return
        if app.middleware_stack is not None:
            verbose_logger.debug("Skipping OTel V2 FastAPI instrumentation: the app has already started")
            return

        # Lazy: only the V2-enabled path needs the optional
        # ``opentelemetry-instrumentation-fastapi`` package, which is not part of the
        # base ``litellm[proxy]`` install. Importing it at module top would make
        # ``proxy_server``'s unconditional ``import`` of this module crash when the
        # package is absent, even with the gate off.
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        from opentelemetry.instrumentation.propagators import (
            get_global_response_propagator,
            set_global_response_propagator,  # pyright: ignore[reportUnknownVariableType]  # upstream leaves its parameter unannotated
        )

        from litellm.integrations.otel.trace_response import (
            UpstreamFirstTraceResponsePropagator,
            claim_server_span,
        )

        def server_request_hook(span: "Span | None", scope: Mapping[str, object]) -> None:
            if span is not None:
                claim_server_span(span)
            _passthrough_span_name_hook(span, scope)

        excluded_urls: Final = (
            os.environ.get("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS")
            if "OTEL_PYTHON_FASTAPI_EXCLUDED_URLS" in os.environ
            else _DEFAULT_EXCLUDED_URLS
        )
        FastAPIInstrumentor.instrument_app(
            app,
            tracer_provider=tracer_provider,
            excluded_urls=excluded_urls,
            server_request_hook=server_request_hook,
            # Drop the ASGI "http receive"/"http send" lifecycle sub-spans: they
            # are low-value noise and (for passthrough) carry the catch-all route
            # template in their name, which can't be rewritten from a hook.
            exclude_spans=["receive", "send"],
        )
        if get_global_response_propagator() is None:
            set_global_response_propagator(UpstreamFirstTraceResponsePropagator())
    except Exception as e:
        verbose_logger.debug("Skipping OTel V2 FastAPI instrumentation: %s", e)


def trace_response_expose_headers() -> tuple[str, ...]:
    return ("traceresponse",) if is_otel_v2_enabled() else ()


def trace_response_headers(span: "Span | None") -> Mapping[str, str]:
    """Response headers naming ``span`` for a response built outside the FastAPI instrumentation.

    Starlette runs the ``Exception`` handler outside every middleware, so its 500 never reaches the response
    propagator and the handler has to add these itself.
    """
    if span is None or not is_otel_v2_enabled():
        return MappingProxyType({})
    try:
        from opentelemetry import trace
        from opentelemetry.instrumentation.asgi import (  # pyright: ignore[reportMissingTypeStubs]  # the ASGI instrumentation ships no py.typed marker
            asgi_getter,
            asgi_setter,
        )
        from opentelemetry.instrumentation.propagators import get_global_response_propagator

        from litellm.integrations.otel.trace_response import claim_server_span
    except ImportError:
        return MappingProxyType({})
    claim_server_span(span)
    propagator: Final = get_global_response_propagator()  # pyright: ignore[reportUnknownVariableType]  # upstream leaves inject unannotated
    if propagator is None:
        return MappingProxyType({})
    start: Final[dict[str, object]] = {"type": "http.response.start"}  # mutable-ok: the propagator writes into it
    propagator.inject(start, context=trace.set_span_in_context(span), setter=asgi_setter)  # pyright: ignore[reportUnknownMemberType]  # upstream leaves inject unannotated
    return MappingProxyType({key: ", ".join(asgi_getter.get(start, key) or ()) for key in asgi_getter.keys(start)})  # pyright: ignore[reportUnknownMemberType]  # the ASGI instrumentation ships no py.typed marker
