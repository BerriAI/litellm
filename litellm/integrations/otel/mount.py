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
from typing import Any

from litellm._logging import verbose_logger
from litellm.integrations.otel.model.config import is_otel_v2_enabled

# Routes excluded from server-span tracing by default: high-frequency pollers and
# static UI/docs assets, none of which are LLM traffic. Entries are substring-matched
# against the request path (unanchored, so they survive a ``server_root_path`` prefix
# and each entry also covers everything beneath it — e.g. ``/health`` covers
# ``/health/readiness``). Operators override the whole set via the standard
# ``OTEL_PYTHON_FASTAPI_EXCLUDED_URLS`` env var (set "" to trace everything).
_DEFAULT_EXCLUDED_ROUTES = (
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
_DEFAULT_EXCLUDED_URLS = ",".join(_DEFAULT_EXCLUDED_ROUTES)

# Passthrough routes are catch-alls (e.g. "/openai/{endpoint:path}"), so the
# default OTel server-span name "{method} {route}" collapses every upstream
# endpoint into "POST /openai/{endpoint:path}". The hook below renames those spans
# to the real request path so each endpoint is distinguishable. Non-catch-all
# routes keep their low-cardinality template name.
PASSTHROUGH_PREFIXES = frozenset(
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


def _passthrough_span_name_hook(span: Any, scope: dict) -> None:
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
        path = scope.get("path") or ""
        method = scope.get("method") or ""
        first_segment = path.lstrip("/").split("/", 1)[0]
        if first_segment in PASSTHROUGH_PREFIXES:
            span.update_name(f"{method} {path}".strip())
            span.set_attribute("http.route", path)
    except Exception:
        pass


def instrument_fastapi_app(app: Any) -> None:
    """Attach OTel server-span instrumentation to the proxy FastAPI app.

    Safe no-op when the V2 gate is off or ``opentelemetry-instrumentation-fastapi``
    is unavailable. This MUST be called at app-creation time — once the lifespan
    runs, the middleware stack is frozen and ``instrument_app`` raises "Cannot add
    middleware after an application has started".

    No ``TracerProvider`` is passed, so the instrumentation binds to the OTel global
    ``ProxyTracerProvider``; the proxy publishes the real provider as the global
    after config load (see ``proxy_startup_event``), and the proxy delegates to it.
    That way server spans and gen-ai spans share one provider and the same trace.
    """
    if not is_otel_v2_enabled():
        return

    # Lazy: only the V2-enabled path needs the optional
    # ``opentelemetry-instrumentation-fastapi`` package. Importing it at module top
    # would make ``proxy_server``'s unconditional ``import`` of this module crash when
    # the package is absent, even with the gate off. When V2 IS on, a missing package
    # is a real misconfiguration -- without the server span the trace has no root and
    # admin-owned destination traces are orphaned -- so it must be loud, not silent.
    try:
        from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    except ImportError:
        verbose_logger.warning(
            "LITELLM_OTEL_V2 is enabled but 'opentelemetry-instrumentation-fastapi' "
            "is not installed. The FastAPI server span will not be created, so traces "
            "exported to admin-owned destinations will be missing their root span "
            "(orphaned children). Install 'opentelemetry-instrumentation-fastapi'."
        )
        return

    try:
        excluded_urls = (
            os.environ.get("OTEL_PYTHON_FASTAPI_EXCLUDED_URLS")
            if "OTEL_PYTHON_FASTAPI_EXCLUDED_URLS" in os.environ
            else _DEFAULT_EXCLUDED_URLS
        )
        FastAPIInstrumentor.instrument_app(
            app,
            excluded_urls=excluded_urls,
            server_request_hook=_passthrough_span_name_hook,
            # Drop the ASGI "http receive"/"http send" lifecycle sub-spans: they
            # are low-value noise and (for passthrough) carry the catch-all route
            # template in their name, which can't be rewritten from a hook.
            exclude_spans=["receive", "send"],
        )
    except Exception as e:
        verbose_logger.warning("OTel V2 FastAPI instrumentation failed: %s", e)
