"""OpenTelemetry tracing for MCP server operations.

LIT-3201: instrument the proxy's MCP entry points with OTEL spans so
operators can see ``mcp.tool.call``, ``mcp.list_tools``, ``mcp.list_prompts``,
``mcp.list_resources``, ``mcp.list_resource_templates``, ``mcp.get_prompt``,
and ``mcp.read_resource`` in their tracing backend with consistent
``mcp.*`` attributes.

The helper is intentionally defensive:
    * It returns a no-op context when the proxy's ``open_telemetry_logger``
      is not initialized, so the call sites can wrap operations
      unconditionally.
    * It catches all exceptions raised by the tracing code itself so
      observability never interferes with the MCP request path.

The exception is still re-raised by the surrounding ``yield`` to preserve
caller-visible behaviour; ERROR status + ``record_exception`` are set on
the span before re-raising. Client-side ``HTTPException`` 4xx outcomes
are NOT elevated to ``StatusCode.ERROR`` because they are expected
caller-side outcomes and would otherwise inflate operator error-rate
dashboards. The exception is still recorded as a span event and the HTTP
status code is recorded as ``mcp.http.status_code``.
"""

from __future__ import annotations

import contextlib
from typing import Any, AsyncIterator, Dict, Optional

from litellm._logging import verbose_logger

# Tracer name used to identify MCP spans emitted by the proxy. Backends
# can filter on ``otel.library.name == "litellm.mcp"`` to scope dashboards
# to MCP traffic.
MCP_OTEL_TRACER_NAME = "litellm.mcp"


def _get_mcp_tracer() -> Optional[Any]:
    """Return the OpenTelemetry tracer used by the active proxy logger.

    Resolution order:
        1. The ``open_telemetry_logger`` global on
           ``litellm.proxy.proxy_server`` (populated when the proxy boots
           with ``otel`` in ``callbacks``).
        2. ``None`` when the proxy isn't running with OTEL enabled.

    Returning ``None`` makes :func:`mcp_otel_span` short-circuit to a no-op,
    which is what we want in non-proxy contexts (unit tests that import the
    MCP server module without instantiating the proxy, custom deployments
    that don't enable OTEL, etc.).
    """
    try:
        from litellm.proxy import proxy_server  # local import to avoid cycles
    except Exception:
        return None

    otel_logger = getattr(proxy_server, "open_telemetry_logger", None)
    if otel_logger is None:
        return None
    return getattr(otel_logger, "tracer", None)


def _safe_set(span: Any, key: str, value: Any) -> None:
    """Set a span attribute, swallowing any exception from the tracing SDK.

    Never raise: an observability bug must not break an MCP call.
    """
    try:
        if value is None:
            return
        if isinstance(value, (str, bool, int, float)):
            span.set_attribute(key, value)
        else:
            span.set_attribute(key, str(value))
    except Exception as e:  # pragma: no cover - defensive
        verbose_logger.debug("mcp_otel_span: failed to set %s: %s", key, e)


def _sanitize_resource_uri(uri: str) -> str:
    """Return uri with credentials, query, and fragment stripped.

    Resource URIs come straight from the MCP client (caller-provided) and may
    legitimately embed presigned tokens, user:password@ credentials, or
    short-lived fragments. Spans get persisted to a tracing backend (often
    long-lived; sometimes shared with vendors), so we MUST NOT export raw
    URLs. We keep scheme + sanitised netloc (host[:port], no userinfo) +
    path so the trace is still useful for debugging which resource was
    accessed, while dropping every component that could carry a secret.
    """
    try:
        from urllib.parse import urlsplit, urlunsplit

        parts = urlsplit(uri)
        host = parts.hostname or ""
        port = parts.port
        netloc = host if port is None else f"{host}:{port}"
        if not parts.scheme and not netloc:
            # Opaque / non-URL value (e.g. plain identifier) - return as-is.
            return uri
        # urlunsplit needs all 5 components; drop query + fragment.
        return urlunsplit((parts.scheme, netloc, parts.path, "", ""))
    except Exception:  # pragma: no cover - defensive
        return uri


def _http_status_code(exc: BaseException) -> Optional[int]:
    """Return a 3-digit HTTP status code if ``exc`` is an HTTPException-like.

    Recognises both Starlette / FastAPI ``HTTPException`` and the upstream
    MCP server's wrapper exceptions. Returns ``None`` when no integer
    status_code attribute is present.
    """
    code = getattr(exc, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


@contextlib.asynccontextmanager
async def mcp_otel_span(  # noqa: PLR0915
    operation: str,
    *,
    server_name: Optional[str] = None,
    tool_name: Optional[str] = None,
    prompt_name: Optional[str] = None,
    resource_uri: Optional[str] = None,
    arguments: Optional[Dict[str, Any]] = None,
    extra_attributes: Optional[Dict[str, Any]] = None,
) -> AsyncIterator[Optional[Any]]:
    """Async context manager that emits one OTEL span per MCP operation.

    Yields the active span (or ``None`` if OTEL isn't configured). The
    span is attached to the active OTEL context for the duration of the
    block so any child spans created inside the block (e.g. auto-
    instrumented HTTP calls to upstream MCP servers) parent to the MCP
    span. The span is detached + ended when the block exits.

    On exception the span is marked ``ERROR``, ``record_exception`` is
    called, and ``mcp.error.type`` is set. Client-side ``HTTPException``
    4xx outcomes are recorded but stay at ``StatusCode.OK`` to avoid
    inflating error-rate metrics for expected outcomes
    (auth-denied, malformed argument, etc.).

    The span name is ``mcp.<operation>``. Standard attributes:

    * ``mcp.operation`` - the bare operation name (``tool.call``,
      ``list_tools`` ...).
    * ``mcp.server.name`` - upstream server, when known.
    * ``mcp.tool.name`` / ``mcp.prompt.name`` / ``mcp.resource.uri`` -
      object-specific identifier (unprefixed when available).
    * ``mcp.arguments_count`` - ``len(arguments)`` when ``arguments`` is
      sized. We intentionally do NOT serialize argument values to spans
      to avoid leaking secrets into the tracing backend.
    """
    tracer = _get_mcp_tracer()
    if tracer is None:
        yield None
        return

    try:
        from opentelemetry import context as otel_context
        from opentelemetry import trace as otel_trace
        from opentelemetry.trace import Status, StatusCode
    except Exception:
        # opentelemetry isn't installed even though a logger object exists -
        # treat as no-op rather than crashing the MCP call.
        yield None
        return

    span_name = "mcp." + operation
    try:
        span = tracer.start_span(span_name)
    except Exception as e:  # pragma: no cover - defensive
        verbose_logger.debug("mcp_otel_span: tracer.start_span failed: %s", e)
        yield None
        return

    # Attach the new span to the active OTEL context so child spans
    # (e.g. HTTP auto-instrumentation calling out to upstream MCP servers)
    # parent to this span instead of to whatever was current before the
    # block. The detach token is restored in the finally branch so we
    # never leak the context slot.
    try:
        ctx_token = otel_context.attach(otel_trace.set_span_in_context(span))
    except Exception as e:  # pragma: no cover - defensive
        verbose_logger.debug("mcp_otel_span: context.attach failed: %s", e)
        ctx_token = None

    try:
        _safe_set(span, "mcp.operation", operation)
        if server_name:
            _safe_set(span, "mcp.server.name", server_name)
        if tool_name:
            _safe_set(span, "mcp.tool.name", tool_name)
        if prompt_name:
            _safe_set(span, "mcp.prompt.name", prompt_name)
        if resource_uri is not None:
            _safe_set(
                span, "mcp.resource.uri", _sanitize_resource_uri(str(resource_uri))
            )
        if arguments is not None:
            try:
                _safe_set(span, "mcp.arguments_count", len(arguments))
            except TypeError:  # pragma: no cover - defensive
                pass
        if extra_attributes:
            for k, v in extra_attributes.items():
                _safe_set(span, k, v)

        try:
            yield span
        except BaseException as exc:
            try:
                span.record_exception(exc)
                _safe_set(span, "mcp.error.type", type(exc).__name__)
                http_status = _http_status_code(exc)
                if http_status is not None:
                    _safe_set(span, "mcp.http.status_code", http_status)
                    # OTEL convention: client errors (4xx) are expected
                    # caller-side outcomes; only treat 5xx + non-HTTP
                    # exceptions as span ERRORs so dashboards don't
                    # conflate "key not authorised" with "upstream MCP
                    # server crashed".
                    if 400 <= http_status < 500:
                        # Leave status UNSET (default) - operators can
                        # still see the recorded exception event and the
                        # ``mcp.error.type`` attribute.
                        pass
                    else:
                        span.set_status(Status(StatusCode.ERROR, str(exc)[:256]))
                else:
                    span.set_status(Status(StatusCode.ERROR, str(exc)[:256]))
            except Exception:  # pragma: no cover - defensive
                pass
            raise
        else:
            try:
                span.set_status(Status(StatusCode.OK))
            except Exception:  # pragma: no cover - defensive
                pass
    finally:
        # Detach context BEFORE ending the span - otherwise the OTEL
        # context slot keeps a reference to the now-ended span for the
        # rest of the request and confuses downstream child spans.
        if ctx_token is not None:
            try:
                otel_context.detach(ctx_token)
            except Exception:  # pragma: no cover - defensive
                pass
        try:
            span.end()
        except Exception:  # pragma: no cover - defensive
            pass
