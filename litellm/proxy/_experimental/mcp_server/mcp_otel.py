"""OpenTelemetry instrumentation helpers for MCP flows.

These helpers emit spans for MCP operations (``list_tools``, ``call_tool``,
``execute_tool`` and the REST equivalents) into the OpenTelemetry tracer
that is configured on the proxy's ``open_telemetry_logger`` callback (or
any ``OpenTelemetry`` instance registered in ``litellm.callbacks``).

When OTEL is not configured (or the SDK is not installed), every helper
degrades to a zero-cost no-op so callers can wrap MCP code paths
unconditionally without conditional checks at every call site.

Design notes
------------

* ``mcp_span`` is a regular (sync) ``contextlib`` context manager. The
  underlying ``tracer.start_as_current_span`` is itself a sync context
  manager that uses ``contextvars`` for span propagation; this works
  correctly inside ``async def`` callers as long as the ``with`` block
  brackets the entire await chain that should be parented under the span.

* Span names live in the ``mcp.*`` namespace so that operators can write
  dashboards/alerts without conflicting with the existing
  ``litellm_request`` / ``gen_ai.*`` spans emitted by the LLM call path.

* Attribute values are coerced through ``OpenTelemetry.safe_set_attribute``
  to keep the same primitive-only contract the rest of the codebase uses.
"""

import contextlib
import functools
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, Optional

import litellm
from litellm._logging import verbose_logger

if TYPE_CHECKING:
    from litellm.integrations.opentelemetry import OpenTelemetry as _OpenTelemetry

    OpenTelemetryType = _OpenTelemetry
else:
    OpenTelemetryType = Any


# ---------------------------------------------------------------------------
# Span names — keep stable; observability dashboards/alerts reference them.
# ---------------------------------------------------------------------------
SPAN_MCP_LIST_TOOLS = "mcp.list_tools"
SPAN_MCP_EXECUTE_TOOL = "mcp.execute_tool"
SPAN_MCP_PROTOCOL_LIST_TOOLS = "mcp.protocol.list_tools"
SPAN_MCP_PROTOCOL_CALL_TOOL = "mcp.protocol.call_tool"
SPAN_MCP_REST_LIST_TOOLS = "mcp.rest.list_tools"
SPAN_MCP_REST_CALL_TOOL = "mcp.rest.call_tool"


# ---------------------------------------------------------------------------
# Attribute keys
# ---------------------------------------------------------------------------
ATTR_MCP_OPERATION = "mcp.operation"
ATTR_MCP_SERVER_NAME = "mcp.server.name"
ATTR_MCP_SERVER_ID = "mcp.server.id"
ATTR_MCP_TOOL_NAME = "mcp.tool.name"
ATTR_MCP_TOOL_COUNT = "mcp.tool.count"
ATTR_MCP_USER_ID = "mcp.user.id"
ATTR_MCP_TRANSPORT = "mcp.transport"
ATTR_MCP_ALLOWED_SERVER_COUNT = "mcp.allowed_server_count"
ATTR_MCP_REQUESTED_SERVER_ID = "mcp.requested_server_id"
ATTR_MCP_AUTH_TYPE = "mcp.auth.type"
ATTR_MCP_IS_BYOK = "mcp.server.is_byok"
ATTR_MCP_RESULT_IS_ERROR = "mcp.result.is_error"


def _get_active_otel_logger() -> Optional[OpenTelemetryType]:
    """Return the proxy's configured ``OpenTelemetry`` logger, or ``None``.

    Checks (in order):

    1. ``litellm.callbacks`` for an ``OpenTelemetry`` instance — covers the
       case where OTEL is registered via ``litellm.callbacks.append(...)``
       (e.g. SDK usage, tests).
    2. ``litellm.proxy.proxy_server.open_telemetry_logger`` — covers the
       proxy config path where OTEL is enabled via
       ``litellm_settings.callbacks: ["otel"]``.

    Imports are local because importing ``proxy_server`` pulls in a large
    chunk of the proxy graph, which we want to avoid in plain SDK usage.
    """
    try:
        from litellm.integrations.opentelemetry import OpenTelemetry
    except Exception:
        return None

    for callback in list(litellm.callbacks or []):
        if isinstance(callback, OpenTelemetry):
            return callback

    try:
        from litellm.proxy.proxy_server import open_telemetry_logger
    except Exception:
        return None

    if isinstance(open_telemetry_logger, OpenTelemetry):
        return open_telemetry_logger
    return None


def _coerce_attributes(
    otel: OpenTelemetryType, span: Any, attributes: Optional[Dict[str, Any]]
) -> None:
    """Apply ``attributes`` to ``span`` via ``safe_set_attribute``.

    ``None`` values are skipped — OTEL rejects ``None`` as an attribute
    value, and serialising ``None`` to the string ``"None"`` is misleading
    in dashboards.
    """
    if not attributes:
        return
    for key, value in attributes.items():
        if value is None:
            continue
        try:
            otel.safe_set_attribute(span=span, key=key, value=value)
        except Exception as exc:  # pragma: no cover - defensive
            verbose_logger.debug(
                "mcp_otel: failed to set attribute %s on span %s: %s",
                key,
                getattr(span, "name", "<span>"),
                exc,
            )


@contextlib.contextmanager
def mcp_span(
    name: str,
    attributes: Optional[Dict[str, Any]] = None,
) -> Iterator[Optional[Any]]:
    """Context manager that emits an OTEL span for an MCP operation.

    The span uses the proxy's OTEL tracer when configured; otherwise this
    is a zero-cost no-op so callers can wrap MCP code paths unconditionally.

    On exception, the span is marked ``StatusCode.ERROR``, the exception is
    recorded, and the original exception is re-raised. On success the span
    is marked ``StatusCode.OK``.

    The yielded value is the underlying OTEL span (so the caller can attach
    more attributes mid-way) or ``None`` when OTEL is not configured.

    Args:
        name: Span name (use one of the ``SPAN_MCP_*`` constants).
        attributes: Initial attributes for the span. Values are coerced
            via ``OpenTelemetry.safe_set_attribute``; ``None`` values are
            skipped.
    """
    otel = _get_active_otel_logger()
    if otel is None or getattr(otel, "tracer", None) is None:
        yield None
        return

    try:
        from opentelemetry.trace import Status, StatusCode
    except Exception:  # pragma: no cover - SDK not installed
        yield None
        return

    # ``start_as_current_span`` is the standard OTEL pattern for creating
    # a span and binding it to the current context so any nested
    # ``mcp_span`` (or other OTEL spans inside the same async call) is
    # automatically parented under it.
    with otel.tracer.start_as_current_span(name) as span:
        _coerce_attributes(otel, span, attributes)
        try:
            yield span
        except Exception as exc:
            try:
                span.record_exception(exc)
            except Exception:  # pragma: no cover - defensive
                pass
            try:
                span.set_status(Status(StatusCode.ERROR, str(exc)))
            except Exception:  # pragma: no cover - defensive
                pass
            raise
        else:
            try:
                span.set_status(Status(StatusCode.OK))
            except Exception:  # pragma: no cover - defensive
                pass


def set_mcp_span_attribute(
    span: Optional[Any],
    key: str,
    value: Any,
    _otel: Optional["OpenTelemetryType"] = None,
) -> None:
    """Set an attribute on a span yielded by :func:`mcp_span`.

    Safe to call when ``span`` is ``None`` (no-op when OTEL is not
    configured) or ``value`` is ``None``.

    Pass ``_otel`` when you already hold the logger reference (e.g. after
    calling :func:`set_mcp_span_attributes` in the same code block) to avoid
    a redundant callback-list scan.
    """
    if span is None or value is None:
        return
    otel = _otel or _get_active_otel_logger()
    if otel is None:
        return
    try:
        otel.safe_set_attribute(span=span, key=key, value=value)
    except Exception as exc:  # pragma: no cover - defensive
        verbose_logger.debug(
            "mcp_otel.set_mcp_span_attribute: failed for %s: %s", key, exc
        )


def set_mcp_span_attributes(
    span: Optional[Any],
    attributes: Optional[Dict[str, Any]],
    _otel: Optional["OpenTelemetryType"] = None,
) -> Optional["OpenTelemetryType"]:
    """Bulk variant of :func:`set_mcp_span_attribute`.

    Returns the resolved OTEL logger so callers that immediately follow with
    a :func:`set_mcp_span_attribute` call can pass it via ``_otel=`` and
    skip the second lookup.
    """
    if span is None or not attributes:
        return None
    otel = _otel or _get_active_otel_logger()
    if otel is None:
        return None
    _coerce_attributes(otel, span, attributes)
    return otel


def with_mcp_span(
    span_name: str,
    attribute_factory: Optional[Callable[..., Dict[str, Any]]] = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that wraps an async function in an :func:`mcp_span`.

    Used to instrument FastAPI handlers without changing their bodies (and
    therefore the indentation of ~150 lines of existing code).
    ``functools.wraps`` preserves ``__wrapped__`` so FastAPI's signature
    introspection (used for ``Depends``/``Query`` parameter resolution)
    still sees the original handler's parameters.

    ``attribute_factory`` receives the same positional / keyword arguments
    as the wrapped callable and returns the attribute dict to attach to
    the span. It must tolerate ``None`` and missing keys, since handlers
    may be invoked in tests with partial kwargs.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            attrs: Optional[Dict[str, Any]] = None
            if attribute_factory is not None:
                try:
                    attrs = attribute_factory(*args, **kwargs)
                except Exception as exc:  # pragma: no cover - defensive
                    # Never let attribute extraction break the request.
                    verbose_logger.debug(
                        "with_mcp_span: attribute_factory raised: %s", exc
                    )
                    attrs = None
            with mcp_span(span_name, attributes=attrs):
                return await fn(*args, **kwargs)

        return wrapper

    return decorator
