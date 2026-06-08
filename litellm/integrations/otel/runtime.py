"""SDK-free entrypoints for proxy-core call sites (auth, identity, …).

Proxy code may run without the OpenTelemetry SDK installed, so it must not import
``litellm.integrations.otel.logger`` (which imports the SDK at module scope) at
module load. These wrappers import it lazily and no-op when the SDK is absent or
V2 is not the active logger — so a call site can wrap a request phase, seed
identity, or decorate an async function with ``@traced`` unconditionally.
"""

import functools
import inspect
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Mapping, Optional

from litellm.integrations.otel.model.spans import SpanRole


@contextmanager
def phase_span(name: str) -> "Iterator[Any]":
    """Run a request phase inside a live active span so its DB/service calls nest.

    Yields ``None`` (a plain no-op) when the OTel SDK is unavailable or V2 is not
    the active logger.
    """
    try:
        from litellm.integrations.otel.logger import phase_span as _phase_span
    except Exception:
        yield None
        return
    with _phase_span(name) as span:
        yield span


def seed_request_identity(user_api_key_dict: Any, model: Any = None) -> None:
    """Seed request-identity Baggage at the auth boundary (no-op without V2)."""
    try:
        from litellm.integrations.otel.logger import (
            seed_request_identity as _seed_request_identity,
        )
    except Exception:
        return
    _seed_request_identity(user_api_key_dict, model=model)


def _apply_span_attrs(span: Any, attrs: Optional[Mapping[str, Any]]) -> None:
    if span is None or not attrs:
        return
    for key, value in attrs.items():
        if value is None:
            continue
        try:
            span.set_attribute(key, value)
        except Exception:
            pass


def _resolve_attrs(
    builder: Optional[Callable[..., Mapping[str, Any]]],
    *,
    args: tuple,
    kwargs: dict,
    result: Any,
) -> Mapping[str, Any]:
    if builder is None:
        return {}
    try:
        sig = inspect.signature(builder)
        accepts_result = "result" in sig.parameters or any(
            p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()
        )
        if accepts_result:
            payload = builder(*args, result=result, **kwargs)
        else:
            payload = builder(result)
        return payload or {}
    except Exception:
        return {}


def traced(
    span_name: str,
    *,
    role: SpanRole = SpanRole.SERVICE,
    attrs: Optional[Callable[..., Mapping[str, Any]]] = None,
) -> Callable:
    """Wrap an async function so it runs inside an OTel v2 span.

    ``SpanRole.SERVICE`` opens a phase span under the active root proxy request
    span; nested DB/service calls become its children. ``SpanRole.DB_CALL`` (and
    any other non-SERVICE role) does not open a new span — it just attaches
    attributes to the current span, so we don't double-count for code paths
    that already produce a DB_CALL via service-logger plumbing.

    ``attrs`` is an optional callback that receives the function's return value
    (and the wrapped function's args/kwargs via ``**kwargs`` if it accepts
    them) and returns a mapping of span attributes to set on success. Any
    exception in ``attrs`` is swallowed — instrumentation must never break the
    request.
    """

    def _decorator(func: Callable) -> Callable:
        if not inspect.iscoroutinefunction(func):
            raise TypeError(
                f"@traced requires an async function; got {func!r}"
            )

        @functools.wraps(func)
        async def _wrapper(*args, **kwargs):
            if role is SpanRole.SERVICE:
                with phase_span(span_name) as span:
                    result = await func(*args, **kwargs)
                    _apply_span_attrs(
                        span,
                        _resolve_attrs(attrs, args=args, kwargs=kwargs, result=result),
                    )
                    return result

            result = await func(*args, **kwargs)
            try:
                from opentelemetry import trace as _otel_trace

                current_span = _otel_trace.get_current_span()
            except Exception:
                current_span = None
            _apply_span_attrs(
                current_span,
                _resolve_attrs(attrs, args=args, kwargs=kwargs, result=result),
            )
            return result

        return _wrapper

    return _decorator
