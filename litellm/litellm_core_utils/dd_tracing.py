"""
Handles Tracing on DataDog Traces.

If the ddtrace package is not installed, the tracer will be a no-op.
"""

from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Optional, Union

from litellm.secret_managers.main import get_secret_bool

if TYPE_CHECKING:
    from ddtrace.tracer import Tracer as DD_TRACER
else:
    DD_TRACER = Any


class NullSpan:
    """A no-op span implementation."""

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def finish(self):
        pass


@contextmanager
def null_tracer(name, **kwargs):
    """Context manager that yields a no-op span."""
    yield NullSpan()


class NullTracer:
    """A no-op tracer implementation."""

    def trace(self, name, **kwargs):
        return NullSpan()

    def wrap(self, name=None, **kwargs):
        # If called with no arguments (as @tracer.wrap())
        if callable(name):
            return name

        # If called with arguments (as @tracer.wrap(name="something"))
        def decorator(f):
            return f

        return decorator


def _should_use_dd_tracer():
    """Returns True if `USE_DDTRACE` is set to True in .env"""
    return get_secret_bool("USE_DDTRACE", False) is True


def _should_use_dd_profiler():
    """Returns True if `USE_DDPROFILER` is set to True in .env"""
    return get_secret_bool("USE_DDPROFILER", False) is True


# Initialize tracer
should_use_dd_tracer = _should_use_dd_tracer()
tracer: Union[NullTracer, DD_TRACER] = NullTracer()
# We need to ensure tracer is never None and always has the required methods
if should_use_dd_tracer:
    try:
        from ddtrace import tracer as dd_tracer

        # Define the type to match what's expected by the code using this module
        tracer = dd_tracer
    except ImportError:
        tracer = NullTracer()
else:
    tracer = NullTracer()


def get_active_span() -> Optional[Any]:
    """
    Return the active Datadog span, checking current span first and then root span.
    """
    try:
        current_span_fn = getattr(tracer, "current_span", None)
        if callable(current_span_fn):
            current_span = current_span_fn()
            if current_span is not None:
                return current_span

        current_root_span_fn = getattr(tracer, "current_root_span", None)
        if callable(current_root_span_fn):
            return current_root_span_fn()
    except Exception:
        return None
    return None


def set_active_span_tag(tag_key: str, tag_value: str) -> bool:
    """
    Best-effort helper to set a tag on the active Datadog span.

    Returns:
        bool: True if a span tag was set, False otherwise.
    """
    if not tag_key or tag_value is None:
        return False

    span = get_active_span()
    if span is None:
        return False

    try:
        if hasattr(span, "set_tag_str"):
            span.set_tag_str(tag_key, str(tag_value))
            return True
        if hasattr(span, "set_tag"):
            span.set_tag(tag_key, str(tag_value))
            return True
    except Exception:
        return False
    return False
