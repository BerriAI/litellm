"""
Handles Tracing on DataDog Traces.

If the ddtrace package is not installed, the tracer will be a no-op.
"""

from contextlib import contextmanager

from litellm.secret_managers.main import get_secret_bool


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
        def decorator(f):
            return f

        return decorator


def _should_use_dd_tracer():
    """Returns True if `USE_DDTRACE` is set to True in .env"""
    return get_secret_bool("USE_DDTRACE", False) is True


# Initialize tracer
should_use_dd_tracer = _should_use_dd_tracer()
tracer = None

if should_use_dd_tracer:
    try:
        from ddtrace import tracer as dd_tracer

        tracer = dd_tracer
    except ImportError:
        tracer = NullTracer()
else:
    tracer = NullTracer()
