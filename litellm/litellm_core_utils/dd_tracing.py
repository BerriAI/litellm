"""
Handles Tracing on DataDog Traces.

If the ddtrace package is not installed, the tracer will be a no-op.
"""

from contextlib import contextmanager

from litellm.secret_managers.main import get_secret_bool


def _should_use_dd_tracer():
    """
    Returns True if `USE_DDTRACE` is set to True in .env
    """
    return get_secret_bool("USE_DDTRACE", False) is True


# Try to import ddtrace, set is_dd_enabled based on both import success and user preference
is_dd_enabled = False
datadog_tracer = None
try:
    from ddtrace import tracer as datadog_tracer

    is_dd_enabled = _should_use_dd_tracer()
except ImportError:
    is_dd_enabled = False

# If ddtrace is not available or not enabled, create a null implementation
if not is_dd_enabled:

    @contextmanager
    def null_tracer(name, **kwargs):
        class NullSpan:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def finish(self):
                pass

        yield NullSpan()

    class NullTracer:
        def trace(self, name, **kwargs):
            class NullSpan:
                def __enter__(self):
                    return self

                def __exit__(self, *args):
                    pass

                def finish(self):
                    pass

            return NullSpan()

        def wrap(self, name=None, **kwargs):
            def decorator(f):
                return f

            return decorator

    datadog_tracer = NullTracer()

# Export the tracer instance
tracer = datadog_tracer
