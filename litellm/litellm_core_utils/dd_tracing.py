"""
Handles Tracing on DataDog Traces.

If the ddtrace package is not installed, the tracer will be a no-op.
"""

from contextlib import contextmanager

try:
    from ddtrace import tracer as dd_tracer

    has_ddtrace = True
except ImportError:
    has_ddtrace = False

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

    dd_tracer = NullTracer()

# Export the tracer instance
tracer = dd_tracer
