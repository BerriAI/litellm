from contextlib import contextmanager

try:
    from ddtrace import tracer as dd_tracer

    has_ddtrace = True
except ImportError:
    has_ddtrace = False

    @contextmanager
    def null_tracer(name, **kwargs):
        yield

    class NullTracer:
        def trace(self, name, **kwargs):
            return null_tracer(name, **kwargs)

    dd_tracer = NullTracer()

# Export the tracer instance
tracer = dd_tracer
