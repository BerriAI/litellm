from contextlib import contextmanager

try:
    from ddtrace import tracer as dd_tracer

    has_ddtrace = True
except ImportError:
    has_ddtrace = False

    @contextmanager
    def null_tracer(name, **kwargs):
        class NullSpan:
            def finish(self):
                pass

        yield NullSpan()

    class NullTracer:
        def trace(self, name, **kwargs):
            class NullSpan:
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
