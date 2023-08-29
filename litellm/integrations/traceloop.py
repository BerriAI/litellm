import sys

class TraceloopLogger:
    def __init__(self):
        from traceloop.tracing import Tracer
        self.tracer = Tracer.init(app_name=sys.argv[0])
