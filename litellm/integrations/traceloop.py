import sys
from traceloop.tracing import Tracer


class TraceloopLogger:
    def __init__(self):
        self.tracer = Tracer.init(app_name=sys.argv[0])
