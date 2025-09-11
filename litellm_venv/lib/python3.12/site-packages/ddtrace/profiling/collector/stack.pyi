import typing

import ddtrace
from ddtrace.profiling import collector

class StackCollector(collector.PeriodicCollector):
    tracer: typing.Optional[ddtrace.Tracer]
