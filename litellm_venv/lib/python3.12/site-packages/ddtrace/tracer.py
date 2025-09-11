from ddtrace._trace.tracer import Tracer  # noqa: F401
from ddtrace.internal.utils.deprecations import DDTraceDeprecationWarning
from ddtrace.vendor.debtcollector import deprecate


deprecate(
    "The ddtrace.tracer module is deprecated and will be removed.",
    message="A new interface will be provided by the trace sub-package.",
    category=DDTraceDeprecationWarning,
)
