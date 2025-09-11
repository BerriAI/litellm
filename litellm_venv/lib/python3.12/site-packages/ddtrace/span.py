from ddtrace._trace.span import Span  # noqa: F401
from ddtrace.internal.utils.deprecations import DDTraceDeprecationWarning
from ddtrace.vendor.debtcollector import deprecate


deprecate(
    "The span module is deprecated and will be moved.",
    message="A new span interface will be provided by the trace sub-package.",
    category=DDTraceDeprecationWarning,
)
