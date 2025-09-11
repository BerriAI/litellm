from ddtrace._trace import trace_handlers  # noqa: F401
from ddtrace._trace._span_link import SpanLink  # noqa: F401
from ddtrace.internal.utils.deprecations import DDTraceDeprecationWarning
from ddtrace.vendor.debtcollector import deprecate


deprecate(
    "The tracing module is deprecated and will be moved.",
    message="A new interface will be provided by the _trace sub-package.",
    category=DDTraceDeprecationWarning,
)
