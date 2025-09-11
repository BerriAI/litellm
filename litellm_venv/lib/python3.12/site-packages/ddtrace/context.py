from ddtrace._trace.context import Context  # noqa: F401
from ddtrace.internal.utils.deprecations import DDTraceDeprecationWarning
from ddtrace.vendor.debtcollector import deprecate


deprecate(
    "The ddtrace.context module is deprecated and will be removed from the public API.",
    message="Context should be imported from the ddtrace.trace package",
    category=DDTraceDeprecationWarning,
)
