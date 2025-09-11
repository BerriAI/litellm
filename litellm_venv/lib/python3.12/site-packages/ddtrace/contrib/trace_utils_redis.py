from ddtrace.contrib.redis_utils import determine_row_count
from ddtrace.contrib.redis_utils import stringify_cache_args
from ddtrace.internal.utils.deprecations import DDTraceDeprecationWarning
from ddtrace.vendor.debtcollector import deprecate


deprecate(
    "The ddtrace.contrib.trace_utils_redis module is deprecated and will be removed.",
    message="A new interface will be provided by the ddtrace.contrib.redis_utils module",
    category=DDTraceDeprecationWarning,
)


format_command_args = stringify_cache_args


def determine_row_count(redis_command, span, result):  # noqa: F811
    determine_row_count(redis_command=redis_command, result=result)
