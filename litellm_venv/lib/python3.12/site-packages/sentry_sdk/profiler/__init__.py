from sentry_sdk.profiler.continuous_profiler import start_profiler, stop_profiler
from sentry_sdk.profiler.transaction_profiler import (
    MAX_PROFILE_DURATION_NS,
    PROFILE_MINIMUM_SAMPLES,
    Profile,
    Scheduler,
    ThreadScheduler,
    GeventScheduler,
    has_profiling_enabled,
    setup_profiler,
    teardown_profiler,
)
from sentry_sdk.profiler.utils import (
    DEFAULT_SAMPLING_FREQUENCY,
    MAX_STACK_DEPTH,
    get_frame_name,
    extract_frame,
    extract_stack,
    frame_id,
)

__all__ = [
    "start_profiler",
    "stop_profiler",
    # DEPRECATED: The following was re-exported for backwards compatibility. It
    # will be removed from sentry_sdk.profiler in a future release.
    "MAX_PROFILE_DURATION_NS",
    "PROFILE_MINIMUM_SAMPLES",
    "Profile",
    "Scheduler",
    "ThreadScheduler",
    "GeventScheduler",
    "has_profiling_enabled",
    "setup_profiler",
    "teardown_profiler",
    "DEFAULT_SAMPLING_FREQUENCY",
    "MAX_STACK_DEPTH",
    "get_frame_name",
    "extract_frame",
    "extract_stack",
    "frame_id",
]
