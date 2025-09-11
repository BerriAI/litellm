from typing import Callable

from ddtrace import config
from ddtrace import version
from ddtrace.internal import agent
from ddtrace.internal.datadog.profiling import crashtracker
from ddtrace.internal.runtime import get_runtime_id
from ddtrace.internal.runtime import on_runtime_id_change
from ddtrace.settings.crashtracker import config as crashtracker_config
from ddtrace.settings.profiling import config as profiling_config
from ddtrace.settings.profiling import config_str


is_available: bool = crashtracker.is_available
failure_msg: str = crashtracker.failure_msg
is_started: Callable[[], bool] = crashtracker.is_started


@on_runtime_id_change
def _update_runtime_id(runtime_id: str) -> None:
    crashtracker.set_runtime_id(runtime_id)


def add_tag(key: str, value: str) -> None:
    if is_available:
        crashtracker.set_tag(key, value)


def start() -> bool:
    if not is_available:
        return False

    import platform

    crashtracker.set_url(crashtracker_config.debug_url or agent.get_trace_url())
    crashtracker.set_service(config.service)
    crashtracker.set_version(config.version)
    crashtracker.set_env(config.env)
    crashtracker.set_runtime_id(get_runtime_id())
    crashtracker.set_runtime_version(platform.python_version())
    crashtracker.set_library_version(version.get_version())
    crashtracker.set_create_alt_stack(bool(crashtracker_config.create_alt_stack))
    crashtracker.set_use_alt_stack(bool(crashtracker_config.use_alt_stack))
    if crashtracker_config.stacktrace_resolver == "fast":
        crashtracker.set_resolve_frames_fast()
    elif crashtracker_config.stacktrace_resolver == "full":
        crashtracker.set_resolve_frames_full()
    elif crashtracker_config.stacktrace_resolver == "safe":
        crashtracker.set_resolve_frames_safe()
    else:
        crashtracker.set_resolve_frames_disable()

    if crashtracker_config.stdout_filename:
        crashtracker.set_stdout_filename(crashtracker_config.stdout_filename)
    if crashtracker_config.stderr_filename:
        crashtracker.set_stderr_filename(crashtracker_config.stderr_filename)

    # Add user tags
    for key, value in crashtracker_config.tags.items():
        add_tag(key, value)

    if profiling_config.enabled:
        add_tag("profiler_config", config_str(profiling_config))

    # Only start if it is enabled
    if crashtracker_config.enabled:
        return crashtracker.start()
    return False
