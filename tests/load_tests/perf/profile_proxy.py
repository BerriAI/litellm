"""Run the LiteLLM proxy under yappi.

Usage:
    profile_proxy.py <output_path.ystat> -- <litellm-cli-args>

yappi control:
    SIGUSR1: start profiling (idempotent)
    SIGUSR2: stop + dump profile to <output_path.ystat> (idempotent)

Uses `set_clock_type("cpu")` so suspended coroutines don't accumulate wall
time. Profiles every thread yappi's Python hook intercepts — covers the
asyncio event-loop thread + any ThreadPoolExecutor workers.
"""
import atexit
import signal
import sys

import yappi
from litellm import run_server

if len(sys.argv) < 2:
    print(
        f"usage: {sys.argv[0]} <output_path.ystat> -- <litellm-cli-args>",
        file=sys.stderr,
    )
    sys.exit(1)

PROFILE_PATH = sys.argv.pop(1)
# Drop a sentinel '--' if present so click doesn't choke on it
if len(sys.argv) > 1 and sys.argv[1] == "--":
    sys.argv.pop(1)

yappi.set_clock_type("cpu")

_started = False
_dumped = False


def start_profiling(*_: object) -> None:
    global _started
    if _started:
        return
    _started = True
    yappi.start()


def dump_profile(*_: object) -> None:
    global _dumped
    if _dumped:
        return
    _dumped = True
    try:
        yappi.stop()
    except Exception:
        pass
    try:
        yappi.get_func_stats().save(PROFILE_PATH, type="ystat")
    except Exception as e:
        print(f"profile_proxy: failed to save profile: {e}", file=sys.stderr)


# uvicorn installs its own SIGTERM/SIGINT handlers after run_server() starts,
# so we use SIGUSR1/SIGUSR2 which it leaves alone.
signal.signal(signal.SIGUSR1, start_profiling)
signal.signal(signal.SIGUSR2, dump_profile)
atexit.register(dump_profile)

run_server()
