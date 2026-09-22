"""Whether a killed process is really gone, for tests that kill whole process trees.

A SIGKILLed grandchild whose parent died in the same ``killpg`` reparents to the
nearest subreaper or PID 1, and until that ancestor reaps it the pid is a zombie
that ``os.kill(pid, 0)`` still accepts. Reading its ``/proc`` state, and reaping
it when it landed on this process, keeps a runner that is slow to reap, or never
does, from turning a dead process into a failed assertion. The reap comes after
the liveness read so a child seen dying between the two is still collected on
the next poll instead of staying this process's own zombie.
"""

import os
import time
from pathlib import Path
from typing import Final

POLL_INTERVAL_S: Final = 0.05


def _exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _is_zombie(pid: int) -> bool:
    try:
        stat: Final = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return False
    return stat.rpartition(")")[2].split()[0] == "Z"


def _reap_if_ours(pid: int) -> None:
    if os.name == "nt":
        return
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass


def _gone_now(pid: int) -> bool:
    dead: Final = not _exists(pid) or _is_zombie(pid)
    _reap_if_ours(pid)
    return dead


def process_is_gone(pid: int, within_seconds: float) -> bool:
    deadline: Final = time.monotonic() + within_seconds
    while time.monotonic() < deadline:
        if _gone_now(pid):
            return True
        time.sleep(POLL_INTERVAL_S)
    return False
