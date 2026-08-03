"""Supervisor-side reaper for orphaned Prisma query-engine processes.

Each proxy worker owns a Prisma query-engine subprocess whose only cleanup
hook is an in-process ``atexit`` handler. When a multi-worker supervisor
(uvicorn's multiprocess manager, the gunicorn arbiter) force-kills a hung or
crashed worker, that handler never runs: the engine reparents to the nearest
subreaper (PID 1 in a container, which is the supervisor itself under the
standard docker entrypoint) and keeps its database connection pool
established forever, while the replacement worker opens a fresh pool. Over
repeated worker deaths the active database connections grow without bound.

The reaper runs only in the supervisor process, where a query-engine process
can never be a legitimate direct child: workers own their engines, and the
supervisor never starts one. Any direct child whose command name begins with
``query-engine`` is therefore an adopted orphan and is terminated
(SIGTERM, bounded grace, SIGKILL) and reaped. On Linux the supervisor also
marks itself a child subreaper so orphans reparent to it even when it is not
PID 1.

Linux-only by construction (``/proc`` scan, ``prctl``); a no-op elsewhere.
"""

import ctypes
import os
import signal
import sys
import threading
import time

from litellm._logging import verbose_proxy_logger

QUERY_ENGINE_COMM_PREFIX = "query-engine"
REAPER_SCAN_INTERVAL_SECONDS = 5.0
SIGTERM_GRACE_SECONDS = 10.0
PR_SET_CHILD_SUBREAPER = 36


def set_child_subreaper() -> bool:
    """Mark this process as a child subreaper so orphaned descendants
    reparent to it instead of PID 1. Best-effort: when it fails (or on
    non-Linux) the reaper still covers the containerized case where the
    supervisor already is PID 1."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        result: int = libc.prctl(  # pyright: ignore[reportAny]  # ctypes types foreign calls as Any; default restype is c_int
            PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0
        )
        return result == 0
    except (OSError, AttributeError):
        return False


def _read_comm_and_ppid(pid: int, proc_root: str) -> tuple[str, int] | None:
    try:
        with open(f"{proc_root}/{pid}/stat", encoding="ascii", errors="replace") as stat_file:
            data = stat_file.read()
    except (FileNotFoundError, ProcessLookupError, PermissionError, OSError):
        return None
    lparen = data.find("(")
    rparen = data.rfind(")")
    if lparen == -1 or rparen == -1 or rparen < lparen:
        return None
    comm = data[lparen + 1 : rparen]
    fields = data[rparen + 2 :].split()
    if len(fields) < 2:
        return None
    try:
        ppid = int(fields[1])
    except ValueError:
        return None
    return comm, ppid


def list_orphaned_engine_pids(parent_pid: int, proc_root: str = "/proc") -> tuple[int, ...]:
    """PIDs of direct children of ``parent_pid`` whose command name marks
    them as Prisma query engines. In the supervisor these are always
    adopted orphans: live engines are children of workers, not of the
    supervisor."""
    try:
        entries = os.listdir(proc_root)
    except (FileNotFoundError, OSError):
        return ()
    candidate_pids = (int(entry) for entry in entries if entry.isdigit())
    return tuple(
        pid
        for pid in candidate_pids
        if (info := _read_comm_and_ppid(pid, proc_root)) is not None
        and info[1] == parent_pid
        and info[0].startswith(QUERY_ENGINE_COMM_PREFIX)
    )


def _try_reap(pid: int) -> bool:
    try:
        reaped_pid, _ = os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        return True
    except OSError:
        return True
    return reaped_pid == pid


def _send_signal(pid: int, signum: int) -> None:
    try:
        os.kill(pid, signum)
    except (ProcessLookupError, PermissionError, OSError):
        pass


def _await_reaped(pids: tuple[int, ...], timeout_seconds: float) -> tuple[int, ...]:
    """Poll until every PID is reaped or the shared deadline passes.
    Returns the PIDs still alive at the deadline."""
    deadline = time.monotonic() + timeout_seconds
    remaining = pids
    while remaining and time.monotonic() < deadline:
        remaining = tuple(pid for pid in remaining if not _try_reap(pid))
        if remaining:
            time.sleep(0.2)
    return remaining


def terminate_and_reap(pid: int, grace_seconds: float = SIGTERM_GRACE_SECONDS) -> None:
    """SIGTERM the orphaned engine, escalate to SIGKILL after the grace
    period, and reap it so it does not linger as a zombie."""
    terminate_and_reap_all((pid,), grace_seconds=grace_seconds)


def terminate_and_reap_all(
    pids: tuple[int, ...],
    grace_seconds: float = SIGTERM_GRACE_SECONDS,
) -> None:
    """Terminate a batch of orphaned engines concurrently: SIGTERM all of
    them, share one grace period, SIGKILL the stragglers, and reap. The
    shared deadline keeps cleanup time bounded when several workers die
    at once instead of paying the grace period once per orphan."""
    for pid in pids:
        verbose_proxy_logger.warning(
            "Reaping orphaned prisma query-engine PID %s (its worker process exited without cleanup).",
            pid,
        )
        _send_signal(pid, signal.SIGTERM)
    survivors = _await_reaped(pids, grace_seconds)
    if not survivors:
        return
    for pid in survivors:
        verbose_proxy_logger.warning(
            "Orphaned prisma query-engine PID %s did not exit within %.1fs of SIGTERM; sending SIGKILL.",
            pid,
            grace_seconds,
        )
        _send_signal(pid, signal.SIGKILL)
    unkillable = _await_reaped(survivors, 5.0)
    for pid in unkillable:
        verbose_proxy_logger.error(
            "Orphaned prisma query-engine PID %s survived SIGKILL; will retry on the next scan.",
            pid,
        )


def reap_orphaned_engines(parent_pid: int, proc_root: str = "/proc") -> tuple[int, ...]:
    """One scan-and-reap pass. Returns the PIDs it acted on."""
    orphaned_pids = list_orphaned_engine_pids(parent_pid, proc_root=proc_root)
    if orphaned_pids:
        terminate_and_reap_all(orphaned_pids)
    return orphaned_pids


def _reaper_loop(parent_pid: int) -> None:
    while True:
        try:
            reap_orphaned_engines(parent_pid)
        except Exception as scan_error:  # noqa: BLE001  # reaper thread must survive any scan failure
            verbose_proxy_logger.debug("Orphaned query-engine scan failed: %s", scan_error)
        time.sleep(REAPER_SCAN_INTERVAL_SECONDS)


REAPER_THREAD_NAME = "litellm-orphan-query-engine-reaper"


def start_query_engine_reaper() -> threading.Thread | None:
    """Start the reaper daemon thread in the supervisor process.

    Must only be called from a process that never hosts the proxy app
    itself (uvicorn with ``workers > 1``, the gunicorn arbiter): with a
    single in-process uvicorn worker the query engine is a legitimate
    direct child and must not be touched. Idempotent: a reaper already
    running in this process is returned instead of starting a second one.
    """
    if not sys.platform.startswith("linux"):
        return None
    existing = next(
        (thread for thread in threading.enumerate() if thread.name == REAPER_THREAD_NAME),
        None,
    )
    if existing is not None:
        return existing
    set_child_subreaper()
    reaper_thread = threading.Thread(
        target=_reaper_loop,
        args=(os.getpid(),),
        daemon=True,
        name=REAPER_THREAD_NAME,
    )
    reaper_thread.start()
    verbose_proxy_logger.info(
        "Started orphaned prisma query-engine reaper in supervisor process %s.",
        os.getpid(),
    )
    return reaper_thread
