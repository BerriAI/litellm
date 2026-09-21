"""
Prometheus multiprocess directory cleanup utilities.

Wipes all .db files on startup so workers start with a clean slate.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Final

from litellm._logging import verbose_proxy_logger

_LIVE_GAUGE_PID: Final = re.compile(r"gauge_live[a-z]*_(\d+)\.db$")


def wipe_directory(directory: str) -> None:
    """Delete all .db files in the directory. Called once before workers fork."""
    files: Final = glob.glob(os.path.join(directory, "*.db"))
    deleted = 0
    for filepath in files:
        try:
            os.remove(filepath)
            deleted += 1
        except OSError as e:
            verbose_proxy_logger.warning("Failed to delete stale prometheus file %s: %s", filepath, e)
    if deleted:
        verbose_proxy_logger.info("Prometheus cleanup: wiped %s stale .db files from %s", deleted, directory)


def mark_worker_exit(worker_pid: int) -> None:
    """Remove prometheus .db files for a dead worker. Called by gunicorn child_exit hook."""
    if not os.environ.get("PROMETHEUS_MULTIPROC_DIR"):
        return
    try:
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(worker_pid)
        verbose_proxy_logger.info("Prometheus cleanup: marked worker %s as dead", worker_pid)
    except Exception as e:
        verbose_proxy_logger.warning("Failed to mark prometheus worker %s as dead: %s", worker_pid, e)


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def mark_dead_workers(directory: str) -> tuple[int, ...]:
    """Drop the live-gauge files of workers that no longer exist and return their pids.

    Uvicorn's multi-worker supervisor has no exit hook, so a replacement worker calls this at startup; without it
    a crashed worker's in-flight gauges stay in the aggregate forever.
    """
    owners: Final = frozenset(
        int(match.group(1))
        for match in map(_LIVE_GAUGE_PID.search, glob.glob(os.path.join(directory, "gauge_live*_*.db")))
        if match is not None
    )
    dead: Final = tuple(sorted(pid for pid in owners if pid != os.getpid() and not _is_running(pid)))
    if not dead:
        return dead
    from prometheus_client import multiprocess

    for pid in dead:
        multiprocess.mark_process_dead(pid, path=directory)
    verbose_proxy_logger.info("Prometheus cleanup: marked dead workers %s in %s", dead, directory)
    return dead
