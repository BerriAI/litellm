"""
Prometheus multiprocess directory cleanup utilities.

Wipes all .db files on startup so workers start with a clean slate.
Provides two complementary cleanup mechanisms:
1. mark_worker_exit(): called by gunicorn's child_exit hook on every graceful worker exit.
2. cleanup_stale_worker_files(): periodic sweep that removes files for PIDs that are no
   longer alive — covers crashed workers and anything missed by the gunicorn hook.
"""

from __future__ import annotations

import glob
import os
import re

from litellm._logging import verbose_proxy_logger

# Matches the PID at the end of every prometheus multiprocess filename.
# prometheus_client names files as:
#   {type}_{pid}.db           (counter, histogram, summary)
#   gauge_{mode}_{pid}.db     (gauge with multiprocess_mode)
# In both cases the PID is the numeric run at the very end, preceded by '_'.
_PID_RE = re.compile(r"_(\d+)\.db$")


def wipe_directory(directory: str) -> None:
    """Delete all .db files in the directory. Called once before workers fork."""
    files = glob.glob(os.path.join(directory, "*.db"))
    deleted = 0
    for filepath in files:
        try:
            os.remove(filepath)
            deleted += 1
        except OSError as e:
            verbose_proxy_logger.warning(
                f"Failed to delete stale prometheus file {filepath}: {e}"
            )
    if deleted:
        verbose_proxy_logger.info(
            f"Prometheus cleanup: wiped {deleted} stale .db files from {directory}"
        )


def _is_pid_alive(pid: int) -> bool:
    """Return True if the process with *pid* is still running on this host."""
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we cannot signal it — still alive.
        return True
    except OSError:
        return False


def cleanup_stale_worker_files(directory: str) -> int:
    """Remove .db files whose PIDs are no longer alive.

    Acts as a periodic safety net complementing mark_worker_exit(): covers
    files left by crashed workers (where gunicorn's child_exit hook never
    fires) and files carried over from a previous process group (e.g. a pod
    restart where wipe_directory() was skipped).

    Returns the number of files deleted.
    """
    files = glob.glob(os.path.join(directory, "*.db"))
    deleted = 0
    for filepath in files:
        filename = os.path.basename(filepath)
        match = _PID_RE.search(filename)
        if not match:
            continue
        pid = int(match.group(1))
        if _is_pid_alive(pid):
            continue
        try:
            os.remove(filepath)
            deleted += 1
            verbose_proxy_logger.debug(
                "Prometheus cleanup: removed stale file %s (pid %d no longer alive)",
                filename,
                pid,
            )
        except OSError as e:
            verbose_proxy_logger.warning(
                "Prometheus cleanup: failed to remove %s: %s", filepath, e
            )
    if deleted:
        verbose_proxy_logger.info(
            "Prometheus cleanup: swept %d stale .db file(s) from %s",
            deleted,
            directory,
        )
    return deleted


def mark_worker_exit(worker_pid: int) -> None:
    """Remove prometheus .db files for a dead worker. Called by gunicorn child_exit hook.

    prometheus_client.multiprocess.mark_process_dead() only removes 'live*' gauge files
    (gauge_livesum, gauge_liveall, etc.) in v0.20. Counter, histogram, and non-live gauge
    files are left behind and accumulate with every worker restart. We explicitly delete
    all *_{pid}.db files to prevent unbounded ephemeral storage growth.
    """
    directory = os.environ.get("PROMETHEUS_MULTIPROC_DIR")
    if not directory:
        return

    # Let prometheus_client handle live-gauge bookkeeping first.
    try:
        from prometheus_client import multiprocess

        multiprocess.mark_process_dead(worker_pid)
        verbose_proxy_logger.info(
            f"Prometheus cleanup: marked worker {worker_pid} as dead"
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            f"Failed to mark prometheus worker {worker_pid} as dead: {e}"
        )

    # Delete all remaining files for this PID. mark_process_dead() only cleans up
    # live gauge files; counters, histograms, and non-live gauges are never removed
    # by prometheus_client itself, causing ~100 MB/day growth when workers restart
    # frequently (e.g. --max_requests_before_restart).
    pattern = os.path.join(directory, f"*_{worker_pid}.db")
    deleted = 0
    for filepath in glob.glob(pattern):
        try:
            os.remove(filepath)
            deleted += 1
        except OSError as e:
            verbose_proxy_logger.warning(
                f"Prometheus cleanup: failed to delete {filepath}: {e}"
            )
    if deleted:
        verbose_proxy_logger.info(
            f"Prometheus cleanup: deleted {deleted} stale .db file(s) for dead worker {worker_pid}"
        )
