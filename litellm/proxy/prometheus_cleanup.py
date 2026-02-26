"""
Prometheus multiprocess directory cleanup utilities.

When running with multiple workers and PROMETHEUS_MULTIPROC_DIR set,
each worker creates memory-mapped .db files (e.g., counter_1234.db).
When workers die or restart, gauge_live* files for dead PIDs must be
cleaned up via mark_process_dead(). Counter and histogram files are
kept since they contain cumulative data needed for correct aggregation.
"""

from __future__ import annotations

import glob
import os
import re
from typing import Optional, Set

from litellm._logging import verbose_proxy_logger

_PID_PATTERN = re.compile(r"_(\d+)\.db$")


def _get_multiproc_dir() -> Optional[str]:
    """Return the PROMETHEUS_MULTIPROC_DIR env var value, or None."""
    return os.environ.get("PROMETHEUS_MULTIPROC_DIR") or os.environ.get(
        "prometheus_multiproc_dir"
    )


def _is_pid_alive(pid: int) -> bool:
    """
    Check if a process with the given PID is alive.

    Uses os.kill(pid, 0) which doesn't send a signal but checks existence.
    - ProcessLookupError: process does not exist (dead)
    - PermissionError: process exists but we can't signal it (alive, conservative)
    - OSError: other error, treat as alive (conservative)
    """
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we don't have permission to signal it
        return True
    except OSError:
        # Conservative: treat unknown errors as alive
        return True


def _extract_pids_from_dir(directory: str) -> Set[int]:
    """
    Scan .db filenames in a directory and extract PIDs.

    Prometheus client creates files like:
    - counter_1234.db
    - histogram_1234.db
    - gauge_livesum_1234.db
    - gauge_liveall_1234.db

    Returns a set of integer PIDs found.
    """
    pids: Set[int] = set()
    try:
        for filename in os.listdir(directory):
            if not filename.endswith(".db"):
                continue
            match = _PID_PATTERN.search(filename)
            if match:
                pids.add(int(match.group(1)))
    except FileNotFoundError:
        pass
    return pids


def wipe_directory(directory: str) -> None:
    """
    Delete all .db files in the prometheus multiproc directory.

    Called once in the master process before workers fork. Per the
    prometheus_client docs: "This directory must be wiped between
    process runs (before startup is recommended)."

    Any .db files present at this point are stale from a previous run.
    """
    files = glob.glob(os.path.join(directory, "*.db"))
    for filepath in files:
        try:
            os.remove(filepath)
        except OSError as e:
            verbose_proxy_logger.warning(
                f"Failed to delete stale prometheus file {filepath}: {e}"
            )
    if files:
        verbose_proxy_logger.info(
            f"Prometheus cleanup: wiped {len(files)} stale .db files from {directory}"
        )


def cleanup_own_pid_files() -> None:
    """
    Mark the current process as dead for prometheus multiproc cleanup.

    Called during per-worker shutdown. Uses mark_process_dead() which
    only removes gauge_live* files — counter and histogram files are
    preserved since they contain cumulative data needed for correct
    aggregation until the directory is wiped on next startup.
    """
    directory = _get_multiproc_dir()
    if not directory or not os.path.isdir(directory):
        return

    from prometheus_client import multiprocess

    pid = os.getpid()
    try:
        multiprocess.mark_process_dead(pid)
        verbose_proxy_logger.info(
            f"Prometheus cleanup: marked worker PID {pid} as dead"
        )
    except Exception as e:
        verbose_proxy_logger.warning(
            f"Failed to mark worker PID {pid} as dead: {e}"
        )


def mark_dead_pids(skip_pid: Optional[int] = None) -> None:
    """
    Scan the prometheus multiproc directory and call mark_process_dead()
    for PIDs that no longer exist.

    Uses prometheus_client.multiprocess.mark_process_dead() which only
    removes gauge_live* files — counter and histogram files are preserved
    since they contain cumulative data needed for correct aggregation.

    Args:
        skip_pid: PID to skip (typically os.getpid()). If None, skips
                  the current process's PID.
    """
    directory = _get_multiproc_dir()
    if not directory or not os.path.isdir(directory):
        return

    if skip_pid is None:
        skip_pid = os.getpid()

    pids = _extract_pids_from_dir(directory)

    from prometheus_client import multiprocess

    dead_pids = []
    for pid in pids:
        if pid == skip_pid:
            continue
        if not _is_pid_alive(pid):
            try:
                multiprocess.mark_process_dead(pid)
                dead_pids.append(pid)
            except Exception as e:
                verbose_proxy_logger.warning(
                    f"Failed to mark PID {pid} as dead: {e}"
                )

    if dead_pids:
        verbose_proxy_logger.info(
            f"Prometheus cleanup: marked {len(dead_pids)} dead PIDs: {dead_pids}"
        )
