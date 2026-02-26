"""
Prometheus multiprocess directory cleanup utilities.

mark_process_dead() only removes gauge_live* files — counter/histogram
files are kept for correct aggregation and wiped in bulk on startup.
"""

from __future__ import annotations

import glob
import os
from typing import Optional

from litellm._logging import verbose_proxy_logger

def _get_multiproc_dir() -> Optional[str]:
    """Return the PROMETHEUS_MULTIPROC_DIR env var value, or None."""
    return os.environ.get("PROMETHEUS_MULTIPROC_DIR") or os.environ.get(
        "prometheus_multiproc_dir"
    )



def wipe_directory(directory: str) -> None:
    """Delete all .db files in the directory. Called once before workers fork."""
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
    """Mark the current process as dead via mark_process_dead() (worker shutdown)."""
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
