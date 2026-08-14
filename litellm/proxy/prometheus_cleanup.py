"""
Prometheus multiprocess directory cleanup utilities.

Wipes all .db files on startup so workers start with a clean slate.
"""

from __future__ import annotations

import glob
import os
from typing import Final

from litellm._logging import verbose_proxy_logger


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
