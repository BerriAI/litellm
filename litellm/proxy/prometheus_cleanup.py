"""
Prometheus multiprocess directory cleanup utilities.

Wipes all .db files on startup so workers start with a clean slate.
"""

from __future__ import annotations

import glob
import os

from litellm._logging import verbose_proxy_logger


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
