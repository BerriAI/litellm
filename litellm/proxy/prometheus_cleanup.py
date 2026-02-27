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
