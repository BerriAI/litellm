"""
Periodic memory cleanup utilities for the LiteLLM proxy.

Python's memory allocator (pymalloc) does not return freed memory arenas back
to the OS by default. After handling a burst of traffic, process RSS stays
elevated even though the objects have been garbage-collected. This module
provides a lightweight scheduled job that:

1. Runs a full GC collection cycle
2. Calls ``malloc_trim(0)`` on Linux (glibc) to return unused heap pages to
   the OS

This is especially important for long-running proxy deployments where memory
grows during load tests / peak traffic and never shrinks back.

Environment variables:
    LITELLM_MEMORY_CLEANUP_INTERVAL  – seconds between cleanup runs (default 60)
"""

import ctypes
import gc
import sys

from litellm._logging import verbose_proxy_logger

_libc = None
_malloc_trim_available = False

if sys.platform == "linux":
    try:
        _libc = ctypes.CDLL("libc.so.6")
        _malloc_trim_available = hasattr(_libc, "malloc_trim")
    except OSError:
        pass


def _periodic_memory_cleanup() -> None:
    """Run a full GC cycle and release freed heap memory back to the OS.

    On Linux with glibc, ``malloc_trim(0)`` forces the allocator to return
    unused pages to the kernel. Without this, RSS never shrinks even after
    all Python objects are freed.

    Safe to call frequently (every 60s by default). ``gc.collect()`` is ~1ms
    when there is little garbage, and ``malloc_trim`` is a no-op when there
    are no reclaimable pages.
    """
    collected = gc.collect()
    trimmed = False

    if _malloc_trim_available and _libc is not None:
        try:
            _libc.malloc_trim(0)
            trimmed = True
        except Exception as exc:
            verbose_proxy_logger.debug(
                "malloc_trim failed (non-critical): %s", exc
            )

    verbose_proxy_logger.debug(
        "Periodic memory cleanup: gc collected %d objects, malloc_trim=%s",
        collected,
        trimmed,
    )
