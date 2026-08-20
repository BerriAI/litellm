"""Leaf-level recent-activity marker buffer for crash diagnostics.

This module is deliberately dependency-free (stdlib only, no litellm imports)
so it can be called from any layer — including ``litellm_core_utils`` and
caching — without creating circular imports or adding meaningful overhead on
hot paths. ``mark()`` is a no-op (early return) when diagnostics is not
enabled, so it is safe to call unconditionally.

``litellm.proxy.crash_diagnostics`` reads ``_buffer`` when dumping context on a
fatal signal; the two modules share state through this leaf module rather than
through the proxy package.
"""

import threading
from typing import List, Optional, Tuple

# (monotonic_seconds, event_text). None when diagnostics disabled.
_buffer: Optional[List[Tuple[float, str]]] = None
_lock = threading.Lock()
_MAX = 50


def enable() -> None:
    """Turn on the rolling buffer. Called by crash_diagnostics.install()."""
    global _buffer
    if _buffer is None:
        _buffer = []


def mark(event: str) -> None:
    """Record a recent-activity marker.

    Intended for strategic call sites (e.g. start/end of ``success_handler``,
    ``model_dump``, ``log_pre_api_call``, ``_observed_load_sync_loop``
    iterations) so the crash dump shows the last few things the proxy was doing.
    Cheap: a list append under a lock. No-op when diagnostics is not enabled.
    """
    if _buffer is None:
        return
    import time

    try:
        with _lock:
            _buffer.append((time.monotonic(), event))
            if len(_buffer) > _MAX:
                del _buffer[: len(_buffer) - _MAX]
    except Exception:
        pass


def snapshot() -> List[Tuple[float, str]]:
    """Return a copy of the current buffer (empty list if disabled)."""
    if _buffer is None:
        return []
    with _lock:
        return list(_buffer)
