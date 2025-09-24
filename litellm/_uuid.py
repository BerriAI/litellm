"""
Internal unified UUID helper.

Tries to use fastuuid (performance) and falls back to stdlib uuid if unavailable.
"""

FASTUUID_AVAILABLE = False

try:
    import fastuuid as _uuid  # type: ignore

    FASTUUID_AVAILABLE = True
except Exception:  # pragma: no cover - fallback path
    import uuid as _uuid  # type: ignore


# Expose a module-like alias so callers can use: uuid.uuid4()
uuid = _uuid


def uuid4():
    """Return a UUID4 using the selected backend."""
    return uuid.uuid4()
