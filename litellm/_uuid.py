"""
Internal unified UUID helper.

Uses fastuuid for performance when installed, otherwise falls back to stdlib uuid.
"""

try:
    import fastuuid as _uuid
except ImportError:
    import uuid as _uuid

# Expose a module-like alias so callers can use: uuid.uuid4()
uuid = _uuid


def uuid4():
    """Return a UUID4 using the selected backend."""
    return uuid.uuid4()

