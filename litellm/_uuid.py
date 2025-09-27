"""
Internal unified UUID helper.

Always uses fastuuid for performance.
"""

import fastuuid as _uuid  # type: ignore


# Expose a module-like alias so callers can use: uuid.uuid4()
uuid = _uuid


def uuid4():
    """Return a UUID4 using the selected backend."""
    return uuid.uuid4()
