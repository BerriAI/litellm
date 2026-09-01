"""
Internal unified UUID helper.

Always uses fastuuid for performance.
"""

import fastuuid as _uuid

# Expose a module-like alias so callers can use: uuid.uuid4()
uuid = _uuid


def uuid4():
    """Return a UUID4 using the selected backend."""
    return uuid.uuid4()
