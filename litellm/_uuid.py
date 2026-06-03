"""
Internal unified UUID helper.

Uses the standard library uuid module so litellm installs on every supported
Python; fastuuid ships no wheels for 3.14.
"""

import uuid as _uuid

# Expose a module-like alias so callers can use: uuid.uuid4()
uuid = _uuid


def uuid4() -> _uuid.UUID:
    """Return a UUID4 using the standard library backend."""
    return uuid.uuid4()
