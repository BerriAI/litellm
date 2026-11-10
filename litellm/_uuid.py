"""
Internal unified UUID helper.

Uses fastuuid where supported and the standard library on Python 3.15+.
"""

import sys
import uuid as _uuid

if sys.version_info < (3, 15):
    import fastuuid as _uuid

# Expose a module-like alias so callers can use: uuid.uuid4()
uuid = _uuid


def uuid4():
    """Return a UUID4 using the selected backend."""
    return uuid.uuid4()
