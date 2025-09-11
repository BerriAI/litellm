from __future__ import annotations

# See #940 for why lazy_import isn't used here for backwards compatibility.
# See #1400 for why listing compatibility imports in __all__ helps PyCharm.
from .legacy.auth import *
from .legacy.auth import __all__  # noqa: F401
