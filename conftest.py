"""Repository-root pytest configuration.

After the SQLAlchemy big-bang migration, the ``prisma`` PyPI package is
no longer installed but a handful of test modules still do ``from prisma
... import ...``. We install a stand-in for the ``prisma`` namespace at
plugin-load time -- *before* pytest collects test modules -- so those
imports resolve without rewriting every test file in this PR.

The stand-in lives in ``tests/_prisma_compat.py`` and re-exports the
LiteLLM-native error classes from ``litellm.proxy.db.sqlmodel.errors``.
"""

from __future__ import annotations

import os
import sys

# Make ``tests/_prisma_compat`` importable regardless of where pytest is
# invoked from.
_TESTS_DIR = os.path.join(os.path.dirname(__file__), "tests")
if _TESTS_DIR not in sys.path:
    sys.path.insert(0, _TESTS_DIR)

try:
    from _prisma_compat import install as _install_prisma_compat

    _install_prisma_compat()
except ImportError:  # pragma: no cover -- shim is best-effort
    pass
