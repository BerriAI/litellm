"""Test-only prisma stand-in.

After the SQLAlchemy big-bang migration, the ``prisma`` PyPI package is
no longer installed. A handful of tests still do
``from prisma.errors import RecordNotFoundError`` etc.; this module
registers thin sys.modules entries so those imports resolve to the
LiteLLM-native error classes defined in
:mod:`litellm.proxy.db.sqlmodel.errors`.

The shim is intentionally only loaded from ``tests/test_litellm/conftest.py``
and ``tests/conftest.py`` -- it does **not** ship in the ``litellm``
wheel, and production code that previously imported from ``prisma`` has
been ported to the native errors module.
"""

from __future__ import annotations

import sys
import types

from litellm.proxy.db.sqlmodel import errors as _native_errors


def _build_prisma_errors_module() -> types.ModuleType:
    mod = types.ModuleType("prisma.errors")
    for name in (
        "PrismaError",
        "DataError",
        "UniqueViolationError",
        "ForeignKeyViolationError",
        "RecordNotFoundError",
        "MissingRequiredValueError",
        "TableNotFoundError",
        "RawQueryError",
        "ClientNotConnectedError",
        "HTTPClientClosedError",
    ):
        setattr(mod, name, getattr(_native_errors, name))
    return mod


def _build_prisma_module(errors_mod: types.ModuleType) -> types.ModuleType:
    mod = types.ModuleType("prisma")
    mod.errors = errors_mod  # type: ignore[attr-defined]

    # ``prisma.Json`` was a marker callable for inserting Python dicts into
    # Prisma ``Json`` columns. SQLAlchemy's JSONB accepts dicts directly,
    # so we re-export an identity function.
    def _json(value):  # noqa: N802 -- preserves prisma-client-py call shape
        return value

    mod.Json = _json  # type: ignore[attr-defined]

    class _PrismaUnusable:
        """Stand-in for the obsolete ``prisma.Prisma`` class.

        Tests that imported it for ``isinstance`` checks or to construct
        a real engine subprocess no longer apply -- we raise on construction
        so any forgotten production usage surfaces loudly rather than
        silently passing.
        """

        def __init__(self, *args, **kwargs):
            raise RuntimeError(
                "prisma.Prisma is no longer available -- the SQLAlchemy "
                "migration removed the prisma-client-py dependency. Use "
                "litellm.proxy.db.sqlmodel.compat.PrismaCompatClient instead."
            )

    mod.Prisma = _PrismaUnusable  # type: ignore[attr-defined]
    return mod


def install() -> None:
    """Register the stub modules in ``sys.modules`` if real prisma is absent."""
    if "prisma" in sys.modules:
        return
    try:
        import prisma  # type: ignore[import-not-found]  # noqa: F401
    except ImportError:
        errors_mod = _build_prisma_errors_module()
        prisma_mod = _build_prisma_module(errors_mod)
        sys.modules["prisma"] = prisma_mod
        sys.modules["prisma.errors"] = errors_mod
