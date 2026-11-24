import importlib
import sys
from collections.abc import Sequence
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Final

import pytest

LIBPQ_IMPORT_ERROR: Final = "couldn't import requested psycopg 'python' implementation: libpq library not found"


def _is_psycopg(name: str) -> bool:
    return name == "psycopg" or name.startswith("psycopg.")


class _LibpqMissingFinder:
    """Fails `import psycopg` the way PSYCOPG_IMPL=python does on a node without libpq: a plain
    ImportError, which `except ModuleNotFoundError` would not catch."""

    def find_spec(
        self, fullname: str, path: Sequence[str] | None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if _is_psycopg(fullname):
            raise ImportError(LIBPQ_IMPORT_ERROR)
        return None


@pytest.fixture(params=("not-installed", "libpq-missing"))
def unloadable_psycopg(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    """Makes `import psycopg` fail for the test and returns the error text it fails with."""
    loaded: Final = tuple(name for name in sys.modules if _is_psycopg(name))
    for name in loaded:
        monkeypatch.delitem(sys.modules, name)
    if request.param == "not-installed":
        monkeypatch.setitem(sys.modules, "psycopg", None)
    else:
        monkeypatch.setattr(sys, "meta_path", [_LibpqMissingFinder(), *sys.meta_path])
    try:
        importlib.import_module("psycopg")
    except ImportError as exc:
        return str(exc)
    raise AssertionError("psycopg still imports")
