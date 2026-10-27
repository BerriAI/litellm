import importlib
import sys
import warnings
from collections.abc import Iterator
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Final


def _old_name_for(new_module_name: str, renames: tuple[tuple[str, str], ...]) -> str:
    for old_prefix, new_prefix in renames:
        if new_module_name == new_prefix or new_module_name.startswith(new_prefix + "."):
            return old_prefix + new_module_name[len(new_prefix) :]
    return new_module_name


def _module_names_under(root: str, package_name: str) -> Iterator[str]:
    for path in Path(root).rglob("*.py"):
        relative: Final = path.relative_to(root).with_suffix("")
        parts: Final = relative.parts[:-1] if relative.name == "__init__" else relative.parts
        for depth in range(1, len(parts) + 1):
            yield ".".join((package_name, *parts[:depth]))


def _submodule_names(package: ModuleType) -> tuple[str, ...]:
    per_root: Final = (_module_names_under(root, package.__name__) for root in package.__path__)
    return tuple(sorted(frozenset(chain.from_iterable(per_root))))


def alias_renamed_package(old_name: str, new_name: str, *, nested_renames: tuple[tuple[str, str], ...] = ()) -> None:
    """Register every module of the package now at ``new_name`` in ``sys.modules`` under its old dotted name.

    ``nested_renames`` lists (old, new) dotted prefixes for sub-packages that changed name inside the move, longest
    first, so a deep import of the old path resolves to the very same module object as the new path.
    """
    warnings.warn(
        f"{old_name} moved to {new_name}; update imports, the old path will be removed in a future release",
        DeprecationWarning,
        stacklevel=3,
    )
    renames: Final = (*nested_renames, (old_name, new_name))
    shim: Final = sys.modules[old_name]
    for new_module_name in _submodule_names(importlib.import_module(new_name)):
        module: Final = importlib.import_module(new_module_name)
        alias: Final = _old_name_for(new_module_name, renames)
        sys.modules[alias] = module
        parent_name, _, child_name = alias.rpartition(".")
        setattr(sys.modules.get(parent_name, shim), child_name, module)
