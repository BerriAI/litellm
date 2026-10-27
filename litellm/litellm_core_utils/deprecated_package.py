import importlib
import sys
import warnings
from itertools import chain
from pathlib import Path
from types import ModuleType
from typing import Final


def _dotted_name(root: Path, path: Path, package_name: str) -> str:
    parts: Final = path.relative_to(root).with_suffix("").parts
    return ".".join((package_name, *parts)).removesuffix(".__init__")


def _parents_and_self(dotted: str, package_name: str) -> tuple[str, ...]:
    parts: Final = dotted.removeprefix(f"{package_name}.").split(".")
    return tuple(f"{package_name}.{'.'.join(parts[:depth])}" for depth in range(1, len(parts) + 1))


def _module_names(package: ModuleType) -> tuple[str, ...]:
    roots: Final = tuple(Path(root) for root in package.__path__)
    files: Final = tuple(
        (root, path) for root in roots for path in root.rglob("*.py")
    )  # comprehension-ok: nested walk over each package root
    leaves: Final = tuple(
        name
        for name in (_dotted_name(root, path, package.__name__) for root, path in files)
        if name != package.__name__
    )
    return tuple(
        sorted(
            set(chain.from_iterable(_parents_and_self(leaf, package.__name__) for leaf in leaves)) - {package.__name__}
        )
    )


def _alias(old_name: str, new_package_name: str, new_dotted: str) -> None:
    module: Final = importlib.import_module(new_dotted)
    old_dotted: Final = f"{old_name}{new_dotted.removeprefix(new_package_name)}"
    old_parent, _, child = old_dotted.rpartition(".")
    sys.modules[old_dotted] = module
    setattr(sys.modules[old_parent], child, module)


def alias_deprecated_package(old_name: str, new_package: ModuleType) -> None:
    """Make every module under ``new_package`` importable and attribute-reachable under ``old_name`` and warn once."""
    warnings.warn(
        f"{old_name} is deprecated, import {new_package.__name__} instead",
        DeprecationWarning,
        stacklevel=3,
    )
    for new_dotted in _module_names(new_package):
        _alias(old_name, new_package.__name__, new_dotted)
