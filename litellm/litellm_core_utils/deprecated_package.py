import importlib
import importlib.abc
import importlib.util
import sys
import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from importlib.machinery import ModuleSpec
from types import ModuleType
from typing import Final


@dataclass(frozen=True, slots=True)
class _AliasLoader(importlib.abc.Loader):
    module: ModuleType
    spec: ModuleSpec | None

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return self.module

    def exec_module(self, module: ModuleType) -> None:
        module.__spec__ = self.spec


@dataclass(frozen=True, slots=True)
class _AliasFinder(importlib.abc.MetaPathFinder):
    """Resolve ``old_name.x.y`` to the already importable ``new_name.x.y`` the first time it is imported."""

    old_name: str
    new_name: str

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> ModuleSpec | None:
        if not fullname.startswith(f"{self.old_name}."):
            return None
        new_dotted: Final = f"{self.new_name}{fullname.removeprefix(self.old_name)}"
        if importlib.util.find_spec(new_dotted) is None:
            return None
        module: Final = importlib.import_module(new_dotted)
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(module=module, spec=module.__spec__),
            is_package=hasattr(module, "__path__"),
        )


def alias_deprecated_package(old_name: str, new_name: str) -> Callable[[str], ModuleType]:
    """Warn once, make every ``old_name`` submodule import resolve lazily to ``new_name``, and return the shim's ``__getattr__``."""
    warnings.warn(f"{old_name} is deprecated, import {new_name} instead", DeprecationWarning, stacklevel=3)
    finder: Final = _AliasFinder(old_name=old_name, new_name=new_name)
    if finder not in sys.meta_path:
        sys.meta_path.insert(0, finder)

    def module_getattr(name: str) -> ModuleType:
        try:
            return importlib.import_module(f"{old_name}.{name}")
        except ModuleNotFoundError as error:
            raise AttributeError(f"module {old_name!r} has no attribute {name!r}") from error

    return module_getattr
