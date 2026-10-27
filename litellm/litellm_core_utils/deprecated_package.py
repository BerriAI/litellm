import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from types import ModuleType
from typing import Final


@dataclass(frozen=True, slots=True)
class _AliasLoader(importlib.abc.Loader):
    new_dotted: str

    def create_module(self, spec: importlib.machinery.ModuleSpec) -> ModuleType:
        return importlib.import_module(self.new_dotted)

    def exec_module(self, module: ModuleType) -> None:
        pass


@dataclass(frozen=True, slots=True)
class _AliasFinder(importlib.abc.MetaPathFinder):
    old_name: str
    new_name: str

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> importlib.machinery.ModuleSpec | None:
        if not fullname.startswith(f"{self.old_name}."):
            return None
        new_dotted: Final = f"{self.new_name}{fullname.removeprefix(self.old_name)}"
        new_spec: Final = importlib.util.find_spec(new_dotted)
        if new_spec is None:
            return None
        return importlib.util.spec_from_loader(
            fullname,
            _AliasLoader(new_dotted),
            is_package=new_spec.submodule_search_locations is not None,
        )


def alias_deprecated_package(old_name: str, new_package: ModuleType) -> None:
    warnings.warn(
        f"{old_name} is deprecated, import {new_package.__name__} instead",
        DeprecationWarning,
        stacklevel=3,
    )
    finder: Final = _AliasFinder(old_name, new_package.__name__)
    if finder not in sys.meta_path:
        sys.meta_path.insert(0, finder)
