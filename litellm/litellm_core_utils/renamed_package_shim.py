import importlib
import sys
import warnings
from collections.abc import Callable, Sequence
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec
from importlib.util import spec_from_loader
from types import ModuleType
from typing import Final


class _AliasLoader(Loader):
    """Hand the import system the module already loaded under its new name instead of executing a copy."""

    def __init__(self, target: ModuleType) -> None:
        self._target: Final = target
        self._spec: Final = target.__spec__

    def create_module(self, spec: ModuleSpec) -> ModuleType:
        return self._target

    def exec_module(self, module: ModuleType) -> None:
        module.__spec__ = self._spec  # rebind-ok: the loader restores the spec the import system replaced


class _RenamedPackageFinder(MetaPathFinder):
    def __init__(self, old_name: str, renames: tuple[tuple[str, str], ...]) -> None:
        self._prefix: Final = old_name + "."
        self._renames: Final = renames

    def new_name_for(self, old_module_name: str) -> str | None:
        if not old_module_name.startswith(self._prefix):
            return None
        for old_prefix, new_prefix in self._renames:
            if old_module_name == old_prefix or old_module_name.startswith(old_prefix + "."):
                return new_prefix + old_module_name[len(old_prefix) :]
        return None

    def find_spec(
        self,
        fullname: str,
        path: Sequence[str] | None = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        new_name: Final = self.new_name_for(fullname)
        if new_name is None:
            return None
        module: Final = importlib.import_module(new_name)
        spec: Final = spec_from_loader(fullname, _AliasLoader(module), is_package=hasattr(module, "__path__"))
        if spec is not None and spec.submodule_search_locations is not None:
            spec.submodule_search_locations = list(module.__path__)  # pyright: ignore[reportAny]  # module attributes are untyped
        return spec


def alias_renamed_package(
    old_name: str,
    new_name: str,
    *,
    nested_renames: tuple[tuple[str, str], ...] = (),
) -> Callable[[str], ModuleType]:
    """Make every module of the package now at ``new_name`` importable under its old dotted name.

    Nothing is imported up front: a meta path finder resolves ``old_name.x.y`` to the module object at
    ``new_name.x.y`` the moment something imports it, so a deep import of the old path is the very same object
    as the new path and importing one old module loads no more than importing its new counterpart would.
    ``nested_renames`` lists (old, new) dotted prefixes for sub-packages that changed name inside the move,
    longest first. The returned function is the shim package's module ``__getattr__``, so attribute access on
    the old package reaches the renamed sub-packages too.
    """
    warnings.warn(
        f"{old_name} moved to {new_name}; update imports, the old path will be removed in a future release",
        DeprecationWarning,
        stacklevel=3,
    )
    finder: Final = _RenamedPackageFinder(old_name, (*nested_renames, (old_name, new_name)))
    if not any(
        isinstance(entry, _RenamedPackageFinder) and entry.new_name_for(old_name + ".x") for entry in sys.meta_path
    ):
        sys.meta_path.insert(0, finder)

    def getattr_(name: str) -> ModuleType:
        try:
            return importlib.import_module(f"{old_name}.{name}")
        except ModuleNotFoundError as error:
            raise AttributeError(f"module {old_name!r} has no attribute {name!r}") from error

    return getattr_
