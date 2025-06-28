import importlib.machinery
import os
from dataclasses import dataclass
from types import ModuleType
from typing import TYPE_CHECKING, Any, Dict, FrozenSet, List, Set, TypeVar, Union

BACKENDS_T = FrozenSet[str]
ImportStructure = Dict[BACKENDS_T, Dict[str, Set[str]]]


# Optional mapping for partial enabling of backends for future implementation
BACKEND_MAPPING: Dict[str, Any] = {}


class _Alias:
    def __init__(self, name: str, alias: str):
        self.name = name
        self.alias = alias


if TYPE_CHECKING:
    _ImportStructure = Dict[str, List[str | _Alias]]
else:

    class _ImportStructure(dict):
        """
        Type hint for the import structure of a module.
        This is used to define the structure of the module and its imports.
        """

        def __setitem__(self, key, value):
            if key in self:
                raise KeyError(f"Key '{key}' already exists in the import structure.")
            return super().__setitem__(key, value)


class _LazyModule(ModuleType):
    """
    Module class that surfaces all objects but only performs associated imports when the objects are requested.
    """

    def __init__(
        self,
        name: str,
        module_file: str,
        import_structure: _ImportStructure,
        module_spec: importlib.machinery.ModuleSpec,
        extra_object: Dict[str, Any] = {},
    ):
        super().__init__(name)
        self._modules: FrozenSet[str] = frozenset(import_structure.keys())

        self._object_to_module: Dict[str, str] = {}
        self._aliases: Dict[str, str] = {}
        self._extra_object: Dict[str, Any] = extra_object

        for module_name, names in import_structure.items():
            for object_name in names:
                if isinstance(object_name, _Alias):
                    self._object_to_module[object_name.alias] = module_name
                    self._aliases[object_name.alias] = object_name.name
                else:
                    self._object_to_module[object_name] = module_name
        # Store submodules for lazy loading
        submodules = []
        for module_name in self._modules:
            parts = []
            for module_part in module_name.split("."):
                parts.append(module_part)
                submodules.append(".".join(parts))
        self._modules = self._modules.union(submodules)

        # Needed for autocompletion in an IDE and wildcard imports (although those won't be lazy)
        self.__all__ = [
            *self._modules,
            *self._object_to_module.keys(),
            *extra_object.keys(),
        ]
        self.__file__ = module_file
        self.__path__ = [os.path.dirname(module_file)]
        self.__spec__ = module_spec
        self._name = name
        self._import_structure = import_structure

    _cached_modules: dict[str, "_LazyModule"] = {}  # type: ignore[assignment]

    @classmethod
    def create(
        cls,
        name: str,
        module_file: str,
        import_structure: _ImportStructure,
        module_spec: importlib.machinery.ModuleSpec,
        extra_object: Dict[str, Any] = {},
    ) -> "_LazyModule":
        """
        Create a new _LazyModule instance.
        This method is used to create a new _LazyModule instance.
        """
        if module_file not in cls._cached_modules:
            cls._cached_modules[module_file] = cls(
                name,
                module_file,
                import_structure,
                module_spec,
                extra_object=extra_object,
            )
        return cls._cached_modules[module_file]

    # Needed for autocompletion in an IDE
    def __dir__(self):
        result = super().__dir__()
        # return [name for name in self.__all__ if name not in result
        return [name for name in self.__all__]

    def __getattr__(self, name: str) -> Any:
        if name in self._object_to_module:
            module_name = self._object_to_module[name]
            value = getattr(
                self._get_module(module_name), self._aliases.get(name, name)
            )
        elif name in self._modules:
            value = self._get_module(name)
        elif name in self._extra_object:
            value = self._extra_object[name]
        else:
            # Check if the name is an submodule
            submodule_name = f"litellm.{name}"
            if submodule_name in self._modules:
                value = self._get_module(submodule_name)
            elif name in ("_object_to_module", "_modules", "_aliases"):
                # Allow access to internal attributes
                return getattr(self, name)
            else:
                raise AttributeError(
                    f"module '{self.__name__}' has no attribute '{name}'"
                )

        setattr(self, name, value)
        return value

    def _get_module(self, module_name: str) -> ModuleType:
        import importlib

        try:
            return importlib.import_module(module_name, self.__name__)
        except ModuleNotFoundError as e:
            raise RuntimeError(
                f"Failed to import {self.__name__}.{module_name} because of the following error (look up to see its"
                f" traceback):\n{e}"
            ) from e

    def __reduce__(self):
        return (
            self.__class__,
            (self._name, self.__file__, self._import_structure, self._object_to_module),
        )