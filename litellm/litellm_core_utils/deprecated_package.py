import importlib
import pkgutil
import sys
import warnings
from types import ModuleType


def alias_deprecated_package(old_name: str, new_package: ModuleType) -> None:
    """Make every submodule of ``new_package`` importable under ``old_name`` and warn once."""
    warnings.warn(
        f"{old_name} is deprecated, import {new_package.__name__} instead",
        DeprecationWarning,
        stacklevel=3,
    )
    sys.modules.update(
        {
            f"{old_name}{info.name.removeprefix(new_package.__name__)}": importlib.import_module(info.name)
            for info in pkgutil.walk_packages(new_package.__path__, prefix=f"{new_package.__name__}.")
        }
    )
