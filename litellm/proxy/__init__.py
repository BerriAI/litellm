from types import ModuleType
from typing import Final


def __getattr__(name: str) -> ModuleType:
    from litellm._lazy_imports import lazy_import_submodule

    submodule: Final = lazy_import_submodule(__name__, name)
    if submodule is not None:
        return submodule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
