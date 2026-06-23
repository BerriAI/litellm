import importlib
from functools import lru_cache
from types import ModuleType
from typing import Any, Iterable, Union

_enabled_rust_core_scopes: set[str] = set()
_rust_core_strict = False


@lru_cache(maxsize=1)
def _load_rust_module() -> ModuleType | None:
    try:
        return importlib.import_module("litellm_python_bridge")
    except Exception:
        return None


def rust_core_available() -> bool:
    return _load_rust_module() is not None


def set_rust_core_enabled(scopes: Union[bool, str, Iterable[str]]) -> None:
    global _enabled_rust_core_scopes

    if scopes is True:
        _enabled_rust_core_scopes = {"all"}
        return
    if scopes is False:
        _enabled_rust_core_scopes = set()
        return
    if isinstance(scopes, str):
        _enabled_rust_core_scopes = {
            scope.strip()
            for scope in scopes.replace(";", ",").split(",")
            if scope.strip()
        }
        return

    _enabled_rust_core_scopes = {scope for scope in scopes if scope}


def rust_core_enabled(scope: str) -> bool:
    return "all" in _enabled_rust_core_scopes or scope in _enabled_rust_core_scopes


def set_rust_core_strict(enabled: bool) -> None:
    global _rust_core_strict
    _rust_core_strict = enabled


def call_rust_function(function_name: str, *args: Any) -> Any | None:
    module = _load_rust_module()
    if module is None:
        return None

    try:
        return getattr(module, function_name)(*args)
    except Exception:
        if _rust_core_strict:
            raise
        return None
