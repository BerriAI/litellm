import importlib
import os
from types import ModuleType
from typing import Any, Optional

_RUST_MODULE: Optional[ModuleType] = None
_RUST_MODULE_LOAD_ATTEMPTED = False


def _load_rust_module() -> Optional[ModuleType]:
    global _RUST_MODULE, _RUST_MODULE_LOAD_ATTEMPTED

    if _RUST_MODULE_LOAD_ATTEMPTED:
        return _RUST_MODULE

    _RUST_MODULE_LOAD_ATTEMPTED = True
    try:
        _RUST_MODULE = importlib.import_module("litellm_python_bridge")
    except Exception:
        _RUST_MODULE = None
    return _RUST_MODULE


def rust_core_available() -> bool:
    return _load_rust_module() is not None


def rust_core_enabled(scope: str) -> bool:
    flag_value = os.getenv("LITELLM_USE_RUST_CORE", "").strip().lower()
    if flag_value in {"1", "true", "all"}:
        return True

    enabled_scopes = {
        scope_name.strip()
        for scope_name in flag_value.replace(";", ",").split(",")
        if scope_name.strip()
    }
    return scope in enabled_scopes


def strict_mode_enabled() -> bool:
    return os.getenv("LITELLM_RUST_CORE_STRICT", "").strip().lower() in {
        "1",
        "true",
    }


def call_rust_function(function_name: str, *args: Any) -> Optional[Any]:
    module = _load_rust_module()
    if module is None:
        return None

    try:
        return getattr(module, function_name)(*args)
    except Exception:
        if strict_mode_enabled():
            raise
        return None
