"""Console entry points for `lite` and `litellm-proxy`.

The CLI is a thin HTTP client, but it lives inside the `litellm` package, so a plain
import would run the whole SDK `__init__` before Click parses argv. These entry points
import the CLI against an uninitialized `litellm` package object instead. The first
attribute access on the SDK, or, once the CLI is imported, the first import that would
load anything new under `litellm`, initializes the SDK in place before that code runs,
so code that needs the SDK sees it exactly as a plain `import litellm` would leave it.
"""

import builtins
import importlib.util
import sys
import threading
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Final

SDK_PACKAGE: Final = "litellm"
_SDK_SUBMODULE_PREFIX: Final = f"{SDK_PACKAGE}."
_INITIALIZATION_LOCK: Final = threading.RLock()
_BUILTIN_IMPORT: Final = builtins.__import__


class _SDKUntilFirstUse(types.ModuleType):
    def __getattr__(self, name: str) -> object:
        _initialize(self)
        return getattr(self, name)


class _SDKInitializing(types.ModuleType):
    def __getattr__(self, name: str) -> object:
        with _INITIALIZATION_LOCK:
            if type(self) is _SDKInitializing:
                raise AttributeError(f"partially initialized module {SDK_PACKAGE!r} has no attribute {name!r}")
        return getattr(self, name)


def _become(sdk: types.ModuleType, state: type[types.ModuleType]) -> None:
    sdk.__class__ = state


def _initialize(sdk: types.ModuleType) -> None:
    with _INITIALIZATION_LOCK:
        if type(sdk) is not _SDKUntilFirstUse:
            return
        spec: Final = sdk.__spec__
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"No module named {SDK_PACKAGE!r}", name=SDK_PACKAGE)
        _become(sdk, _SDKInitializing)
        if builtins.__import__ is _import_initializing_sdk:
            builtins.__import__ = _BUILTIN_IMPORT
        try:
            spec.loader.exec_module(sdk)
        finally:
            _become(sdk, types.ModuleType)


def _import_target(name: str, importer_globals: Mapping[str, object] | None, level: int) -> str | None:
    if level == 0:
        return name
    package: Final = (importer_globals or {}).get("__package__")
    if not isinstance(package, str) or not package:
        return None
    return importlib.util.resolve_name("." * level + name, package)


def _loads_new_sdk_code(target: str, fromlist: Sequence[str]) -> bool:
    if target != SDK_PACKAGE and not target.startswith(_SDK_SUBMODULE_PREFIX):
        return False
    loaded: Final = sys.modules.get(target)
    if loaded is None:
        return True
    return any(name != "*" and name not in vars(loaded) for name in fromlist)


def _import_initializing_sdk(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] | None = (),
    level: int = 0,
) -> types.ModuleType:
    target: Final = _import_target(name, globals, level)
    sdk: Final = sys.modules.get(SDK_PACKAGE)
    if target is not None and sdk is not None and _loads_new_sdk_code(target, fromlist or ()):
        _initialize(sdk)
    return _BUILTIN_IMPORT(name, globals, locals, fromlist or (), level)


def _imported_without_sdk() -> tuple[Callable[[], None], Callable[[], None]]:
    sdk_spec: Final = importlib.util.find_spec(SDK_PACKAGE)
    if sdk_spec is None:
        raise ModuleNotFoundError(f"No module named {SDK_PACKAGE!r}", name=SDK_PACKAGE)
    sdk: Final = importlib.util.module_from_spec(sdk_spec)
    sdk.__class__ = _SDKUntilFirstUse
    sys.modules[SDK_PACKAGE] = sdk

    from litellm._process_defaults import apply_process_defaults

    apply_process_defaults()

    from litellm.proxy.client.cli.main import cli as lite
    from litellm.proxy.client.cli.main import litellm_proxy_cli as deprecated_litellm_proxy

    builtins.__import__ = _import_initializing_sdk
    return lite, deprecated_litellm_proxy


def cli() -> None:
    lite, _ = _imported_without_sdk()
    lite()


def litellm_proxy_cli() -> None:
    _, deprecated_litellm_proxy = _imported_without_sdk()
    deprecated_litellm_proxy()
