"""Console entry points for `lite` and `litellm-proxy`.

The CLI is a thin HTTP client, but it lives inside the `litellm` package, so a plain
import would run the whole SDK `__init__` before Click parses argv. These entry points
import the CLI against an uninitialized `litellm` package object instead. CLI code under
`litellm.proxy.client`, and whatever it imports, loads without the SDK whenever it is
first imported. The first attribute read on the SDK, or the first import of any other
new code under `litellm`, initializes the SDK in place before that code runs, so code
that needs the SDK sees it exactly as a plain `import litellm` would leave it. An init that
raises leaves the SDK uninitialized, so the next use runs it again and raises the same error.
"""

import builtins
import contextvars
import importlib
import importlib.util
import sys
import threading
import types
from collections.abc import Callable, Mapping, Sequence
from typing import Final

SDK_PACKAGE: Final = "litellm"
CLI_PACKAGE: Final = "litellm.proxy.client"
_INITIALIZATION_LOCK: Final = threading.RLock()
_BUILTIN_IMPORT: Final = builtins.__import__
_IMPORT_MODULE: Final = importlib.import_module
_LOADING_CLI_CODE: Final = contextvars.ContextVar("_LOADING_CLI_CODE", default=False)


def _is_dunder(name: str) -> bool:
    return name.startswith("__") and name.endswith("__")


class _SDKUntilFirstUse(types.ModuleType):
    def __getattribute__(self, name: str) -> object:
        if not _is_dunder(name):
            _initialize(self)
        return types.ModuleType.__getattribute__(self, name)


class _SDKInitializing(types.ModuleType):
    def __getattr__(self, name: str) -> object:
        with _INITIALIZATION_LOCK:
            if type(self) is _SDKInitializing:
                raise AttributeError(f"partially initialized module {SDK_PACKAGE!r} has no attribute {name!r}")
        return getattr(self, name)


def _become(sdk: types.ModuleType, state: type[types.ModuleType]) -> None:
    sdk.__class__ = state  # rebind-ok: every holder of the litellm module must see its state change in place


def _restore_interpreter_imports() -> None:
    if builtins.__import__ is _import_initializing_sdk:
        builtins.__import__ = _BUILTIN_IMPORT
    if importlib.import_module is _import_module_initializing_sdk:
        importlib.import_module = _IMPORT_MODULE


def _initialize(sdk: types.ModuleType) -> None:
    with _INITIALIZATION_LOCK:
        if type(sdk) is not _SDKUntilFirstUse:
            return
        spec: Final = sdk.__spec__
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"No module named {SDK_PACKAGE!r}", name=SDK_PACKAGE)
        _become(sdk, _SDKInitializing)
        try:
            spec.loader.exec_module(sdk)
        except BaseException:
            _become(sdk, _SDKUntilFirstUse)
            raise
        _become(sdk, types.ModuleType)
        _restore_interpreter_imports()


def _in_package(target: str, package: str) -> bool:
    return target == package or target.startswith(f"{package}.")


def _needs_more_of(loaded: types.ModuleType, name: str) -> bool:
    if name == "*":
        return type(loaded) is _SDKUntilFirstUse
    return name not in vars(loaded)


def _loads_new_sdk_code(target: str, fromlist: Sequence[str]) -> bool:
    if not _in_package(target, SDK_PACKAGE):
        return False
    loaded: Final = sys.modules.get(target)
    if loaded is None:
        return True
    return any(_needs_more_of(loaded, name) for name in fromlist)


def _load_cli_code(load: Callable[[], types.ModuleType]) -> types.ModuleType:
    token: Final = _LOADING_CLI_CODE.set(True)
    try:
        return load()
    finally:
        _LOADING_CLI_CODE.reset(token)


def _imported(target: str | None, fromlist: Sequence[str], load: Callable[[], types.ModuleType]) -> types.ModuleType:
    if _LOADING_CLI_CODE.get() or target is None or not _loads_new_sdk_code(target, fromlist):
        return load()
    if _in_package(target, CLI_PACKAGE):
        return _load_cli_code(load)
    sdk: Final = sys.modules.get(SDK_PACKAGE)
    if sdk is not None:
        _initialize(sdk)
    return load()


def _import_target(name: str, importer_globals: Mapping[str, object] | None, level: int) -> str | None:
    if level == 0:
        return name
    package: Final = (importer_globals or {}).get("__package__")
    if not isinstance(package, str) or not package:
        return None
    return importlib.util.resolve_name("." * level + name, package)


def _import_initializing_sdk(
    name: str,
    globals: Mapping[str, object] | None = None,
    locals: Mapping[str, object] | None = None,
    fromlist: Sequence[str] | None = (),
    level: int = 0,
) -> types.ModuleType:
    return _imported(
        _import_target(name, globals, level),
        fromlist or (),
        lambda: _BUILTIN_IMPORT(name, globals, locals, fromlist or (), level),
    )


def _import_module_target(name: str, package: str | None) -> str | None:
    if not name.startswith("."):
        return name
    if not package:
        return None
    return importlib.util.resolve_name(name, package)


def _import_module_initializing_sdk(name: str, package: str | None = None) -> types.ModuleType:
    return _imported(_import_module_target(name, package), (), lambda: _IMPORT_MODULE(name, package))


def _install_sdk_until_first_use() -> None:
    sdk_spec: Final = importlib.util.find_spec(SDK_PACKAGE)
    if sdk_spec is None:
        raise ModuleNotFoundError(f"No module named {SDK_PACKAGE!r}", name=SDK_PACKAGE)
    sdk: Final = importlib.util.module_from_spec(sdk_spec)
    sdk.__class__ = _SDKUntilFirstUse
    sys.modules[SDK_PACKAGE] = sdk

    from litellm._process_defaults import apply_process_defaults

    apply_process_defaults()
    builtins.__import__ = _import_initializing_sdk
    importlib.import_module = _import_module_initializing_sdk


def _entry_points() -> tuple[Callable[[], None], Callable[[], None]]:
    if SDK_PACKAGE not in sys.modules:
        _install_sdk_until_first_use()

    from litellm.proxy.client.cli.main import cli as lite
    from litellm.proxy.client.cli.main import litellm_proxy_cli as deprecated_litellm_proxy

    return lite, deprecated_litellm_proxy


def cli() -> None:
    lite, _ = _entry_points()
    lite()


def litellm_proxy_cli() -> None:
    _, deprecated_litellm_proxy = _entry_points()
    deprecated_litellm_proxy()
