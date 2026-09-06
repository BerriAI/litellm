from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Final, Generic, TypeVar, cast  # noqa: TID251  # PyO3 module boundary

from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.protocols import NativeModule

BindingT = TypeVar("BindingT")


class NativeBinding(Generic[BindingT]):
    """Resolve one callable exported by the native module."""

    def __init__(
        self,
        select: Callable[[NativeModule], BindingT],
        *,
        module_loader: Callable[[], ModuleType | None] | None = None,
    ) -> None:
        self._select: Final = select
        self._module_loader: Final = module_loader

    def load(self) -> BindingT | None:
        native: Final = self._module_loader() if self._module_loader is not None else get_native_bridge()
        if native is None:
            return None
        module: Final = cast(NativeModule, native)  # cast-ok: PyO3 exports are validated individually below
        try:
            value: Final = self._select(module)
        except AttributeError:
            return None
        return value if callable(value) else None


_DECLINED: Final = NativeBinding(lambda native: native.RustBridgeDeclined)
_UPSTREAM: Final = NativeBinding(lambda native: native.RustUpstreamError)


def _exception_class(value: object) -> type[BaseException] | None:
    if isinstance(value, type) and issubclass(value, BaseException):
        return value
    return None


def native_upstream_types() -> tuple[type[BaseException], ...]:
    upstream: Final = _exception_class(_UPSTREAM.load())
    return () if upstream is None else (upstream,)


def native_declined_types() -> tuple[type[BaseException], ...]:
    declined: Final = _exception_class(_DECLINED.load())
    return () if declined is None else (declined,)
