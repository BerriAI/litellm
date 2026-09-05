from __future__ import annotations

from collections.abc import Callable
from typing import Final, Generic, TypeVar, cast  # noqa: TID251  # PyO3 module boundary

from litellm.rust_bridge.loader import get_native_bridge
from litellm.rust_bridge.protocols import NativeModule

BindingT = TypeVar("BindingT")


class _Unset:
    pass


_UNSET: Final = _Unset()


class Unchanged:
    pass


UNCHANGED: Final = Unchanged()


class NativeBinding(Generic[BindingT]):
    """Resolve one native attribute with an explicit, resettable test override."""

    def __init__(self, select: Callable[[NativeModule], BindingT]) -> None:
        self._select: Final = select
        self._override: BindingT | None | _Unset = _UNSET

    def load(self) -> BindingT | None:
        if not isinstance(self._override, _Unset):
            return self._override
        native: Final = get_native_bridge()
        if native is None:
            return None
        module: Final = cast(NativeModule, native)  # cast-ok: PyO3 exports are validated individually below
        try:
            value: Final = self._select(module)
        except AttributeError:
            return None
        return value if callable(value) else None

    def override(self, value: BindingT | None) -> None:
        self._override = value

    def reset(self) -> None:
        self._override = _UNSET


_DECLINED: Final = NativeBinding(lambda native: native.RustBridgeDeclined)
_UPSTREAM: Final = NativeBinding(lambda native: native.RustUpstreamError)


def _exception_class(value: object) -> type[BaseException] | None:
    if isinstance(value, type) and issubclass(value, BaseException):
        return value
    return None


def native_exception_types() -> tuple[type[BaseException], type[BaseException]] | None:
    declined: Final = _exception_class(_DECLINED.load())
    upstream: Final = _exception_class(_UPSTREAM.load())
    if declined is None or upstream is None:
        return None
    return declined, upstream
