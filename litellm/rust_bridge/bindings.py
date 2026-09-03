from __future__ import annotations

from collections.abc import Callable
from typing import Final, Generic, TypeVar

from litellm.rust_bridge.loader import get_native_bridge

BindingT = TypeVar("BindingT")


class _Unset:
    pass


_UNSET: Final = _Unset()


class NativeBinding(Generic[BindingT]):
    """Resolve one native attribute with an explicit, resettable test override."""

    def __init__(self, attribute: str, *, validate: Callable[[object], BindingT | None]) -> None:
        self._attribute: Final = attribute
        self._validate: Final = validate
        self._override: BindingT | None | _Unset = _UNSET

    def load(self) -> BindingT | None:
        if not isinstance(self._override, _Unset):
            return self._override
        native: Final = get_native_bridge()
        if native is None:
            return None
        return self._validate(getattr(native, self._attribute, None))

    def override(self, value: BindingT | None) -> None:
        self._override = value

    def reset(self) -> None:
        self._override = _UNSET


def native_exception_types() -> tuple[type[BaseException], type[BaseException]] | None:
    native: Final = get_native_bridge()
    if native is None:
        return None
    declined: Final = getattr(native, "RustBridgeDeclined", None)
    upstream: Final = getattr(native, "RustUpstreamError", None)
    if not isinstance(declined, type) or not isinstance(upstream, type):
        return None
    return declined, upstream
