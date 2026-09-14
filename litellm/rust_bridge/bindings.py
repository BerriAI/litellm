from __future__ import annotations

from collections.abc import Callable
from types import ModuleType
from typing import Final, Generic, TypeVar

from litellm.rust_bridge.loader import get_native_bridge

BindingT = TypeVar("BindingT")


class BindingUnset:
    pass


BINDING_UNSET: Final = BindingUnset()


class NativeBinding(Generic[BindingT]):
    """Resolve one native attribute with an explicit, resettable test override."""

    def __init__(
        self,
        attribute: str,
        *,
        validate: Callable[[object], BindingT | None],
        module_loader: Callable[[], ModuleType | None] | None = None,
    ) -> None:
        self._attribute: Final = attribute
        self._validate: Final = validate
        self._module_loader: Final = module_loader
        self._override: BindingT | None | BindingUnset = BINDING_UNSET

    def load(self) -> BindingT | None:
        if not isinstance(self._override, BindingUnset):
            return self._override
        native: Final = self._module_loader() if self._module_loader is not None else get_native_bridge()
        if native is None:
            return None
        return self._validate(getattr(native, self._attribute, None))

    def override(self, value: BindingT | None) -> None:
        self._override = value

    def reset(self) -> None:
        self._override = BINDING_UNSET

    def configure(self, value: BindingT | None | BindingUnset) -> None:
        if isinstance(value, BindingUnset):
            return
        if value is None:
            self.reset()
            return
        self.override(value)


def native_exception_types() -> tuple[type[BaseException], type[BaseException]] | None:
    native: Final = get_native_bridge()
    if native is None:
        return None
    declined: Final = getattr(native, "RustBridgeDeclined", None)
    upstream: Final = getattr(native, "RustUpstreamError", None)
    if not isinstance(declined, type) or not isinstance(upstream, type):
        return None
    return declined, upstream
