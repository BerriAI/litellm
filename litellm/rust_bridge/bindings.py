from __future__ import annotations

from collections.abc import Callable
from typing import Final, Generic, TypeVar

from litellm.rust_bridge import _native

BindingT = TypeVar("BindingT")


class _Unset:
    pass


_UNSET: Final = _Unset()


class NativeBinding(Generic[BindingT]):
    """Resolve one native attribute with an explicit, resettable test override."""

    def __init__(
        self, attribute: str, *, validate: Callable[[object], BindingT | None], source: object = _native
    ) -> None:
        self._attribute: Final = attribute
        self._validate: Final = validate
        self._source: Final = source
        self._override: BindingT | None | _Unset = _UNSET

    def load(self) -> BindingT | None:
        if not isinstance(self._override, _Unset):
            return self._override
        return self._validate(getattr(self._source, self._attribute, None))

    def override(self, value: BindingT | None) -> None:
        self._override = value

    def reset(self) -> None:
        self._override = _UNSET
