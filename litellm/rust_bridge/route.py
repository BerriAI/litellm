from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from types import ModuleType
from typing import (
    Final,
    Literal,
    Protocol,
    TypeVar,
    cast,  # noqa: TID251  # validate callability at the native boundary
    overload,
)

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.configuration import ROUTE_POLICIES, RouteName, RoutePolicy, rust_enabled

BindingT = TypeVar("BindingT")
RequestT = TypeVar("RequestT", contravariant=True)
ResponseT = TypeVar("ResponseT", covariant=True)


class NativeLifecycle(Protocol[RequestT, ResponseT]):
    @overload
    def __call__(
        self,
        request: RequestT,
        args: tuple[object, ...],
        kwargs: dict[str, object],  # mutable-ok: PyO3 requires the original concrete dict
        asynchronous: Literal[False],
    ) -> ResponseT: ...

    @overload
    def __call__(
        self,
        request: RequestT,
        args: tuple[object, ...],
        kwargs: dict[str, object],  # mutable-ok: PyO3 requires the original concrete dict
        asynchronous: Literal[True],
    ) -> Coroutine[object, object, ResponseT]: ...

    @overload
    def __call__(
        self,
        request: RequestT,
        args: tuple[object, ...],
        kwargs: dict[str, object],  # mutable-ok: PyO3 requires the original concrete dict
        asynchronous: bool,
    ) -> ResponseT | Coroutine[object, object, ResponseT]: ...


def _lifecycle(value: object) -> NativeLifecycle[object, object] | None:
    if not callable(value):
        return None
    return cast(NativeLifecycle[object, object], value)  # cast-ok: callable native entrypoint validated above


@dataclass(frozen=True, slots=True)
class NativeRoute:
    name: RouteName

    @property
    def policy(self) -> RoutePolicy:
        return ROUTE_POLICIES[self.name]

    def enabled(self) -> bool:
        return rust_enabled(self.name)

    def bind(
        self,
        export: str,
        *,
        validate: Callable[[object], BindingT | None],
        module_loader: Callable[[], ModuleType | None] | None = None,
    ) -> NativeBinding[BindingT]:
        return NativeBinding(export, validate=validate, module_loader=module_loader)

    def select(self, binding: NativeBinding[BindingT]) -> BindingT | None:
        return binding.load() if self.enabled() else None

    def lifecycle(self) -> NativeBinding[NativeLifecycle[object, object]]:
        export: Final = f"_{self.name.value}_lifecycle"
        return self.bind(export, validate=_lifecycle)
