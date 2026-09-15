from __future__ import annotations

from collections.abc import Callable, Coroutine
from dataclasses import dataclass
from types import ModuleType
from typing import Final, Literal, Protocol, TypeVar, cast, overload

from litellm.rust_bridge.bindings import NativeBinding
from litellm.rust_bridge.configuration import (
    CapabilityContext,
    CapabilitySpec,
    ComponentName,
    ExecutionDecision,
    capability_decision,
)
from litellm.rust_bridge.errors import RustRouteUnavailableError, RustRouteUnsupportedError

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
        host: object,
    ) -> ResponseT: ...

    @overload
    def __call__(
        self,
        request: RequestT,
        args: tuple[object, ...],
        kwargs: dict[str, object],  # mutable-ok: PyO3 requires the original concrete dict
        asynchronous: Literal[True],
        host: object,
    ) -> Coroutine[object, object, ResponseT]: ...

    @overload
    def __call__(
        self,
        request: RequestT,
        args: tuple[object, ...],
        kwargs: dict[str, object],  # mutable-ok: PyO3 requires the original concrete dict
        asynchronous: bool,
        host: object,
    ) -> ResponseT | Coroutine[object, object, ResponseT]: ...


def _lifecycle(value: object) -> NativeLifecycle[object, object] | None:
    if not callable(value):
        return None
    return cast(NativeLifecycle[object, object], value)  # cast-ok: callable native entrypoint validated above


@dataclass(frozen=True, slots=True)
class ComponentExecution:
    route_name: ComponentName
    decision: ExecutionDecision

    def require_supported(self) -> None:
        if self.decision is ExecutionDecision.UNSUPPORTED:
            raise RustRouteUnsupportedError(
                f"No Python or Rust implementation for {self.route_name.value}"
            )

    def select(self, binding: NativeBinding[BindingT]) -> BindingT | None:
        self.require_supported()
        if self.decision is ExecutionDecision.PYTHON:
            return None
        selected: Final = binding.load()
        if selected is None and self.decision is ExecutionDecision.RUST_REQUIRED:
            raise RustRouteUnavailableError(
                f"Rust {self.route_name.value} bridge is unavailable"
            )
        return selected


@dataclass(frozen=True, slots=True)
class NativeComponent:
    name: ComponentName
    capability: CapabilitySpec
    exports: tuple[str, ...]

    def resolve(self, context: CapabilityContext = CapabilityContext()) -> ComponentExecution:
        return ComponentExecution(
            route_name=self.name,
            decision=capability_decision(self.capability, context=context),
        )

    def bind(
        self,
        export: str,
        *,
        validate: Callable[[object], BindingT | None],
        module_loader: Callable[[], ModuleType | None] | None = None,
    ) -> NativeBinding[BindingT]:
        if export not in self.exports:
            raise ValueError(f"native export {export!r} is not declared for {self.name.value}")
        return NativeBinding(export, validate=validate, module_loader=module_loader)

    def lifecycle(self) -> NativeBinding[NativeLifecycle[object, object]]:
        export: Final = f"_{self.name.value}_lifecycle"
        return self.bind(export, validate=_lifecycle)
