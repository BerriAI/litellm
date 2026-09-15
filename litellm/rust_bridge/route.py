from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Generic, NoReturn, TypeAlias, TypeVar

from typing_extensions import assert_never

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


@dataclass(frozen=True, slots=True)
class ComponentBinding(Generic[BindingT]):
    component: ComponentName
    native: NativeBinding[BindingT]


@dataclass(frozen=True, slots=True)
class PythonRoute:
    pass


@dataclass(frozen=True, slots=True)
class NativeRoute(Generic[BindingT]):
    native: BindingT


@dataclass(frozen=True, slots=True)
class RouteUnsupported:
    component: ComponentName


@dataclass(frozen=True, slots=True)
class RouteUnavailable:
    component: ComponentName


@dataclass(frozen=True, slots=True)
class RouteMismatch:
    component: ComponentName
    binding: ComponentName


RouteFailure: TypeAlias = RouteUnsupported | RouteUnavailable | RouteMismatch


def raise_route_failure(failure: RouteFailure) -> NoReturn:
    match failure:
        case RouteUnsupported(component):
            raise RustRouteUnsupportedError(f"No Python or Rust implementation for {component.value}")
        case RouteUnavailable(component):
            raise RustRouteUnavailableError(f"Rust {component.value} bridge is unavailable")
        case RouteMismatch(component, binding):
            raise ValueError(f"{binding.value} binding cannot serve a {component.value} route")
        case _:
            assert_never(failure)


@dataclass(frozen=True, slots=True)
class ComponentExecution:
    component: ComponentName
    decision: ExecutionDecision

    def select(self, binding: ComponentBinding[BindingT]) -> PythonRoute | NativeRoute[BindingT] | RouteFailure:
        if binding.component is not self.component:
            return RouteMismatch(component=self.component, binding=binding.component)
        match self.decision:
            case ExecutionDecision.UNSUPPORTED:
                return RouteUnsupported(self.component)
            case ExecutionDecision.PYTHON:
                return PythonRoute()
            case ExecutionDecision.RUST_WITH_FALLBACK:
                native: Final = binding.native.load()
                return PythonRoute() if native is None else NativeRoute(native)
            case ExecutionDecision.RUST_REQUIRED:
                required: Final = binding.native.load()
                return RouteUnavailable(self.component) if required is None else NativeRoute(required)
        return assert_never(self.decision)


@dataclass(frozen=True, slots=True)
class NativeComponent:
    name: ComponentName
    capability: CapabilitySpec
    exports: tuple[str, ...]

    def resolve(self, context: CapabilityContext | None = None) -> ComponentExecution:
        return ComponentExecution(
            component=self.name,
            decision=capability_decision(
                self.capability, context=context if context is not None else CapabilityContext()
            ),
        )

    def bind(self, export: str, *, validate: Callable[[object], BindingT | None]) -> ComponentBinding[BindingT]:
        if export not in self.exports:
            raise ValueError(f"native export {export!r} is not declared for {self.name.value}")
        return ComponentBinding(component=self.name, native=NativeBinding(export, validate=validate))
