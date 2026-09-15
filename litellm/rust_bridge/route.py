from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Final, Generic, TypeVar

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
class ComponentExecution:
    component: ComponentName
    decision: ExecutionDecision

    def require_supported(self) -> None:
        if self.decision is ExecutionDecision.UNSUPPORTED:
            raise RustRouteUnsupportedError(f"No Python or Rust implementation for {self.component.value}")

    def select(self, binding: ComponentBinding[BindingT]) -> BindingT | None:
        if binding.component is not self.component:
            raise ValueError(f"{binding.component.value} binding cannot serve a {self.component.value} route")
        self.require_supported()
        if self.decision is ExecutionDecision.PYTHON:
            return None
        selected: Final = binding.native.load()
        if selected is None and self.decision is ExecutionDecision.RUST_REQUIRED:
            raise RustRouteUnavailableError(f"Rust {self.component.value} bridge is unavailable")
        return selected


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
