from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.configuration import (
    CapabilityContext,
    CapabilityDefinition,
    ComponentName,
    DeliveryMode,
    ExecutionDecision,
    RolloutPolicy,
    RustImplementationState,
)
from litellm.rust_bridge.errors import RustRouteUnavailableError, RustRouteUnsupportedError
from litellm.rust_bridge.route import (
    ComponentBinding,
    ComponentExecution,
    NativeComponent,
    NativeRoute,
    PythonRoute,
    RouteFailure,
    RouteMismatch,
    RouteUnavailable,
    RouteUnsupported,
    raise_route_failure,
)

OPT_IN: Final = CapabilityDefinition(
    rust=RustImplementationState.EXPERIMENTAL, python_available=True, rollout=RolloutPolicy.RUST_OPT_IN
)
RUST_REQUIRED: Final = CapabilityDefinition(
    rust=RustImplementationState.EXPERIMENTAL, python_available=False, rollout=RolloutPolicy.RUST_REQUIRED
)
UNSUPPORTED: Final = CapabilityDefinition(
    rust=RustImplementationState.UNIMPLEMENTED, python_available=False, rollout=RolloutPolicy.UNSUPPORTED
)


@pytest.fixture(autouse=True)
def _isolated_configuration(  # pyright: ignore[reportUnusedFunction]  # pytest discovers fixtures dynamically
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None]:
    configuration.reset_rust_configuration()
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    yield
    configuration.reset_rust_configuration()


def _component(name: ComponentName) -> NativeComponent:
    return NativeComponent(name=name, capability=OPT_IN, exports=("route",))


def _binding(name: ComponentName, native: object | None) -> ComponentBinding[object]:
    return _component(name).bind("route", validate=lambda value: value, loader=lambda: native)


def _unloadable_binding(name: ComponentName) -> ComponentBinding[object]:
    return _component(name).bind(
        "route", validate=lambda value: value, loader=lambda: pytest.fail("native must not be loaded")
    )


def _failure(route: PythonRoute | NativeRoute[object] | RouteFailure) -> RouteFailure:
    assert not isinstance(route, PythonRoute | NativeRoute)
    return route


def test_resolve_applies_process_override_to_optional_component() -> None:
    component: Final = _component(ComponentName.CHAT_COMPLETIONS)

    assert component.resolve().decision is ExecutionDecision.PYTHON
    configuration.rust(True)
    assert component.resolve() == ComponentExecution(
        component=ComponentName.CHAT_COMPLETIONS, decision=ExecutionDecision.RUST_WITH_FALLBACK
    )


def test_resolve_passes_context_to_capability_resolver() -> None:
    def capability(context: CapabilityContext) -> CapabilityDefinition:
        return RUST_REQUIRED if context.delivery is DeliveryMode.STREAMING else UNSUPPORTED

    component: Final = NativeComponent(name=ComponentName.MESSAGES, capability=capability, exports=())

    assert component.resolve().decision is ExecutionDecision.UNSUPPORTED
    assert (
        component.resolve(CapabilityContext(delivery=DeliveryMode.STREAMING)).decision
        is ExecutionDecision.RUST_REQUIRED
    )


def test_select_returns_python_route_without_loading_native() -> None:
    execution: Final = ComponentExecution(component=ComponentName.OCR, decision=ExecutionDecision.PYTHON)

    assert execution.select(_unloadable_binding(ComponentName.OCR)) == PythonRoute()


@pytest.mark.parametrize("decision", [ExecutionDecision.RUST_WITH_FALLBACK, ExecutionDecision.RUST_REQUIRED])
def test_select_returns_native_route(decision: ExecutionDecision) -> None:
    export: Final = object()
    execution: Final = ComponentExecution(component=ComponentName.OCR, decision=decision)

    assert execution.select(_binding(ComponentName.OCR, SimpleNamespace(route=export))) == NativeRoute(export)


def test_select_falls_back_to_python_when_native_is_missing() -> None:
    execution: Final = ComponentExecution(component=ComponentName.OCR, decision=ExecutionDecision.RUST_WITH_FALLBACK)

    assert execution.select(_binding(ComponentName.OCR, None)) == PythonRoute()


def test_select_reports_unavailable_when_required_native_is_missing() -> None:
    execution: Final = ComponentExecution(
        component=ComponentName.TRANSCRIPTION, decision=ExecutionDecision.RUST_REQUIRED
    )

    failure: Final = execution.select(_binding(ComponentName.TRANSCRIPTION, None))

    assert failure == RouteUnavailable(ComponentName.TRANSCRIPTION)
    with pytest.raises(RustRouteUnavailableError, match="transcription"):
        raise_route_failure(_failure(failure))


def test_select_rejects_a_binding_owned_by_another_component() -> None:
    execution: Final = ComponentExecution(component=ComponentName.OCR, decision=ExecutionDecision.RUST_WITH_FALLBACK)

    failure: Final = execution.select(_unloadable_binding(ComponentName.MESSAGES))

    assert failure == RouteMismatch(component=ComponentName.OCR, binding=ComponentName.MESSAGES)
    with pytest.raises(ValueError, match=r"messages.*ocr"):
        raise_route_failure(_failure(failure))


def test_select_reports_unsupported_without_loading_native() -> None:
    execution: Final = ComponentExecution(component=ComponentName.RESPONSES, decision=ExecutionDecision.UNSUPPORTED)

    failure: Final = execution.select(_unloadable_binding(ComponentName.RESPONSES))

    assert failure == RouteUnsupported(ComponentName.RESPONSES)
    with pytest.raises(RustRouteUnsupportedError, match="responses"):
        raise_route_failure(_failure(failure))


def test_bind_only_allows_declared_exports() -> None:
    native: Final = SimpleNamespace(route=3)
    component: Final = _component(ComponentName.TOKEN_COUNTER)

    binding: Final = component.bind(
        "route", validate=lambda value: value if isinstance(value, int) else None, loader=lambda: native
    )
    assert binding.component is ComponentName.TOKEN_COUNTER
    assert binding.native.load() == 3
    with pytest.raises(ValueError, match=r"'other'.*token_counter"):
        component.bind("other", validate=lambda value: value)
