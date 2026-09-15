from __future__ import annotations

from collections.abc import Generator
from types import SimpleNamespace
from typing import Final

import pytest

from litellm.rust_bridge import bindings, configuration
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
from litellm.rust_bridge.route import ComponentExecution, NativeComponent

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


def _binding(monkeypatch: pytest.MonkeyPatch, native: object | None) -> bindings.NativeBinding[object]:
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    return bindings.NativeBinding("route", validate=lambda value: value)


def test_resolve_applies_process_override_to_optional_component() -> None:
    component: Final = NativeComponent(name=ComponentName.CHAT_COMPLETIONS, capability=OPT_IN, exports=("route",))

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


def test_select_returns_none_for_python_without_loading_native(monkeypatch: pytest.MonkeyPatch) -> None:
    execution: Final = ComponentExecution(component=ComponentName.OCR, decision=ExecutionDecision.PYTHON)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: pytest.fail("native must not be loaded"))

    assert execution.select(bindings.NativeBinding("route", validate=lambda value: value)) is None


@pytest.mark.parametrize("decision", [ExecutionDecision.RUST_WITH_FALLBACK, ExecutionDecision.RUST_REQUIRED])
def test_select_returns_native_export(monkeypatch: pytest.MonkeyPatch, decision: ExecutionDecision) -> None:
    export: Final = object()
    execution: Final = ComponentExecution(component=ComponentName.OCR, decision=decision)

    assert execution.select(_binding(monkeypatch, SimpleNamespace(route=export))) is export


def test_select_falls_back_to_python_when_native_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    execution: Final = ComponentExecution(component=ComponentName.OCR, decision=ExecutionDecision.RUST_WITH_FALLBACK)

    assert execution.select(_binding(monkeypatch, None)) is None


def test_select_raises_when_required_native_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    execution: Final = ComponentExecution(
        component=ComponentName.TRANSCRIPTION, decision=ExecutionDecision.RUST_REQUIRED
    )

    with pytest.raises(RustRouteUnavailableError, match="transcription"):
        execution.select(_binding(monkeypatch, None))


def test_unsupported_execution_raises_before_loading_native(monkeypatch: pytest.MonkeyPatch) -> None:
    execution: Final = ComponentExecution(component=ComponentName.RESPONSES, decision=ExecutionDecision.UNSUPPORTED)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: pytest.fail("native must not be loaded"))

    with pytest.raises(RustRouteUnsupportedError, match="responses"):
        execution.require_supported()
    with pytest.raises(RustRouteUnsupportedError, match="responses"):
        execution.select(bindings.NativeBinding("route", validate=lambda value: value))


def test_bind_only_allows_declared_exports(monkeypatch: pytest.MonkeyPatch) -> None:
    native: Final = SimpleNamespace(route=3)
    monkeypatch.setattr(bindings, "get_native_bridge", lambda: native)
    component: Final = NativeComponent(name=ComponentName.TOKEN_COUNTER, capability=OPT_IN, exports=("route",))

    assert component.bind("route", validate=lambda value: value if isinstance(value, int) else None).load() == 3
    with pytest.raises(ValueError, match="'other'.*token_counter"):
        component.bind("other", validate=lambda value: value)
