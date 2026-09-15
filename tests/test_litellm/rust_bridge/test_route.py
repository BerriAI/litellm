from __future__ import annotations

from collections.abc import Iterator
from types import ModuleType
from typing import Final

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.bindings import BINDING_UNSET
from litellm.rust_bridge.catalog import COMPONENTS
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
from litellm.rust_bridge.route import NativeComponent


@pytest.fixture(autouse=True)
def reset_configuration() -> Iterator[None]:
    configuration.reset_rust_configuration()
    yield
    configuration.reset_rust_configuration()


def _string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _unexpected_load() -> ModuleType:
    raise AssertionError("Python selection loaded the native extension")


def test_python_decision_does_not_load_native(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    component: Final = COMPONENTS[ComponentName.MESSAGES]
    binding: Final = component.bind("messages", validate=_string, module_loader=_unexpected_load)
    assert component.resolve().select(binding) is None


def test_delivery_mode_uses_one_component_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")
    component: Final = COMPONENTS[ComponentName.MESSAGES]
    completed: Final = component.resolve(CapabilityContext(delivery=DeliveryMode.COMPLETED))
    streaming: Final = component.resolve(CapabilityContext(delivery=DeliveryMode.STREAMING))
    assert completed.decision is ExecutionDecision.RUST_WITH_FALLBACK
    assert streaming.decision is ExecutionDecision.PYTHON


def test_binding_discovery_validation_and_override() -> None:
    module: Final = ModuleType("fake_native")
    setattr(module, "messages", "native")
    component: Final = COMPONENTS[ComponentName.MESSAGES]
    binding: Final = component.bind("messages", validate=_string, module_loader=lambda: module)
    assert binding.load() == "native"
    binding.configure("override")
    binding.configure(BINDING_UNSET)
    assert binding.load() == "override"
    binding.override(None)
    assert binding.load() is None
    binding.configure(None)
    assert binding.load() == "native"
    setattr(module, "messages", 42)
    assert binding.load() is None


def test_component_rejects_undeclared_binding() -> None:
    with pytest.raises(ValueError, match="not declared"):
        COMPONENTS[ComponentName.MESSAGES].bind("typo", validate=_string)


class ModelCapability:
    def __init__(self) -> None:
        self.calls: int = 0

    def __call__(self, context: CapabilityContext) -> CapabilityDefinition:
        self.calls += 1
        if context.provider == "provider" and context.model == "new-model":
            return CapabilityDefinition(
                rust=RustImplementationState.EXPERIMENTAL,
                python_available=False,
                rollout=RolloutPolicy.RUST_REQUIRED,
            )
        return CapabilityDefinition(
            rust=RustImplementationState.READY,
            python_available=True,
            rollout=RolloutPolicy.RUST_OPT_IN,
        )


def test_dynamic_capability_resolves_once_before_binding_selection() -> None:
    resolver: Final = ModelCapability()
    component: Final = NativeComponent(
        name=ComponentName.TRANSCRIPTION,
        capability=resolver,
        exports=("transcription",),
    )
    binding: Final = component.bind("transcription", validate=_string, module_loader=_unexpected_load)
    binding.override("native")
    execution: Final = component.resolve(CapabilityContext(provider="provider", model="new-model"))
    assert execution.select(binding) == "native"
    assert resolver.calls == 1


@pytest.mark.parametrize("process_override", (None, False, True))
@pytest.mark.parametrize("environment_override", ("0", "1", "invalid"))
def test_bedrock_transcription_requires_rust_regardless_of_overrides(
    monkeypatch: pytest.MonkeyPatch, process_override: bool | None, environment_override: str
) -> None:
    monkeypatch.setenv("LITELLM_RUST", environment_override)
    if process_override is not None:
        configuration.rust(process_override)
    component: Final = COMPONENTS[ComponentName.TRANSCRIPTION]
    binding: Final = component.bind("transcription", validate=_string, module_loader=lambda: None)
    execution: Final = component.resolve(CapabilityContext(provider="bedrock", model="model"))
    assert execution.decision is ExecutionDecision.RUST_REQUIRED
    with pytest.raises(RustRouteUnavailableError, match="transcription bridge is unavailable"):
        execution.select(binding)


@pytest.mark.parametrize("provider", ("openai", "azure", "azure_ai", "groq", "mistral", "nvidia_riva", "soniox"))
def test_python_transcription_providers_skip_native_discovery(provider: str) -> None:
    configuration.rust(True)
    component: Final = COMPONENTS[ComponentName.TRANSCRIPTION]
    binding: Final = component.bind("transcription", validate=_string, module_loader=_unexpected_load)
    execution: Final = component.resolve(CapabilityContext(provider=provider, model="model"))
    assert execution.decision is ExecutionDecision.PYTHON
    assert execution.select(binding) is None


@pytest.mark.parametrize("provider", ("unknown-provider", "anthropic"))
def test_unsupported_transcription_never_selects_an_implementation(provider: str) -> None:
    component: Final = COMPONENTS[ComponentName.TRANSCRIPTION]
    binding: Final = component.bind("transcription", validate=_string, module_loader=_unexpected_load)
    execution: Final = component.resolve(CapabilityContext(provider=provider, model="model"))
    assert execution.decision is ExecutionDecision.UNSUPPORTED
    with pytest.raises(RustRouteUnsupportedError, match="No Python or Rust implementation for transcription"):
        execution.select(binding)


@pytest.mark.parametrize(
    ("rust", "python_available", "rollout"),
    (
        (RustImplementationState.READY, True, RolloutPolicy.RUST_REQUIRED),
        (RustImplementationState.EXPERIMENTAL, False, RolloutPolicy.RUST_OPT_IN),
        (RustImplementationState.READY, False, RolloutPolicy.RUST_OPT_OUT),
        (RustImplementationState.UNIMPLEMENTED, False, RolloutPolicy.PYTHON_ONLY),
        (RustImplementationState.UNIMPLEMENTED, False, RolloutPolicy.RUST_REQUIRED),
        (RustImplementationState.UNIMPLEMENTED, True, RolloutPolicy.UNSUPPORTED),
        (RustImplementationState.READY, False, RolloutPolicy.UNSUPPORTED),
        (RustImplementationState.EXPERIMENTAL, False, RolloutPolicy.PYTHON_ONLY),
    ),
)
def test_invalid_capability_definitions_rejected(
    rust: RustImplementationState, python_available: bool, rollout: RolloutPolicy
) -> None:
    with pytest.raises(ValueError, match="capability"):
        CapabilityDefinition(rust=rust, python_available=python_available, rollout=rollout)
