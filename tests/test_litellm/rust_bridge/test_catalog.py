from __future__ import annotations

from collections.abc import Generator
from typing import Final

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.catalog import COMPONENTS, NATIVE_EXPORTS
from litellm.rust_bridge.configuration import CapabilityContext, ComponentName, DeliveryMode, ExecutionDecision
from litellm.rust_bridge.loader import get_native_bridge


@pytest.fixture(autouse=True)
def _isolated_configuration(  # pyright: ignore[reportUnusedFunction]  # pytest discovers fixtures dynamically
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None]:
    configuration.reset_rust_configuration()
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    yield
    configuration.reset_rust_configuration()


def test_every_component_name_is_cataloged_under_its_own_name() -> None:
    assert set(COMPONENTS) == set(ComponentName)
    assert all(component.name is name for name, component in COMPONENTS.items())


def test_exports_are_unique_across_components() -> None:
    declared: Final = [export for component in COMPONENTS.values() for export in component.exports]

    assert len(declared) == len(NATIVE_EXPORTS) == len(set(declared))


def test_declared_exports_exist_on_the_native_bridge() -> None:
    native: Final = get_native_bridge()
    if native is None:
        pytest.skip("native bridge is not built")

    assert {export for export in NATIVE_EXPORTS if not hasattr(native, export)} == set()


@pytest.mark.parametrize(
    ("name", "context", "expected"),
    (
        (ComponentName.OCR, CapabilityContext(), ExecutionDecision.RUST_WITH_FALLBACK),
        (ComponentName.OCR, CapabilityContext(delivery=DeliveryMode.STREAMING), ExecutionDecision.PYTHON),
        (ComponentName.MESSAGES, CapabilityContext(), ExecutionDecision.PYTHON),
        (ComponentName.CHAT_COMPLETIONS, CapabilityContext(), ExecutionDecision.PYTHON),
        (ComponentName.CHAT_COMPLETIONS, CapabilityContext(delivery=DeliveryMode.STREAMING), ExecutionDecision.PYTHON),
        (ComponentName.TOKEN_COUNTER, CapabilityContext(), ExecutionDecision.PYTHON),
        (ComponentName.TRANSCRIPTION, CapabilityContext(provider="bedrock"), ExecutionDecision.RUST_REQUIRED),
        (ComponentName.TRANSCRIPTION, CapabilityContext(provider="openai"), ExecutionDecision.PYTHON),
        (
            ComponentName.TRANSCRIPTION,
            CapabilityContext(provider="bedrock", delivery=DeliveryMode.STREAMING),
            ExecutionDecision.PYTHON,
        ),
        (ComponentName.RESPONSES, CapabilityContext(), ExecutionDecision.PYTHON),
        (ComponentName.RESPONSES, CapabilityContext(delivery=DeliveryMode.WEBSOCKET), ExecutionDecision.PYTHON),
    ),
)
def test_release_defaults(name: ComponentName, context: CapabilityContext, expected: ExecutionDecision) -> None:
    assert COMPONENTS[name].resolve(context).decision is expected


@pytest.mark.parametrize(
    ("name", "context", "expected"),
    (
        (ComponentName.OCR, CapabilityContext(), ExecutionDecision.RUST_WITH_FALLBACK),
        (ComponentName.MESSAGES, CapabilityContext(), ExecutionDecision.RUST_WITH_FALLBACK),
        (ComponentName.MESSAGES, CapabilityContext(delivery=DeliveryMode.STREAMING), ExecutionDecision.PYTHON),
        (ComponentName.CHAT_COMPLETIONS, CapabilityContext(), ExecutionDecision.RUST_WITH_FALLBACK),
        (
            ComponentName.CHAT_COMPLETIONS,
            CapabilityContext(delivery=DeliveryMode.STREAMING),
            ExecutionDecision.RUST_WITH_FALLBACK,
        ),
        (ComponentName.TOKEN_COUNTER, CapabilityContext(), ExecutionDecision.RUST_WITH_FALLBACK),
        (ComponentName.TRANSCRIPTION, CapabilityContext(provider="openai"), ExecutionDecision.PYTHON),
        (ComponentName.RESPONSES, CapabilityContext(), ExecutionDecision.PYTHON),
        (
            ComponentName.RESPONSES,
            CapabilityContext(delivery=DeliveryMode.WEBSOCKET),
            ExecutionDecision.RUST_WITH_FALLBACK,
        ),
    ),
)
def test_opt_in_decisions(name: ComponentName, context: CapabilityContext, expected: ExecutionDecision) -> None:
    configuration.rust(True)

    assert COMPONENTS[name].resolve(context).decision is expected


def test_bedrock_transcription_ignores_the_optional_rust_switch() -> None:
    context: Final = CapabilityContext(provider="bedrock")

    configuration.rust(False)
    assert COMPONENTS[ComponentName.TRANSCRIPTION].resolve(context).decision is ExecutionDecision.RUST_REQUIRED
