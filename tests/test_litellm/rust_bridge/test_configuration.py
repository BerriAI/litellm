from __future__ import annotations

import os
import subprocess
import sys
from collections.abc import Generator
from concurrent.futures import ThreadPoolExecutor
from typing import Final

import pytest

from litellm.rust_bridge import configuration
from litellm.rust_bridge.configuration import (
    CapabilityContext,
    CapabilityDefinition,
    DeliveryMode,
    ExecutionDecision,
    RolloutPolicy,
    RustImplementationState,
)

OPT_IN: Final = CapabilityDefinition(
    rust=RustImplementationState.EXPERIMENTAL, python_available=True, rollout=RolloutPolicy.RUST_OPT_IN
)
OPT_OUT: Final = CapabilityDefinition(
    rust=RustImplementationState.READY, python_available=True, rollout=RolloutPolicy.RUST_OPT_OUT
)
FIXED: Final = (
    (
        CapabilityDefinition(
            rust=RustImplementationState.UNIMPLEMENTED, python_available=False, rollout=RolloutPolicy.UNSUPPORTED
        ),
        ExecutionDecision.UNSUPPORTED,
    ),
    (
        CapabilityDefinition(
            rust=RustImplementationState.UNIMPLEMENTED, python_available=True, rollout=RolloutPolicy.PYTHON_ONLY
        ),
        ExecutionDecision.PYTHON,
    ),
    (
        CapabilityDefinition(
            rust=RustImplementationState.EXPERIMENTAL, python_available=True, rollout=RolloutPolicy.PYTHON_ONLY
        ),
        ExecutionDecision.PYTHON,
    ),
    (
        CapabilityDefinition(
            rust=RustImplementationState.EXPERIMENTAL, python_available=False, rollout=RolloutPolicy.RUST_REQUIRED
        ),
        ExecutionDecision.RUST_REQUIRED,
    ),
)


@pytest.fixture(autouse=True)
def _isolated_configuration(  # pyright: ignore[reportUnusedFunction]  # pytest discovers fixtures dynamically
    monkeypatch: pytest.MonkeyPatch,
) -> Generator[None]:
    configuration.reset_rust_configuration()
    monkeypatch.delenv("LITELLM_RUST", raising=False)
    yield
    configuration.reset_rust_configuration()


@pytest.mark.parametrize(
    ("process", "environment", "capability", "expected"),
    (
        (False, True, OPT_OUT, ExecutionDecision.PYTHON),
        (True, False, OPT_IN, ExecutionDecision.RUST_WITH_FALLBACK),
        (None, False, OPT_OUT, ExecutionDecision.PYTHON),
        (None, True, OPT_IN, ExecutionDecision.RUST_WITH_FALLBACK),
        (None, None, OPT_IN, ExecutionDecision.PYTHON),
        (None, None, OPT_OUT, ExecutionDecision.RUST_WITH_FALLBACK),
    ),
)
def test_optional_rust_resolution_precedence(
    process: bool | None,
    environment: bool | None,
    capability: CapabilityDefinition,
    expected: ExecutionDecision,
) -> None:
    assert (
        configuration.resolve_capability(
            capability,
            process_override=process,
            environment_override=environment,
        )
        is expected
    )


@pytest.mark.parametrize(("capability", "expected"), FIXED)
@pytest.mark.parametrize("process", (None, False, True))
@pytest.mark.parametrize("environment", (None, False, True))
def test_fixed_policies_ignore_overrides(
    capability: CapabilityDefinition,
    expected: ExecutionDecision,
    process: bool | None,
    environment: bool | None,
) -> None:
    assert (
        configuration.resolve_capability(capability, process_override=process, environment_override=environment)
        is expected
    )


@pytest.mark.parametrize(
    ("rust", "python_available", "rollout"),
    (
        (RustImplementationState.READY, True, RolloutPolicy.RUST_REQUIRED),
        (RustImplementationState.EXPERIMENTAL, False, RolloutPolicy.RUST_OPT_IN),
        (RustImplementationState.READY, False, RolloutPolicy.RUST_OPT_OUT),
        (RustImplementationState.UNIMPLEMENTED, True, RolloutPolicy.RUST_OPT_IN),
        (RustImplementationState.UNIMPLEMENTED, True, RolloutPolicy.RUST_OPT_OUT),
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


def test_capability_resolver_receives_context(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    seen: Final[list[CapabilityContext]] = []

    def resolver(context: CapabilityContext) -> CapabilityDefinition:
        seen.append(context)
        if context.provider == "bedrock" and context.delivery is DeliveryMode.COMPLETED:
            return CapabilityDefinition(
                rust=RustImplementationState.EXPERIMENTAL,
                python_available=False,
                rollout=RolloutPolicy.RUST_REQUIRED,
            )
        return OPT_IN

    bedrock: Final = CapabilityContext(provider="bedrock", model="whisper", delivery=DeliveryMode.COMPLETED)
    streaming: Final = CapabilityContext(provider="bedrock", model="whisper", delivery=DeliveryMode.STREAMING)
    assert configuration.capability_decision(resolver, context=bedrock) is ExecutionDecision.RUST_REQUIRED
    assert configuration.capability_decision(resolver, context=streaming) is ExecutionDecision.PYTHON
    assert seen == [bedrock, streaming]


def test_capability_decision_uses_process_and_environment_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    context: Final = CapabilityContext()
    assert configuration.capability_decision(OPT_IN, context=context) is ExecutionDecision.PYTHON
    monkeypatch.setenv("LITELLM_RUST", "1")
    assert configuration.capability_decision(OPT_IN, context=context) is ExecutionDecision.RUST_WITH_FALLBACK
    configuration.rust(False)
    assert configuration.capability_decision(OPT_IN, context=context) is ExecutionDecision.PYTHON


def test_release_default_remains_disabled() -> None:
    assert configuration.rust_enabled() is False
    assert configuration.rust_ocr_enabled() is True


@pytest.mark.parametrize("process", [None, False, True])
@pytest.mark.parametrize("environment", [None, "0", "1", "off"])
def test_ocr_configuration(monkeypatch: pytest.MonkeyPatch, process: bool | None, environment: str | None) -> None:
    if environment is not None:
        monkeypatch.setenv("LITELLM_RUST", environment)
    if process is not None:
        configuration.rust(process)

    assert configuration.rust_ocr_enabled() is (environment not in {"0", "off"} and process is not False)


def test_process_override_wins_over_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "0")
    configuration.rust(True)

    assert configuration.rust_enabled() is True


def test_global_environment_accepts_explicit_false(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "off")

    assert configuration.rust_enabled() is False


@pytest.mark.parametrize("value", ("", " ", "sometimes", "2"))
def test_invalid_environment_value_disables_rust(monkeypatch: pytest.MonkeyPatch, value: str) -> None:
    monkeypatch.setenv("LITELLM_RUST", value)

    assert configuration.rust_enabled() is False


def test_process_override_and_reset_apply_to_existing_threads(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "1")

    with ThreadPoolExecutor(max_workers=1) as executor:
        assert executor.submit(configuration.rust_enabled).result() is True
        configuration.rust(False)
        assert executor.submit(configuration.rust_enabled).result() is False
        configuration.reset_rust_configuration()
        assert executor.submit(configuration.rust_enabled).result() is True


def test_explicit_override_precedes_invalid_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LITELLM_RUST", "sometimes")

    configuration.rust(True)
    assert configuration.rust_enabled() is True


@pytest.mark.parametrize(("value", "expected"), (("1", "True"), ("0", "False")))
def test_environment_controls_startup(value: str, expected: str) -> None:
    environment: Final = {**os.environ, "LITELLM_RUST": value}
    result: Final = subprocess.run(
        (
            sys.executable,
            "-c",
            "from litellm.rust_bridge.configuration import rust_enabled; print(rust_enabled())",
        ),
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert result.stdout.strip() == expected
