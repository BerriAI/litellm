from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Final, Protocol, TypeAlias

from typing_extensions import assert_never

_TRUE_ENV_VALUES: Final = frozenset({"1", "true", "yes", "on"})
_GLOBAL_ENV_NAME: Final = "LITELLM_RUST"


class ComponentName(str, Enum):
    OCR = "ocr"
    MESSAGES = "messages"
    CHAT_COMPLETIONS = "chat_completions"
    TRANSCRIPTION = "transcription"
    RESPONSES = "responses"
    TOKEN_COUNTER = "token_counter"


class RustImplementationState(str, Enum):
    UNIMPLEMENTED = "unimplemented"
    EXPERIMENTAL = "experimental"
    READY = "ready"


class RolloutPolicy(str, Enum):
    UNSUPPORTED = "unsupported"
    PYTHON_ONLY = "python_only"
    RUST_OPT_IN = "rust_opt_in"
    RUST_OPT_OUT = "rust_opt_out"
    RUST_REQUIRED = "rust_required"


class ExecutionDecision(str, Enum):
    UNSUPPORTED = "unsupported"
    PYTHON = "python"
    RUST_WITH_FALLBACK = "rust_with_fallback"
    RUST_REQUIRED = "rust_required"


class DeliveryMode(str, Enum):
    COMPLETED = "completed"
    STREAMING = "streaming"
    WEBSOCKET = "websocket"


@dataclass(frozen=True, slots=True)
class CapabilityContext:
    provider: str = ""
    model: str = ""
    delivery: DeliveryMode = DeliveryMode.COMPLETED


@dataclass(frozen=True, slots=True)
class CapabilityDefinition:
    rust: RustImplementationState
    python_available: bool
    rollout: RolloutPolicy

    def __post_init__(self) -> None:
        match self.rollout:
            case RolloutPolicy.UNSUPPORTED:
                if self.rust is not RustImplementationState.UNIMPLEMENTED or self.python_available:
                    raise ValueError("an unsupported capability must have neither implementation")
            case RolloutPolicy.PYTHON_ONLY:
                if not self.python_available:
                    raise ValueError("a Python-only capability requires a Python implementation")
            case RolloutPolicy.RUST_OPT_IN | RolloutPolicy.RUST_OPT_OUT:
                if self.rust is RustImplementationState.UNIMPLEMENTED:
                    raise ValueError("an optional Rust capability requires a Rust implementation")
                if not self.python_available:
                    raise ValueError("an optional Rust capability requires a Python fallback")
            case RolloutPolicy.RUST_REQUIRED:
                if self.rust is RustImplementationState.UNIMPLEMENTED:
                    raise ValueError("a required Rust capability requires a Rust implementation")
                if self.python_available:
                    raise ValueError("a required Rust capability must have no Python implementation")
            case _:
                assert_never(self.rollout)


class CapabilityResolver(Protocol):
    def __call__(self, context: CapabilityContext, /) -> CapabilityDefinition: ...


CapabilitySpec: TypeAlias = CapabilityDefinition | CapabilityResolver

_OPTIONAL_RUST: Final = CapabilityDefinition(
    rust=RustImplementationState.EXPERIMENTAL,
    python_available=True,
    rollout=RolloutPolicy.RUST_OPT_IN,
)
_OCR: Final = CapabilityDefinition(
    rust=RustImplementationState.READY,
    python_available=True,
    rollout=RolloutPolicy.RUST_OPT_OUT,
)


class _RustConfiguration:
    def __init__(self) -> None:
        self.override: bool | None = None


_CONFIGURATION: Final = _RustConfiguration()


def _parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    return value.strip().lower() in _TRUE_ENV_VALUES


def resolve_capability(
    capability: CapabilityDefinition,
    *,
    process_override: bool | None,
    environment_override: bool | None,
) -> ExecutionDecision:
    match capability.rollout:
        case RolloutPolicy.UNSUPPORTED:
            return ExecutionDecision.UNSUPPORTED
        case RolloutPolicy.PYTHON_ONLY:
            return ExecutionDecision.PYTHON
        case RolloutPolicy.RUST_REQUIRED:
            return ExecutionDecision.RUST_REQUIRED
        case RolloutPolicy.RUST_OPT_IN:
            enabled: Final = (
                process_override
                if process_override is not None
                else environment_override
                if environment_override is not None
                else False
            )
            return ExecutionDecision.RUST_WITH_FALLBACK if enabled else ExecutionDecision.PYTHON
        case RolloutPolicy.RUST_OPT_OUT:
            if environment_override is False or process_override is False:
                return ExecutionDecision.PYTHON
            return ExecutionDecision.RUST_WITH_FALLBACK
    return assert_never(capability.rollout)


def capability_decision(spec: CapabilitySpec, *, context: CapabilityContext) -> ExecutionDecision:
    capability: Final = spec if isinstance(spec, CapabilityDefinition) else spec(context)
    return resolve_capability(
        capability,
        process_override=_CONFIGURATION.override,
        environment_override=_parse_env_bool(os.getenv(_GLOBAL_ENV_NAME)),
    )


def rust_enabled() -> bool:
    return capability_decision(_OPTIONAL_RUST, context=CapabilityContext()) is ExecutionDecision.RUST_WITH_FALLBACK


def rust_ocr_enabled() -> bool:
    return capability_decision(_OCR, context=CapabilityContext()) is ExecutionDecision.RUST_WITH_FALLBACK


def reset_rust_configuration() -> None:
    _CONFIGURATION.override = None


def rust(enabled: bool) -> None:
    """Set the process override for optional Rust capabilities. Rust-required capabilities ignore it."""
    _CONFIGURATION.override = enabled
