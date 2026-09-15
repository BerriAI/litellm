from __future__ import annotations

import os
from dataclasses import dataclass
from enum import Enum
from typing import Final, Protocol, TypeAlias, assert_never

DEFAULT_RUST_ENABLED: Final = False
_GLOBAL_ENV_NAME: Final = "LITELLM_RUST"


class ComponentName(str, Enum):
    OCR = "ocr"
    MESSAGES = "messages"
    CHAT_COMPLETIONS = "chat_completions"
    TRANSCRIPTION = "transcription"
    EMBEDDINGS = "embeddings"
    RERANK = "rerank"
    IMAGE_GENERATION = "image_generation"
    IMAGE_EDIT = "image_edit"
    SPEECH = "speech"
    MODERATION = "moderation"
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
        if self.rollout is RolloutPolicy.UNSUPPORTED:
            if self.rust is not RustImplementationState.UNIMPLEMENTED or self.python_available:
                raise ValueError("an unsupported capability must have neither implementation")
            return
        if self.rust is RustImplementationState.UNIMPLEMENTED:
            if self.rollout is not RolloutPolicy.PYTHON_ONLY or not self.python_available:
                raise ValueError("an unimplemented Rust capability must use its Python implementation")
            return
        if self.rollout is RolloutPolicy.PYTHON_ONLY:
            if not self.python_available:
                raise ValueError("a Python-only capability requires a Python implementation")
            return
        if self.rollout is RolloutPolicy.RUST_OPT_IN or self.rollout is RolloutPolicy.RUST_OPT_OUT:
            if not self.python_available:
                raise ValueError("an optional Rust capability requires a Python fallback")
            return
        if self.rollout is RolloutPolicy.RUST_REQUIRED:
            if self.python_available:
                raise ValueError("a required Rust capability must have no Python implementation")
            return
        assert_never(self.rollout)


class CapabilityResolver(Protocol):
    def __call__(self, context: CapabilityContext, /) -> CapabilityDefinition: ...


CapabilitySpec: TypeAlias = CapabilityDefinition | CapabilityResolver


class _RustConfiguration:
    def __init__(self) -> None:
        self.override: bool | None = None


_CONFIGURATION: Final = _RustConfiguration()


def _parse_env_bool(value: str | None) -> bool | None:
    if value is None:
        return None
    match value.strip():
        case "1":
            return True
        case "0":
            return False
        case invalid:
            raise ValueError(f"{_GLOBAL_ENV_NAME} must be '1' or '0', got {invalid!r}")


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
        case RolloutPolicy.RUST_OPT_IN | RolloutPolicy.RUST_OPT_OUT:
            enabled: Final = (
                process_override
                if process_override is not None
                else environment_override
                if environment_override is not None
                else capability.rollout is RolloutPolicy.RUST_OPT_OUT
            )
            return ExecutionDecision.RUST_WITH_FALLBACK if enabled else ExecutionDecision.PYTHON
    assert_never(capability.rollout)


def _definition(spec: CapabilitySpec, context: CapabilityContext) -> CapabilityDefinition:
    if isinstance(spec, CapabilityDefinition):
        return spec
    return spec(context)


def capability_decision(spec: CapabilitySpec, *, context: CapabilityContext) -> ExecutionDecision:
    capability: Final = _definition(spec, context)
    environment_override: Final = (
        _parse_env_bool(os.getenv(_GLOBAL_ENV_NAME))
        if _CONFIGURATION.override is None
        and capability.rollout in (RolloutPolicy.RUST_OPT_IN, RolloutPolicy.RUST_OPT_OUT)
        else None
    )
    return resolve_capability(
        capability,
        process_override=_CONFIGURATION.override,
        environment_override=environment_override,
    )


def rust_enabled() -> bool:
    environment_override: Final = (
        _parse_env_bool(os.getenv(_GLOBAL_ENV_NAME)) if _CONFIGURATION.override is None else None
    )
    return (
        _CONFIGURATION.override
        if _CONFIGURATION.override is not None
        else environment_override
        if environment_override is not None
        else DEFAULT_RUST_ENABLED
    )


def reset_rust_configuration() -> None:
    _CONFIGURATION.override = None


def rust(enabled: bool) -> None:
    """Set the process override for optional Rust capabilities."""
    _CONFIGURATION.override = enabled
