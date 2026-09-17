from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

from litellm.rust_bridge.configuration import (
    CapabilityContext,
    CapabilityDefinition,
    ComponentName,
    DeliveryMode,
    RolloutPolicy,
    RustImplementationState,
)
from litellm.rust_bridge.route import NativeComponent

_PYTHON_ONLY: Final = CapabilityDefinition(
    rust=RustImplementationState.UNIMPLEMENTED,
    python_available=True,
    rollout=RolloutPolicy.PYTHON_ONLY,
)
_EXPERIMENTAL: Final = CapabilityDefinition(
    rust=RustImplementationState.EXPERIMENTAL,
    python_available=True,
    rollout=RolloutPolicy.RUST_OPT_IN,
)
_READY: Final = CapabilityDefinition(
    rust=RustImplementationState.READY,
    python_available=True,
    rollout=RolloutPolicy.RUST_OPT_OUT,
)
_RUST_REQUIRED: Final = CapabilityDefinition(
    rust=RustImplementationState.EXPERIMENTAL,
    python_available=False,
    rollout=RolloutPolicy.RUST_REQUIRED,
)
_UNSUPPORTED: Final = CapabilityDefinition(
    rust=RustImplementationState.UNIMPLEMENTED,
    python_available=False,
    rollout=RolloutPolicy.UNSUPPORTED,
)


def _completed_only(context: CapabilityContext, completed: CapabilityDefinition) -> CapabilityDefinition:
    return completed if context.delivery is DeliveryMode.COMPLETED else _PYTHON_ONLY


def _ocr_capability(context: CapabilityContext) -> CapabilityDefinition:
    return _completed_only(context, _READY)


def _messages_capability(context: CapabilityContext) -> CapabilityDefinition:
    return _completed_only(context, _EXPERIMENTAL)


def _chat_completions_capability(context: CapabilityContext) -> CapabilityDefinition:
    return _completed_only(context, _EXPERIMENTAL)


def _transcription_capability(context: CapabilityContext) -> CapabilityDefinition:
    if context.provider != "bedrock":
        return _PYTHON_ONLY
    return _RUST_REQUIRED if context.delivery is DeliveryMode.COMPLETED else _UNSUPPORTED


def _responses_capability(context: CapabilityContext) -> CapabilityDefinition:
    return _EXPERIMENTAL if context.delivery is DeliveryMode.WEBSOCKET else _PYTHON_ONLY


COMPONENTS: Final[Mapping[ComponentName, NativeComponent]] = MappingProxyType(
    {
        ComponentName.OCR: NativeComponent(
            name=ComponentName.OCR,
            capability=_ocr_capability,
            exports=(
                "ocr",
                "aocr",
                "_ocr_lifecycle",
                "_ocr_file_document",
                "_ocr_upload_document",
                "_OCR_MAX_FILE_BYTES",
                "_ocr_mime_type",
            ),
        ),
        ComponentName.MESSAGES: NativeComponent(
            name=ComponentName.MESSAGES,
            capability=_messages_capability,
            exports=("messages", "amessages"),
        ),
        ComponentName.CHAT_COMPLETIONS: NativeComponent(
            name=ComponentName.CHAT_COMPLETIONS,
            capability=_chat_completions_capability,
            exports=("chat_completions", "achat_completions", "chat_completions_decline"),
        ),
        ComponentName.TRANSCRIPTION: NativeComponent(
            name=ComponentName.TRANSCRIPTION,
            capability=_transcription_capability,
            exports=("transcription", "atranscription"),
        ),
        ComponentName.RESPONSES: NativeComponent(
            name=ComponentName.RESPONSES,
            capability=_responses_capability,
            exports=("ResponsesWebSocketConnection",),
        ),
        ComponentName.TOKEN_COUNTER: NativeComponent(
            name=ComponentName.TOKEN_COUNTER,
            capability=_EXPERIMENTAL,
            exports=("TokenCounter",),
        ),
    }
)

NATIVE_EXPORTS: Final = frozenset(export for component in COMPONENTS.values() for export in component.exports)
