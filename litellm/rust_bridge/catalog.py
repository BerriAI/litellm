from types import MappingProxyType
from typing import Final

from litellm.rust_bridge.configuration import (
    CapabilityContext,
    CapabilityDefinition,
    CapabilitySpec,
    ComponentName,
    DeliveryMode,
    RolloutPolicy,
    RustImplementationState,
)
from litellm.rust_bridge.route import NativeComponent


def _unimplemented(*, python_available: bool = True) -> CapabilityDefinition:
    return CapabilityDefinition(
        rust=RustImplementationState.UNIMPLEMENTED,
        python_available=python_available,
        rollout=RolloutPolicy.PYTHON_ONLY if python_available else RolloutPolicy.UNSUPPORTED,
    )


def _experimental() -> CapabilityDefinition:
    return CapabilityDefinition(
        rust=RustImplementationState.EXPERIMENTAL,
        python_available=True,
        rollout=RolloutPolicy.RUST_OPT_IN,
    )


def _ready_default() -> CapabilityDefinition:
    return CapabilityDefinition(
        rust=RustImplementationState.READY,
        python_available=True,
        rollout=RolloutPolicy.RUST_OPT_OUT,
    )


def _completed_only(context: CapabilityContext, completed: CapabilityDefinition) -> CapabilityDefinition:
    return completed if context.delivery is DeliveryMode.COMPLETED else _unimplemented()


def _ocr_capability(context: CapabilityContext) -> CapabilityDefinition:
    return _completed_only(context, _ready_default())


def _experimental_completed(context: CapabilityContext) -> CapabilityDefinition:
    return _completed_only(context, _experimental())


def _python_completed(context: CapabilityContext) -> CapabilityDefinition:
    return _completed_only(context, _unimplemented())


def _responses_capability(context: CapabilityContext) -> CapabilityDefinition:
    return _experimental() if context.delivery is DeliveryMode.WEBSOCKET else _unimplemented()


def _transcription_capability(context: CapabilityContext) -> CapabilityDefinition:
    if context.delivery is not DeliveryMode.COMPLETED:
        return _unimplemented()
    if context.provider == "bedrock":
        return CapabilityDefinition(
            rust=RustImplementationState.EXPERIMENTAL,
            python_available=False,
            rollout=RolloutPolicy.RUST_REQUIRED,
        )

    import litellm
    from litellm.constants import AZURE_OPENAI_AUDIO_PROVIDERS
    from litellm.types.utils import LlmProviders
    from litellm.utils import ProviderConfigManager

    provider: Final = next((provider for provider in LlmProviders if provider.value == context.provider), None)
    python_available: Final = provider is not None and (
        context.provider in AZURE_OPENAI_AUDIO_PROVIDERS
        or context.provider in litellm.openai_compatible_providers
        or ProviderConfigManager.get_provider_audio_transcription_config(model=context.model, provider=provider)
        is not None
    )
    return _unimplemented(python_available=python_available)


def _component(
    name: ComponentName,
    capability: CapabilitySpec,
    exports: tuple[str, ...],
) -> NativeComponent:
    return NativeComponent(name=name, capability=capability, exports=exports)


COMPONENTS: Final = MappingProxyType(
    {
        ComponentName.OCR: _component(
            ComponentName.OCR,
            _ocr_capability,
            (
                "ocr",
                "aocr",
                "_ocr_file_document",
                "_ocr_upload_document",
                "_OCR_MAX_FILE_BYTES",
                "_ocr_mime_type",
                "_ocr_lifecycle",
            ),
        ),
        ComponentName.MESSAGES: _component(
            ComponentName.MESSAGES,
            _experimental_completed,
            ("messages", "amessages", "_messages_lifecycle"),
        ),
        ComponentName.CHAT_COMPLETIONS: _component(
            ComponentName.CHAT_COMPLETIONS,
            _experimental_completed,
            ("chat_completions", "achat_completions", "_chat_completions_lifecycle"),
        ),
        ComponentName.TRANSCRIPTION: _component(
            ComponentName.TRANSCRIPTION,
            _transcription_capability,
            ("transcription", "atranscription", "_transcription_lifecycle"),
        ),
        ComponentName.EMBEDDINGS: _component(ComponentName.EMBEDDINGS, _python_completed, ("_embeddings_lifecycle",)),
        ComponentName.RERANK: _component(ComponentName.RERANK, _python_completed, ("_rerank_lifecycle",)),
        ComponentName.IMAGE_GENERATION: _component(
            ComponentName.IMAGE_GENERATION, _python_completed, ("_image_generation_lifecycle",)
        ),
        ComponentName.IMAGE_EDIT: _component(ComponentName.IMAGE_EDIT, _python_completed, ("_image_edit_lifecycle",)),
        ComponentName.SPEECH: _component(ComponentName.SPEECH, _python_completed, ("_speech_lifecycle",)),
        ComponentName.MODERATION: _component(ComponentName.MODERATION, _python_completed, ("_moderation_lifecycle",)),
        ComponentName.RESPONSES: _component(
            ComponentName.RESPONSES,
            _responses_capability,
            ("ResponsesWebSocketConnection", "_responses_lifecycle"),
        ),
        ComponentName.TOKEN_COUNTER: _component(
            ComponentName.TOKEN_COUNTER,
            CapabilityDefinition(
                rust=RustImplementationState.EXPERIMENTAL,
                python_available=True,
                rollout=RolloutPolicy.PYTHON_ONLY,
            ),
            ("count_input_tokens",),
        ),
    }
)

NATIVE_EXPORTS: Final = frozenset(export for component in COMPONENTS.values() for export in component.exports)
