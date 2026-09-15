from types import MappingProxyType
from typing import Final

from litellm.rust_bridge.configuration import (
    CapabilityContext,
    CapabilityDefinition,
    CapabilitySpec,
    DeliveryMode,
    RolloutPolicy,
    RouteName,
    RustImplementationState,
    UtilityName,
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
    name: RouteName | UtilityName,
    capability: CapabilitySpec,
    exports: tuple[str, ...],
) -> NativeComponent:
    return NativeComponent(name=name, capability=capability, exports=exports)


COMPONENTS: Final = MappingProxyType(
    {
        RouteName.OCR: _component(
            RouteName.OCR,
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
        RouteName.MESSAGES: _component(
            RouteName.MESSAGES,
            _experimental_completed,
            ("messages", "amessages", "_messages_lifecycle"),
        ),
        RouteName.CHAT_COMPLETIONS: _component(
            RouteName.CHAT_COMPLETIONS,
            _experimental_completed,
            ("chat_completions", "achat_completions", "_chat_completions_lifecycle"),
        ),
        RouteName.TRANSCRIPTION: _component(
            RouteName.TRANSCRIPTION,
            _transcription_capability,
            ("transcription", "atranscription", "_transcription_lifecycle"),
        ),
        RouteName.EMBEDDINGS: _component(RouteName.EMBEDDINGS, _python_completed, ("_embeddings_lifecycle",)),
        RouteName.RERANK: _component(RouteName.RERANK, _python_completed, ("_rerank_lifecycle",)),
        RouteName.IMAGE_GENERATION: _component(
            RouteName.IMAGE_GENERATION, _python_completed, ("_image_generation_lifecycle",)
        ),
        RouteName.IMAGE_EDIT: _component(RouteName.IMAGE_EDIT, _python_completed, ("_image_edit_lifecycle",)),
        RouteName.SPEECH: _component(RouteName.SPEECH, _python_completed, ("_speech_lifecycle",)),
        RouteName.MODERATION: _component(RouteName.MODERATION, _python_completed, ("_moderation_lifecycle",)),
        RouteName.RESPONSES: _component(
            RouteName.RESPONSES,
            _responses_capability,
            ("ResponsesWebSocketConnection", "_responses_lifecycle"),
        ),
        UtilityName.TOKEN_COUNTER: _component(
            UtilityName.TOKEN_COUNTER,
            _unimplemented(),
            (),
        ),
        UtilityName.REQUEST_INPUT_TOKEN_COUNTER: _component(
            UtilityName.REQUEST_INPUT_TOKEN_COUNTER,
            _experimental_completed,
            ("count_input_tokens",),
        ),
    }
)

NATIVE_EXPORTS: Final = frozenset(export for component in COMPONENTS.values() for export in component.exports)
