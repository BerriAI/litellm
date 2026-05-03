"""OpenRouter-specific validation for audio transcription."""

from litellm.utils import UnsupportedParamsError


def ensure_audio_transcription_supported(
    model: str,
    custom_llm_provider: str,
) -> None:
    """
    OpenRouter does not expose /audio/transcriptions; fail fast with a clear error.
    """
    if custom_llm_provider != "openrouter":
        return
    raise UnsupportedParamsError(
        message=(
            "OpenRouter does not support audio transcription. "
            "Use model='whisper-1' with provider OpenAI directly."
        ),
        model=model,
        llm_provider=custom_llm_provider,
    )
