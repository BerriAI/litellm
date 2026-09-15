import httpx

from litellm.types.utils import FileTypes, TranscriptionResponse


class BedrockAudioTranscriptionRustDispatch:
    def audio_transcriptions(
        self,
        *,
        model: str,
        audio_file: FileTypes,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout: float | httpx.Timeout | None,
    ) -> TranscriptionResponse:
        raise RuntimeError("Bedrock audio transcription must be selected at the public boundary")

    async def async_audio_transcriptions(
        self,
        *,
        model: str,
        audio_file: FileTypes,
        api_key: str | None,
        api_base: str | None,
        custom_llm_provider: str,
        extra_headers: dict[str, object] | None,
        optional_params: dict[str, object],
        timeout: float | httpx.Timeout | None,
    ) -> TranscriptionResponse:
        raise RuntimeError("Bedrock audio transcription must be selected at the public boundary")
