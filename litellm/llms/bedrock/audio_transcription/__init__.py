import base64
from typing import Final, NoReturn

import httpx

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.rust_bridge import transcription as rust_transcription_bridge
from litellm.rust_bridge.dispatch import anative_first, native_first, provider_errors
from litellm.rust_bridge.runtime import DispatchResult, adapt_result
from litellm.types.utils import FileTypes, TranscriptionResponse


def _unavailable() -> NoReturn:
    raise RuntimeError("Rust audio transcription bridge is unavailable")


async def _aunavailable() -> NoReturn:
    _unavailable()


class BedrockAudioTranscriptionRustDispatch:
    @staticmethod
    def _audio_payload(audio_file: FileTypes) -> dict[str, object]:
        processed_audio: Final = process_audio_file(audio_file)
        formats: Final = {
            "audio/flac": "flac",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/ogg": "ogg",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
        }
        audio_format: Final = formats.get(processed_audio.content_type) or (
            processed_audio.filename.rsplit(".", 1)[-1].lower() if "." in processed_audio.filename else ""
        )
        if audio_format not in {"wav", "mp3", "flac", "ogg"}:
            raise ValueError(f"Unsupported Bedrock audio format for file {processed_audio.filename!r}")
        return {
            "data": base64.b64encode(processed_audio.file_content).decode("ascii"),
            "format": audio_format,
            "filename": processed_audio.filename,
        }

    def _attempt_audio_transcriptions(
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
    ) -> DispatchResult[TranscriptionResponse]:
        result: Final = rust_transcription_bridge.transcription(
            model=model,
            audio=self._audio_payload(audio_file),
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout=timeout,
        )
        return adapt_result(result, lambda response: TranscriptionResponse(**response))

    @native_first(
        native=_attempt_audio_transcriptions,
        route="audio transcription",
        errors=lambda self, model, audio_file, api_key, api_base, custom_llm_provider, extra_headers, optional_params, timeout: (
            provider_errors(custom_llm_provider, model)
        ),
    )
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
        _unavailable()

    async def _attempt_async_audio_transcriptions(
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
    ) -> DispatchResult[TranscriptionResponse]:
        result: Final = await rust_transcription_bridge.atranscription(
            model=model,
            audio=self._audio_payload(audio_file),
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout=timeout,
        )
        return adapt_result(result, lambda response: TranscriptionResponse(**response))

    @anative_first(
        native=_attempt_async_audio_transcriptions,
        route="audio transcription",
        errors=lambda self, model, audio_file, api_key, api_base, custom_llm_provider, extra_headers, optional_params, timeout: (
            provider_errors(custom_llm_provider, model)
        ),
    )
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
        await _aunavailable()
