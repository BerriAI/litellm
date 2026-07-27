import base64
from typing import Union

import httpx

from litellm.litellm_core_utils.audio_utils.utils import process_audio_file
from litellm.rust_bridge import transcription as rust_transcription_bridge
from litellm.types.utils import FileTypes, TranscriptionResponse


class BedrockAudioTranscriptionRustDispatch:
    @staticmethod
    def _audio_payload(audio_file: FileTypes) -> dict[str, object]:
        processed_audio = process_audio_file(audio_file)
        formats = {
            "audio/flac": "flac",
            "audio/mpeg": "mp3",
            "audio/mp3": "mp3",
            "audio/ogg": "ogg",
            "audio/wav": "wav",
            "audio/x-wav": "wav",
        }
        audio_format = formats.get(processed_audio.content_type) or (
            processed_audio.filename.rsplit(".", 1)[-1].lower() if "." in processed_audio.filename else ""
        )
        if audio_format not in {"wav", "mp3", "flac", "ogg"}:
            raise ValueError(f"Unsupported Bedrock audio format for file {processed_audio.filename!r}")
        return {
            "data": base64.b64encode(processed_audio.file_content).decode("ascii"),
            "format": audio_format,
            "filename": processed_audio.filename,
        }

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
        timeout: Union[float, httpx.Timeout] | None,
    ) -> TranscriptionResponse:
        rust_response = rust_transcription_bridge.transcription(
            model=model,
            audio=self._audio_payload(audio_file),
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout=timeout,
        )
        if rust_response is None:
            raise RuntimeError("Rust audio transcription bridge is unavailable")
        return TranscriptionResponse(**rust_response)

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
        timeout: Union[float, httpx.Timeout] | None,
    ) -> TranscriptionResponse:
        rust_response = await rust_transcription_bridge.atranscription(
            model=model,
            audio=self._audio_payload(audio_file),
            api_key=api_key,
            api_base=api_base,
            custom_llm_provider=custom_llm_provider,
            extra_headers=extra_headers,
            optional_params=optional_params,
            timeout=timeout,
        )
        if rust_response is None:
            raise RuntimeError("Rust audio transcription bridge is unavailable")
        return TranscriptionResponse(**rust_response)
