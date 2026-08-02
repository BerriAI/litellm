from typing import Final

from litellm.llms.base_llm.audio_transcription.transformation import (
    AudioTranscriptionRequestData,
)
from litellm.types.llms.openai import OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes

from .whisper_transformation import OpenAIWhisperAudioTranscriptionConfig


class OpenAIGPTAudioTranscriptionConfig(OpenAIWhisperAudioTranscriptionConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIAudioTranscriptionOptionalParams]:
        """
        Get the supported OpenAI params for the `gpt-4o-transcribe` models
        """
        return [  # mutable-ok: base transcription interface requires a mutable supported-parameter list
            "language",
            "prompt",
            "response_format",
            "temperature",
            "include",
        ]

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> AudioTranscriptionRequestData:
        """
        Transform the audio transcription request
        """
        data: Final = {"model": model, "file": audio_file, **optional_params}

        return AudioTranscriptionRequestData(
            data=data,
        )


class OpenAIGPTTranscribeAudioTranscriptionConfig(OpenAIGPTAudioTranscriptionConfig):
    def get_supported_openai_params(  # mutable-ok: base transcription interface returns a mutable parameter list
        self, model: str
    ) -> list[OpenAIAudioTranscriptionOptionalParams]:
        return [
            "prompt",
            "response_format",
            "keywords",
            "languages",
            "stream",
        ]

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,  # mutable-ok: base transformation interface supplies a mutable request payload
        litellm_params: dict,  # mutable-ok: base transformation interface supplies mutable provider parameters
    ) -> AudioTranscriptionRequestData:
        data = {  # mutable-ok: OpenAI SDK consumes this multipart request mapping
            "model": model,
            "file": audio_file,
            "response_format": "json",
            **optional_params,
        }
        return AudioTranscriptionRequestData(data=data)
