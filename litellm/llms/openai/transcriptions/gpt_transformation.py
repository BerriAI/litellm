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
        return [
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
