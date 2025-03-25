from typing import List

from litellm.llms.base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from litellm.types.llms.openai import OpenAIAudioTranscriptionOptionalParams
from litellm.types.utils import FileTypes


class OpenAIWhisperAudioTranscriptionConfig(BaseAudioTranscriptionConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        """
        Get the supported OpenAI params for the `whisper-1` models
        """
        return [
            "language",
            "prompt",
            "response_format",
            "temperature",
            "timestamp_granularities",
        ]

    def transform_audio_transcription_request(
        self,
        model: str,
        audio_file: FileTypes,
        optional_params: dict,
        litellm_params: dict,
    ) -> dict:
        """
        Transform the audio transcription request
        """

        data = {"model": model, "file": audio_file, **optional_params}

        if "response_format" not in data or (
            data["response_format"] == "text" or data["response_format"] == "json"
        ):
            data["response_format"] = (
                "verbose_json"  # ensures 'duration' is received - used for cost calculation
            )

        return data
