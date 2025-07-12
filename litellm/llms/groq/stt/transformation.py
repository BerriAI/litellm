"""
Translate from OpenAI's `/v1/audio/transcriptions` to Groq's `/v1/audio/transcriptions`
"""

from typing import List
from litellm.types.llms.openai import OpenAIAudioTranscriptionOptionalParams

from ...openai.transcriptions.whisper_transformation import (
    OpenAIWhisperAudioTranscriptionConfig,
)

class GroqAudioTranscriptionConfig(OpenAIWhisperAudioTranscriptionConfig):

    def get_supported_openai_params(
            self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "prompt", "response_format", "temperature", "timestamp_granularities"]
