from typing import List

from litellm.types.llms.openai import OpenAIAudioTranscriptionOptionalParams

from ...openai.transcriptions.whisper_transformation import (
    OpenAIWhisperAudioTranscriptionConfig,
)
from ..common_utils import FireworksAIMixin


class FireworksAIAudioTranscriptionConfig(
    FireworksAIMixin, OpenAIWhisperAudioTranscriptionConfig
):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "prompt", "response_format", "timestamp_granularities"]
