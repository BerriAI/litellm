from typing import List

from litellm.types.llms.openai import OpenAIAudioTranscriptionOptionalParams

from ...base_llm.audio_transcription.transformation import BaseAudioTranscriptionConfig
from ..common_utils import FireworksAIMixin


class FireworksAIAudioTranscriptionConfig(
    FireworksAIMixin, BaseAudioTranscriptionConfig
):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return ["language", "prompt", "response_format", "timestamp_granularities"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params
