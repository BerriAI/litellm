"""
Translate from OpenAI's `/v1/audio/transcriptions` to Groq's `/v1/audio/transcriptions`
"""

from typing import List
from ...base_llm.audio_transcription.transformation import (
    BaseAudioTranscriptionConfig,
)
from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import OpenAIAudioTranscriptionOptionalParams
import litellm


class GroqSTTConfig(OpenAIGPTConfig, BaseAudioTranscriptionConfig):
    def get_supported_openai_params(self, model: str) -> List[OpenAIAudioTranscriptionOptionalParams]:
        return [
            "prompt",
            "response_format",
            "temperature",
            "language",
        ]

    def get_supported_openai_response_formats_stt(self) -> List[str]:
        return ["json", "verbose_json", "text"]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        response_formats = self.get_supported_openai_response_formats_stt()
        for param, value in non_default_params.items():
            if param == "response_format":
                if value in response_formats:
                    optional_params[param] = value
                else:
                    if litellm.drop_params is True or drop_params is True:
                        pass
                    else:
                        raise litellm.utils.UnsupportedParamsError(
                            message="Groq doesn't support response_format={}. To drop unsupported openai params from the call, set `litellm.drop_params = True`".format(
                                value
                            ),
                            status_code=400,
                        )
            else:
                optional_params[param] = value
        return optional_params