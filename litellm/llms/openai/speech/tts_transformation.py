from typing import List
from litellm.llms.base_llm.speech.transformation import (
    BaseSpeechConfig,
)
from litellm.types.llms.openai import OpenAISpeechOptionalParams

class OpenAITTSSpeechConfig(BaseSpeechConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAISpeechOptionalParams]:
        """
        Get the supported OpenAI params for the tts models
        """
        return [
            "response_format",
            "speed",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map the OpenAI params to the Speech params
        """
        supported_params = self.get_supported_openai_params(model)
        for k, v in non_default_params.items():
            if k in supported_params:
                optional_params[k] = v
        return optional_params
