"""
Support for GPT-4o audio Family

OpenAI Doc: https://platform.openai.com/docs/guides/audio/quickstart?audio-generation-quickstart-example=audio-in&lang=python
"""

import types
from typing import Optional, Union

import litellm
from litellm.types.llms.openai import AllMessageValues, ChatCompletionUserMessage

from .gpt_transformation import OpenAIGPTConfig


class OpenAIGPTAudioConfig(OpenAIGPTConfig):
    """
    Reference: https://platform.openai.com/docs/guides/audio
    """

    @classmethod
    def get_config(cls):
        return {
            k: v
            for k, v in cls.__dict__.items()
            if not k.startswith("__")
            and not isinstance(
                v,
                (
                    types.FunctionType,
                    types.BuiltinFunctionType,
                    classmethod,
                    staticmethod,
                ),
            )
            and v is not None
        }

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for the `gpt-audio` models

        """

        all_openai_params = super().get_supported_openai_params(model=model)
        audio_specific_params = ["audio"]
        return all_openai_params + audio_specific_params

    def is_model_gpt_audio_model(self, model: str) -> bool:
        if model in litellm.open_ai_chat_completion_models and "audio" in model:
            return True
        return False

    def _map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        return super()._map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
