"""
Support for OpenAI's `/v1/chat/completions` endpoint. 

Calls done in OpenAI/openai.py as Novita AI is openai-compatible.

Docs: https://novita.ai/docs/guides/llm-api
"""

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class NovitaConfig(OpenAIGPTConfig):

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params
        )
        return mapped_openai_params
