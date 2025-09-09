"""
Support for Llama API's `https://api.llama.com/compat/v1` endpoint.

Calls done in OpenAI/openai.py as Llama API is openai-compatible.

Docs: https://llama.developer.meta.com/docs/features/compatibility/
"""

import warnings

# Suppress Pydantic serialization warnings for Meta Llama responses
warnings.filterwarnings("ignore", message="Pydantic serializer warnings")

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class LlamaAPIConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        Llama API has limited support for OpenAI parameters

        function_call, tools, and tool_choice are working
        response_format: only json_schema is working
        """
        # Function calling and tool choice are now supported on Llama API
        optional_params = super().get_supported_openai_params(model)
        return optional_params

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

        # Only json_schema is working for response_format
        if (
            "response_format" in mapped_openai_params
            and mapped_openai_params["response_format"].get("type") != "json_schema"
        ):
            mapped_openai_params.pop("response_format")
        return mapped_openai_params
