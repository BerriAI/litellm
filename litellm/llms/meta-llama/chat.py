"""
Support for Llama API's `https://api.llama.com/compat/v1` endpoint.

Calls done in OpenAI/openai.py as Llama API is openai-compatible.

Docs: https://llama.developer.meta.com/docs/features/compatibility/
"""

from typing import Optional

from litellm import get_model_info, verbose_logger

from ..openai.chat.gpt_transformation import OpenAIGPTConfig


class LlamaAPIConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        Llama API has limited support for OpenAI parameters

        Tool calling, Functional Calling, tool choice are not working right now
        response_format: only json_schema is working
        """
        supports_function_calling: Optional[bool] = None
        try:
            model_info = get_model_info(model, custom_llm_provider="meta-llama")
            supports_function_calling = model_info.get(
                "supports_function_calling", False
            )
        except Exception as e:
            verbose_logger.debug(f"Error getting supported openai params: {e}")
            pass

        optional_params = super().get_supported_openai_params(model)
        # Tool calling, Functional Calling, tool choice are not working
        optional_params.remove("tools")
        optional_params.remove("tool_choice")
        optional_params.remove("function_call")
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
