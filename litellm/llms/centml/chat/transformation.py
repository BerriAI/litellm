"""
Support for CentML's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as CentML is openai-compatible.

Docs: https://docs.centml.ai/reference/chat-completions
"""

from typing import Optional

from litellm import get_model_info, verbose_logger

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class CentmlConfig(OpenAIGPTConfig):

    def get_supported_openai_params(self, model: str) -> list:
        """
        Check which CentML models support specific OpenAI parameters like response_format / tool calling
        """
        supports_function_calling: Optional[bool] = None
        try:
            model_info = get_model_info(model, custom_llm_provider="centml")
            supports_function_calling = model_info.get(
                "supports_function_calling", False)
        except Exception as e:
            verbose_logger.debug(f"Error getting supported openai params: {e}")
            pass

        optional_params = super().get_supported_openai_params(model)
        if supports_function_calling is not True:
            verbose_logger.debug(
                "Only some CentML models support function calling/response_format"
            )
            optional_params.remove("tools")
            optional_params.remove("tool_choice")
            optional_params.remove("function_call")
            optional_params.remove("response_format")
        return optional_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params = super().map_openai_params(
            non_default_params, optional_params, model, drop_params)

        if "response_format" in mapped_openai_params and mapped_openai_params[
                "response_format"] == {
                    "type": "text"
                }:
            mapped_openai_params.pop("response_format")
        return mapped_openai_params
