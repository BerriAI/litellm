"""
Support for OpenAI's `/v1/chat/completions` endpoint. 

Calls done in OpenAI/openai.py as TogetherAI is openai-compatible.

Docs: https://docs.together.ai/reference/completions-1
"""

from typing import Optional

from litellm.utils import supports_function_calling
from litellm._logging import verbose_logger

from ..openai.chat.gpt_transformation import OpenAIGPTConfig


class TogetherAIConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        Only some together models support response_format / tool calling

        Docs: https://docs.together.ai/docs/json-mode
        """
        # Use supports_function_calling() — which reads _get_model_info_helper
        # directly — instead of get_model_info(). get_model_info() calls
        # get_supported_openai_params() as its first step, which routes back
        # into this method for together_ai models, creating a recursion that
        # only terminates when Python's recursion limit or the "not mapped"
        # exception in _get_model_info_helper is hit (~332 deep calls).
        supports_fc: Optional[bool] = None
        try:
            supports_fc = supports_function_calling(
                model, custom_llm_provider="together_ai"
            )
        except Exception as e:
            verbose_logger.debug(f"Error getting supported openai params: {e}")
            pass

        optional_params = super().get_supported_openai_params(model)
        if supports_fc is not True:
            verbose_logger.debug(
                "Only some together models support function calling/response_format. Docs - https://docs.together.ai/docs/function-calling"
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
            non_default_params, optional_params, model, drop_params
        )

        if "response_format" in mapped_openai_params and mapped_openai_params[
            "response_format"
        ] == {"type": "text"}:
            mapped_openai_params.pop("response_format")
        return mapped_openai_params
