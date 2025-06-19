"""
Support for CentML's `/v1/chat/completions` endpoint.

Calls done in OpenAI/openai.py as CentML is openai-compatible.

Docs: https://docs.centml.ai/reference/chat-completions
"""

from typing import Optional

from litellm import verbose_logger
from litellm.utils import _get_model_info_helper

from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig


class CentmlConfig(OpenAIGPTConfig):
    def get_supported_openai_params(self, model: str) -> list:
        """
        Check which CentML models support specific OpenAI parameters like response_format / tool calling
        """
        supports_function_calling: Optional[bool] = None
        supports_response_schema: Optional[bool] = None
        try:
            # The model parameter here is the stripped name (e.g., meta-llama/Llama-4-Scout-17B-16E-Instruct)
            # but the model cost map stores it with the provider prefix (e.g., centml/meta-llama/Llama-4-Scout-17B-16E-Instruct)
            # So we need to construct the full model name for lookup
            full_model_name = f"centml/{model}" if not model.startswith("centml/") else model
            # Use _get_model_info_helper to avoid circular dependency with get_supported_openai_params
            model_info = _get_model_info_helper(full_model_name, custom_llm_provider="centml")
            supports_function_calling = model_info.get("supports_function_calling", False)
            supports_response_schema = model_info.get("supports_response_schema", False)
        except Exception as e:
            verbose_logger.debug(f"Error getting supported openai params: {e}")
            pass

        optional_params = super().get_supported_openai_params(model)

        # Remove function calling parameters if not supported
        if supports_function_calling is not True:
            verbose_logger.debug("This CentML model does not support function calling")
            if "tools" in optional_params:
                optional_params.remove("tools")
            if "tool_choice" in optional_params:
                optional_params.remove("tool_choice")
            if "function_call" in optional_params:
                optional_params.remove("function_call")

        # Remove response_format if not supported (separate from function calling)
        if supports_response_schema is not True:
            verbose_logger.debug("This CentML model does not support response_format/JSON schema")
            if "response_format" in optional_params:
                optional_params.remove("response_format")

        return optional_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_openai_params = super().map_openai_params(non_default_params, optional_params, model, drop_params)

        if "response_format" in mapped_openai_params and mapped_openai_params["response_format"] == {"type": "text"}:
            mapped_openai_params.pop("response_format")
        return mapped_openai_params
