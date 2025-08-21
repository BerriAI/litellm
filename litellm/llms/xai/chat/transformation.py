from typing import List, Optional, Tuple

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    filter_value_from_dict,
    strip_name_from_messages,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import Choices, ModelResponse

from ...openai.chat.gpt_transformation import OpenAIGPTConfig

XAI_API_BASE = "https://api.x.ai/v1"


class XAIChatConfig(OpenAIGPTConfig):
    @property
    def custom_llm_provider(self) -> Optional[str]:
        return "xai"

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = api_base or get_secret_str("XAI_API_BASE") or XAI_API_BASE  # type: ignore
        dynamic_api_key = api_key or get_secret_str("XAI_API_KEY")
        return api_base, dynamic_api_key

    def get_supported_openai_params(self, model: str) -> list:
        base_openai_params = [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "max_tokens",
            "n",
            "presence_penalty",
            "response_format",
            "seed",
            "stream",
            "stream_options",
            "temperature",
            "tool_choice",
            "tools",
            "top_logprobs",
            "top_p",
            "user",
            "web_search_options",
        ]
        # for some reason, grok-3-mini does not support stop tokens
        if self._supports_stop_reason(model):
            base_openai_params.append("stop")
        try:
            if litellm.supports_reasoning(
                model=model, custom_llm_provider=self.custom_llm_provider
            ):
                base_openai_params.append("reasoning_effort")
        except Exception as e:
            verbose_logger.debug(f"Error checking if model supports reasoning: {e}")

        return base_openai_params
    
    def _supports_stop_reason(self, model: str) -> bool:
        if "grok-3-mini" in model:
            return False
        elif "grok-4" in model:
            return False
        return True

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool = False,
    ) -> dict:
        supported_openai_params = self.get_supported_openai_params(model=model)
        for param, value in non_default_params.items():
            if param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param == "tools" and value is not None:
                tools = []
                for tool in value:
                    tool = filter_value_from_dict(tool, "strict")
                    if tool is not None:
                        tools.append(tool)
                if len(tools) > 0:
                    optional_params["tools"] = tools
            elif param in supported_openai_params:
                if value is not None:
                    optional_params[param] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Handle https://github.com/BerriAI/litellm/issues/9720

        Filter out 'name' from messages
        """
        messages = strip_name_from_messages(messages)
        return super().transform_request(
            model, messages, optional_params, litellm_params, headers
        )

    @staticmethod
    def _fix_choice_finish_reason_for_tool_calls(choice: Choices) -> None:
        """
        Helper to fix finish_reason for tool calls when XAI API returns empty string.
        
        XAI API returns empty string for finish_reason when using tools,
        so we need to set it to "tool_calls" when tool_calls are present.
        """
        if (choice.finish_reason == "" and 
            choice.message.tool_calls and 
            len(choice.message.tool_calls) > 0):
            choice.finish_reason = "tool_calls"

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform the response from the XAI API.
        
        XAI API returns empty string for finish_reason when using tools,
        so we need to fix this after the standard OpenAI transformation.
        """
        
        # First, let the parent class handle the standard transformation
        response = super().transform_response(
            model=model,
            raw_response=raw_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=request_data,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            encoding=encoding,
            api_key=api_key,
            json_mode=json_mode,
        )

        # Fix finish_reason for tool calls across all choices
        if response.choices:
            for choice in response.choices:
                if isinstance(choice, Choices):
                    self._fix_choice_finish_reason_for_tool_calls(choice)

        return response
