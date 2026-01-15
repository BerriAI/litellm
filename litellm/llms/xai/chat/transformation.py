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
from litellm.types.utils import Choices, ModelResponse, Usage, PromptTokensDetailsWrapper

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
        #########################################################
        # stop tokens check
        #########################################################
        if self._supports_stop_reason(model):
            base_openai_params.append("stop")
        

        #########################################################
        # frequency penalty check
        #########################################################
        if self._supports_frequency_penalty(model):
            base_openai_params.append("frequency_penalty")
        
        #########################################################
        # reasoning check
        #########################################################
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
        elif "grok-code-fast" in model:
            return False
        return True
    
    def _supports_frequency_penalty(self, model: str) -> bool:
        """
        From manual testing grok-4 does not support `frequency_penalty`

        When sent the model fails from xAI API
        """
        if "grok-4" in model:
            return False
        if "grok-code-fast" in model:
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
        
        Also handles X.AI web search usage tracking by extracting num_sources_used.
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

        # Handle X.AI web search usage tracking
        try:
            raw_response_json = raw_response.json()
            self._enhance_usage_with_xai_web_search_fields(response, raw_response_json)
        except Exception as e:
            verbose_logger.debug(f"Error extracting X.AI web search usage: {e}")
        return response

    def _enhance_usage_with_xai_web_search_fields(
        self, model_response: ModelResponse, raw_response_json: dict
    ) -> None:
        """
        Extract num_sources_used from X.AI response and map it to web_search_requests.
        """
        if not hasattr(model_response, "usage") or model_response.usage is None:
            return

        usage: Usage = model_response.usage
        num_sources_used = None
        response_usage = raw_response_json.get("usage", {})
        if isinstance(response_usage, dict) and "num_sources_used" in response_usage:
            num_sources_used = response_usage.get("num_sources_used")
        
        # Map num_sources_used to web_search_requests for cost detection
        if num_sources_used is not None and num_sources_used > 0:
            if usage.prompt_tokens_details is None:
                usage.prompt_tokens_details = PromptTokensDetailsWrapper()
            
            usage.prompt_tokens_details.web_search_requests = int(num_sources_used)
            setattr(usage, "num_sources_used", int(num_sources_used))
            verbose_logger.debug(f"X.AI web search sources used: {num_sources_used}")
