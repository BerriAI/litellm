"""
Translates from OpenAI's `/v1/chat/completions` to Moonshot AI's `/v1/chat/completions`
"""

import litellm
from typing import Any, Coroutine, List, Literal, Optional, Tuple, Union, overload

from litellm.litellm_core_utils.prompt_templates.common_utils import (
    handle_messages_with_content_list_to_str_conversion,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig


class MoonshotChatConfig(OpenAIGPTConfig):
    @overload
    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: Literal[True]
    ) -> Coroutine[Any, Any, List[AllMessageValues]]:
        ...

    @overload
    def _transform_messages(
        self,
        messages: List[AllMessageValues],
        model: str,
        is_async: Literal[False] = False,
    ) -> List[AllMessageValues]:
        ...

    def _transform_messages(
        self, messages: List[AllMessageValues], model: str, is_async: bool = False
    ) -> Union[List[AllMessageValues], Coroutine[Any, Any, List[AllMessageValues]]]:
        """
        Moonshot AI does not support content in list format.
        """
        messages = handle_messages_with_content_list_to_str_conversion(messages)
        if is_async:
            return super()._transform_messages(
                messages=messages, model=model, is_async=True
            )
        else:
            return super()._transform_messages(
                messages=messages, model=model, is_async=False
            )

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> Tuple[Optional[str], Optional[str]]:
        api_base = (
            api_base
            or get_secret_str("MOONSHOT_API_BASE")
            or "https://api.moonshot.ai/v1"
        )  # type: ignore
        dynamic_api_key = api_key or get_secret_str("MOONSHOT_API_KEY")
        return api_base, dynamic_api_key

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        If api_base is not provided, use the default Moonshot AI /chat/completions endpoint.
        """
        if not api_base:
            api_base = "https://api.moonshot.ai/v1"

        if not api_base.endswith("/chat/completions"):
            api_base = f"{api_base}/chat/completions"

        return api_base

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI params for Moonshot AI models
        
        Moonshot AI limitations:
        - functions parameter is not supported (use tools instead)
        - tool_choice doesn't support "required" value
        """
        base_openai_params = [
            "frequency_penalty",
            "logit_bias",
            "logprobs",
            "top_logprobs",
            "max_tokens",
            "max_completion_tokens",
            "n",
            "presence_penalty",
            "seed",
            "stop",
            "stream",
            "stream_options",
            "temperature",
            "top_p",
            "tools",
            "tool_choice",
            "response_format",
            "user",
            "extra_headers",
            "parallel_tool_calls",
        ]
        # Note: "functions" is not included as it's not supported by Moonshot AI
        return base_openai_params

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Moonshot AI parameters
        
        Handles Moonshot AI specific limitations:
        - tool_choice doesn't support "required" value
        - Temperature <0.3 limitation for n>1
        """
        supported_openai_params = self.get_supported_openai_params(model=model)
        
        for param, value in non_default_params.items():
            if param == "tool_choice":
                # Moonshot AI doesn't support tool_choice="required"
                if value == "required":
                    if litellm.drop_params is True or drop_params is True:
                        continue  # Skip this parameter
                    else:
                        raise litellm.utils.UnsupportedParamsError(
                            message="Moonshot AI doesn't support tool_choice='required'. To drop unsupported openai params from the call, set `litellm.drop_params = True`",
                            status_code=400,
                        )
                else:
                    optional_params[param] = value
            elif param == "max_completion_tokens":
                optional_params["max_tokens"] = value
            elif param in supported_openai_params:
                optional_params[param] = value
        
        # Handle temperature and n limitation
        if "temperature" in optional_params and "n" in optional_params:
            temp = optional_params.get("temperature", 1.0)
            if temp < 0.3 and optional_params.get("n", 1) > 1:
                optional_params["n"] = 1
        
        return optional_params