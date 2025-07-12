"""
Translates from OpenAI's `/v1/chat/completions` to Moonshot AI's `/v1/chat/completions`
"""

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

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request to handle Moonshot AI specific limitations:
        - tool_choice doesn't support "required"
        - functions isn't supported at all
        """
        # Remove unsupported parameters
        if "functions" in optional_params:
            optional_params.pop("functions")
        
        # Handle tool_choice limitation - remove "required" if present
        if "tool_choice" in optional_params and optional_params["tool_choice"] == "required":
            optional_params.pop("tool_choice")
            
        # Handle temperature limitation (close to 0 <0.3 can only produce n=1 results)
        if "temperature" in optional_params and "n" in optional_params:
            temp = optional_params.get("temperature", 1.0)
            if temp < 0.3 and optional_params.get("n", 1) > 1:
                optional_params["n"] = 1
        
        return super().transform_request(
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            headers=headers,
        )