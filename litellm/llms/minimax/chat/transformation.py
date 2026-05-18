"""
MiniMax OpenAI transformation config - extends OpenAI chat config for MiniMax's OpenAI-compatible API
"""

from typing import Any, Dict, List, Optional, Tuple

import litellm
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues, ChatCompletionToolParam


class MinimaxChatConfig(OpenAIGPTConfig):
    """
    MiniMax OpenAI configuration that extends OpenAIGPTConfig.
    MiniMax provides an OpenAI-compatible API at:
    - International: https://api.minimax.io/v1
    - China: https://api.minimaxi.com/v1

    Supported models:
    - MiniMax-M2.1
    - MiniMax-M2.1-lightning
    - MiniMax-M2
    """

    @staticmethod
    def get_api_key(api_key: Optional[str] = None) -> Optional[str]:
        """
        Get MiniMax API key from environment or parameters.
        """
        return api_key or get_secret_str("MINIMAX_API_KEY") or litellm.api_key

    @staticmethod
    def get_api_base(
        api_base: Optional[str] = None,
    ) -> str:
        """
        Get MiniMax API base URL.
        Defaults to international endpoint: https://api.minimax.io/v1
        For China, set to: https://api.minimaxi.com/v1
        """
        return (
            api_base
            or get_secret_str("MINIMAX_API_BASE")
            or "https://api.minimax.io/v1"
        )

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
        Get the complete URL for MiniMax OpenAI API.
        Override to ensure we use MiniMax's endpoint.
        """
        # Get the base URL (either provided or default MiniMax endpoint)
        base_url = self.get_api_base(api_base=api_base)

        # Ensure it ends with /chat/completions
        if base_url.endswith("/chat/completions"):
            return base_url
        elif base_url.endswith("/v1"):
            return f"{base_url}/chat/completions"
        elif base_url.endswith("/"):
            return f"{base_url}v1/chat/completions"
        else:
            return f"{base_url}/v1/chat/completions"

    def remove_cache_control_flag_from_messages_and_tools(
        self,
        model: str,
        messages: List[AllMessageValues],
        tools: Optional[List[ChatCompletionToolParam]] = None,
    ) -> Tuple[List[AllMessageValues], Optional[List[ChatCompletionToolParam]]]:
        """
        Override to preserve cache_control for MiniMax.
        MiniMax supports cache_control - don't strip it.
        """
        # MiniMax supports cache_control, so return messages and tools unchanged
        return messages, tools

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get supported OpenAI parameters for MiniMax.
        Adds reasoning_split and thinking to the list of supported params.
        """
        base_params = super().get_supported_openai_params(model=model)
        additional_params = ["reasoning_split"]

        # Add thinking parameter if model supports reasoning
        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider="minimax"):
                additional_params.append("thinking")
        except Exception:
            pass

        return base_params + additional_params

    @staticmethod
    def _normalize_custom_tools_to_function(tools: Any) -> Any:
        """
        MiniMax's OpenAI-compatible endpoint rejects tools with type="custom".
        Convert OpenAI Responses style custom tools into function tools.
        """
        if not isinstance(tools, list):
            return tools

        normalized_tools: List[Any] = []
        for tool in tools:
            if not isinstance(tool, dict):
                normalized_tools.append(tool)
                continue

            if tool.get("type") != "custom":
                normalized_tools.append(tool)
                continue

            tool_name = tool.get("name")
            if not isinstance(tool_name, str) or not tool_name:
                normalized_tools.append(tool)
                continue

            parameters = tool.get("input_schema")
            if not isinstance(parameters, dict):
                parameters = {"type": "object", "properties": {}}
            elif parameters.get("type") != "object":
                parameters = dict(parameters)
                parameters["type"] = "object"
                parameters.setdefault("properties", {})

            function_tool: Dict[str, Any] = {
                "type": "function",
                "function": {
                    "name": tool_name,
                    "parameters": parameters,
                },
            }
            description = tool.get("description")
            if isinstance(description, str) and description:
                function_tool["function"]["description"] = description
            normalized_tools.append(function_tool)

        return normalized_tools

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        updated_params = super().map_openai_params(
            non_default_params=non_default_params,
            optional_params=optional_params,
            model=model,
            drop_params=drop_params,
        )
        if "tools" in updated_params:
            updated_params["tools"] = self._normalize_custom_tools_to_function(
                updated_params["tools"]
            )
        return updated_params
