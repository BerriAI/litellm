from typing import Any, Dict, Optional

import httpx
from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)

from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

DEFAULT_ANTHROPIC_API_BASE = "https://api.anthropic.com"
DEFAULT_ANTHROPIC_API_VERSION = "2023-06-01"


class AnthropicMessagesConfig(BaseAnthropicMessagesConfig):
    def get_supported_anthropic_messages_params(self, model: str) -> list:
        return [
            "messages",
            "model",
            "system",
            "max_tokens",
            "stop_sequences",
            "temperature",
            "top_p",
            "top_k",
            "tools",
            "tool_choice",
            "thinking",
            # TODO: Add Anthropic `metadata` support
            # "metadata",
        ]

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        api_base = api_base or DEFAULT_ANTHROPIC_API_BASE
        if not api_base.endswith("/v1/messages"):
            api_base = f"{api_base}/v1/messages"
        return api_base

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        if "x-api-key" not in headers:
            headers["x-api-key"] = api_key
        if "anthropic-version" not in headers:
            headers["anthropic-version"] = DEFAULT_ANTHROPIC_API_VERSION
        if "content-type" not in headers:
            headers["content-type"] = "application/json"
        return headers

    def map_openai_params(self, model: str, optional_params: dict) -> dict:
        supported_params = self.get_supported_anthropic_messages_params(model)
        mapped_params = {}
        for key, value in optional_params.items():
            if key in supported_params:
                mapped_params[key] = value
        return mapped_params

    def transform_request(
        self,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        mapped_params = self.map_openai_params(model, optional_params)
        # Combine the request data
        data = {
            "model": model,
            "messages": messages,
            **mapped_params,
        }

        return data
    
    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: Any,
        logging_obj: LiteLLMLoggingObj,
        api_key: str,
        request_data: dict,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        encoding=None,
        json_mode: bool = False,
    ) -> AnthropicMessagesResponse:
        """
        Transform the raw response to match the expected format
        """
        # For Anthropic Messages, the response is already in the correct format
        # Just parse the JSON response
        response_json = raw_response.json()

        # Return the parsed response
        return response_json

    def transform_streaming_response(
        self,
        streaming_response: httpx.Response,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        request_body: Dict[str, Any]
    ) -> Any:
        """
        Transform a streaming response from Anthropic
        """
        # The streaming response processing is handled in the handler.py file
        # This function is a placeholder for completeness
        return streaming_response
