"""
AskSage chat completion transformation and configuration
"""

import json
from typing import Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse


class AskSageException(BaseLLMException):
    """Custom exception for AskSage API errors"""
    def __init__(self, status_code, message, headers=None):
        super().__init__(status_code=status_code, message=message, headers=headers)


class AskSageConfig(BaseConfig):
    """
    Configuration class for AskSage provider
    """

    def _get_openai_compatible_provider_info(
        self, api_base: Optional[str], api_key: Optional[str]
    ) -> tuple[Optional[str], Optional[str]]:
        """Get API base and key for AskSage"""
        
        api_base = (
            api_base
            or get_secret_str("ASKSAGE_API_BASE") 
            or "https://api.asksage.ai/server"
        )
        dynamic_api_key = api_key or get_secret_str("ASKSAGE_API_KEY")
        
        return api_base, dynamic_api_key

    @property
    def custom_llm_provider(self) -> str:
        return "asksage"

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the supported OpenAI parameters for AskSage
        Based on AskSage /query endpoint parameters
        """
        return [
            "messages",  # Will be transformed to "message" 
            "model",
            "temperature",
            # AskSage specific parameters that can be passed through litellm_params
            "persona",
            "system_prompt", 
            "dataset",
            "limit_references",
            "live",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to AskSage API format
        """
        # Map temperature directly
        if "temperature" in non_default_params:
            optional_params["temperature"] = non_default_params["temperature"]

        # Map AskSage-specific parameters if provided
        asksage_specific_params = [
            "persona",
            "system_prompt",
            "dataset", 
            "limit_references",
            "live",
        ]
        
        for param in asksage_specific_params:
            if param in non_default_params:
                optional_params[param] = non_default_params[param]

        # Drop unsupported OpenAI parameters
        if drop_params:
            unsupported_params = [
                "max_tokens",  # AskSage doesn't have max_tokens
                "top_p",
                "frequency_penalty", 
                "presence_penalty",
                "stop",
                "stream",  # Need to handle streaming differently
                "tools",
                "tool_choice",
                "response_format",
                "seed",
                "logit_bias",
                "user",
            ]
            for param in unsupported_params:
                non_default_params.pop(param, None)

        return optional_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validate the environment for AskSage API requests
        """
        if api_key is None:
            raise ValueError("AskSage API key is required")
        
        # Set default headers - AskSage uses x-access-tokens instead of Authorization Bearer
        headers.update({
            "x-access-tokens": api_key,
            "Content-Type": "application/json",
        })

        return headers

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
        Get the complete URL for AskSage API requests
        """
        if api_base is None:
            api_base = "https://api.asksage.ai/server"
        
        return f"{api_base.rstrip('/')}/query"

    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request to AskSage API format
        Expected AskSage /query format:
        {
          "message": "string",
          "persona": "string", 
          "system_prompt": "string",
          "dataset": "string",
          "limit_references": 0,
          "temperature": 0,
          "live": 0,
          "model": "string"
        }
        """
        # Translate developer role to system role for AskSage
        messages = self.translate_developer_role_to_system_role(messages)
        
        # Convert OpenAI messages format to a single message string
        # Take the last user message as the main message
        message_content = ""
        system_prompt = ""
        
        for msg in messages:
            if msg["role"] == "user":
                if isinstance(msg["content"], str):
                    message_content = msg["content"]
                elif isinstance(msg["content"], list):
                    # Handle multi-part content (text + images)
                    text_parts = [part["text"] for part in msg["content"] if part.get("type") == "text"]
                    message_content = " ".join(text_parts)
            elif msg["role"] == "system":
                if isinstance(msg["content"], str):
                    system_prompt = msg["content"]

        # Build the request data for AskSage /query endpoint
        data = {
            "message": message_content,
            "model": model,
        }

        # Add system prompt if available
        if system_prompt:
            data["system_prompt"] = system_prompt
        elif optional_params.get("system_prompt"):
            data["system_prompt"] = optional_params["system_prompt"]

        # Add optional AskSage-specific parameters with defaults
        data["temperature"] = optional_params.get("temperature", 0)
        data["limit_references"] = optional_params.get("limit_references", 0)
        data["live"] = optional_params.get("live", 0)
        
        # Add optional parameters if provided
        if optional_params.get("persona"):
            data["persona"] = optional_params["persona"]
        if optional_params.get("dataset"):
            data["dataset"] = optional_params["dataset"]

        return data

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform AskSage API response to LiteLLM format
        """
        try:
            response_json = raw_response.json()
        except json.JSONDecodeError:
            raise AskSageException(
                status_code=raw_response.status_code,
                message=f"Invalid JSON response: {raw_response.text}",
                headers=raw_response.headers,
            )

        response_content = ""
        if isinstance(response_json, dict):
            # Common patterns for response content
            response_content = (
                response_json.get("message") or
                str(response_json)
            )
        else:
            response_content = str(response_json)

        # Build OpenAI-compatible response
        from litellm.types.utils import Choices, Message
        import time

        model_response.id = f"asksage-{int(time.time())}"
        model_response.created = int(time.time())
        model_response.model = model
        model_response.object = "chat.completion"
        
        model_response.choices = [
            Choices(
                index=0,
                message=Message(
                    role="assistant",
                    content=response_content,
                ),
                finish_reason="stop",
            )
        ]

        model_response._hidden_params["original_response"] = response_json

        return model_response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        """
        Return the appropriate error class for AskSage errors
        """
        return AskSageException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
