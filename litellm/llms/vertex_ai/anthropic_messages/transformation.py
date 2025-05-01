from typing import Any, Dict, Optional

import httpx

from litellm.llms.base_llm.anthropic_messages.transformation import (
    BaseAnthropicMessagesConfig,
)
from litellm.types.llms.anthropic_messages.anthropic_response import (
    AnthropicMessagesResponse,
)
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


class VertexAIAnthropicMessagesConfig(BaseAnthropicMessagesConfig):
    def get_supported_anthropic_messages_params(self, model: str) -> list:
        """
        Return the supported parameters for Vertex AI Anthropic Messages API
        """
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
        ]
    
    def map_openai_params(self, model: str, optional_params: dict) -> dict:
        """
        Maps OpenAI-style parameters to Vertex AI Anthropic parameters
        """
        mapped_params = {}
        
        # Handle max_tokens (required by Anthropic)
        if "max_tokens" not in optional_params and "max_completion_tokens" in optional_params:
            mapped_params["max_tokens"] = optional_params.pop("max_completion_tokens")
            
        # Handle stop sequences
        if "stop" in optional_params:
            mapped_params["stop_sequences"] = optional_params.pop("stop")
            
        # Map response_format.type=json to json_mode=True (for tool usage with json output)
        if "response_format" in optional_params:
            response_format = optional_params.pop("response_format")
            if isinstance(response_format, dict) and response_format.get("type") == "json":
                mapped_params["json_mode"] = True
        
        return mapped_params

    def get_complete_url(self, api_base: Optional[str], model: str) -> str:
        """
        Get the complete URL for the Vertex AI API
        For Vertex AI, this will be handled differently in the main handler
        """
        return api_base or ""

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """
        Validate and return the environment for the API call
        Vertex AI uses GCP authentication, not API keys
        """
        # For Vertex AI, auth is handled through Google service account
        # Make sure proper content-type is set
        if "content-type" not in headers:
            headers["content-type"] = "application/json"
        return headers
    
    def transform_request(
        self,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the request to match Vertex AI Anthropic format
        """
        # Map OpenAI style parameters to Anthropic parameters
        mapped_params = self.map_openai_params(model, optional_params.copy())
        
        # Get all supported parameters from the original request
        anthropic_params = {
            k: v for k, v in optional_params.items() 
            if k in self.get_supported_anthropic_messages_params(model)
        }
        
        # Vertex AI already uses Anthropic's /v1/messages format, so minimal transformation needed
        data = {
            "model": model,
            "messages": messages,
            **anthropic_params,
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
        Transform the raw Vertex AI response to match the expected Anthropic format
        Vertex AI already returns data in Anthropic format, so minimal transformation needed
        """
        # For Vertex AI, the response should already be in the Anthropic format
        response_json = raw_response.json()
        
        # TODO: Add any specific transformations if needed
        
        return response_json
    
    def transform_streaming_response(
        self,
        streaming_response: httpx.Response,
        model: str,
        logging_obj: LiteLLMLoggingObj,
        request_body: Dict[str, Any]
    ) -> Any:
        """
        Transform a streaming response from Vertex AI Anthropic
        """
        # TODO: Implement the streaming response transformation for Vertex AI
        return streaming_response 