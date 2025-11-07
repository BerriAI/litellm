# File: litellm/llms/zai/chat/transformation.py

from typing import Optional, Any, Dict
import os
import json
from litellm.llms.base_llm.chat.transformation import BaseConfig, BaseLLMException
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.utils import ModelResponse, Usage, Choices, Message
from litellm.utils import token_counter


class ZaiError(BaseLLMException):
    """Base error class for Zai API exceptions."""
    pass


class ZaiAuthenticationError(ZaiError):
    """Authentication error for Zai API."""
    pass


class ZaiBadRequestError(ZaiError):
    """Bad request error for Zai API."""
    pass


class ZaiRateLimitError(ZaiError):
    """Rate limit error for Zai API."""
    pass


class ZaiChatConfig(BaseConfig):
    """
    Configuration class for Zai chat completions.
    Inherits from OpenAIGPTConfig to leverage proven tool calling patterns.
    """

    def get_supported_openai_params(self, model: str) -> list:
        """
        Returns list of supported OpenAI parameters for Zai models.
        Full OpenAI compatibility including tool calling.
        """
        return [
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
            "frequency_penalty",
            "stop",
            "reasoning_tokens",  # ZAI-specific reasoning support
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Simple parameter mapping with proper tool support.
        No complex transformations that could interfere with tools.
        """
        supported_params = self.get_supported_openai_params(model)
        for param, value in non_default_params.items():
            if param in supported_params:
                optional_params[param] = value
        return optional_params

    def transform_request(
        self,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform request with proper tool handling.
        Fixed to preserve tools and handle reasoning separately.
        """
        # ZAI model name cleanup
        zai_model = model.replace("zai/", "")
        if zai_model.startswith("zai-org/"):
            zai_model = zai_model.replace("zai-org/", "")
        
        payload = {
            "model": zai_model,
            "messages": messages,
        }

        # Handle reasoning via extra_body (doesn't interfere with tools)
        if litellm_params and litellm_params.get("reasoning_tokens", False):
            extra_body = optional_params.get("extra_body", {})
            extra_body["thinking"] = {"type": "enabled"}
            optional_params["extra_body"] = extra_body
            litellm_params.pop("reasoning_tokens", None)

        # ✅ PRESERVE TOOLS - Pass through directly without popping
        TOOL_PARAMS = ["tools", "tool_choice", "temperature", "max_tokens", 
                       "top_p", "stream", "stop", "parallel_tool_calls"]
        
        for param in TOOL_PARAMS:
            if param in optional_params:
                payload[param] = optional_params[param]

        # Add any remaining optional params
        for param, value in optional_params.items():
            if param not in TOOL_PARAMS and param != "extra_body":
                payload[param] = value

        return payload

    def transform_response(
        self,
        model: str,
        raw_response: Any,
        model_response: ModelResponse,
        logging_obj: Any,
        request_data: dict,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> "ModelResponse":
        """
        Transform response with proper tool call extraction.
        Fixed to prioritize tool calls while preserving reasoning.
        """
        try:
            # Handle different raw_response formats
            if hasattr(raw_response, 'json'):
                completion_response = raw_response.json()
            else:
                completion_response = raw_response
            
            # Extract message info
            choice = completion_response.get("choices", [{}])[0]
            message = choice.get("message", {})
            finish_reason = choice.get("finish_reason", "stop")

            # ✅ PRIORITY 1: Extract tool calls first
            tool_calls = message.get("tool_calls")
            
            if tool_calls:
                # Tool calling response - prioritize tool_calls over content
                message_obj = Message(
                    role=message.get("role", "assistant"),
                    content=None,  # Usually null when tools are used
                    tool_calls=tool_calls,
                )
            else:
                # Regular or reasoning response
                content = message.get("content", "")
                reasoning_content = message.get("reasoning_content", "")
                
                message_obj = Message(
                    role=message.get("role", "assistant"),
                    content=content,
                    reasoning_content=reasoning_content if reasoning_content else None,
                )

            # Handle usage statistics
            usage_data = completion_response.get("usage", {})
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
            total_tokens = usage_data.get("total_tokens", 0)
            
            # Calculate reasoning tokens if present
            reasoning_tokens = 0
            reasoning_content = message.get("reasoning_content", "")
            
            if reasoning_content:
                try:
                    reasoning_tokens = token_counter(
                        text=reasoning_content,
                        model="zai/glm-4.6"
                    )
                except:
                    reasoning_tokens = max(1, len(reasoning_content) // 4)

            # Create usage object with reasoning tokens
            usage_obj = Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens if reasoning_content else None,
            )

            # Return the enhanced response
            new_response = ModelResponse(
                id=completion_response.get("id"),
                model=completion_response.get("model"),
                choices=[
                    Choices(
                        index=0,
                        message=message_obj,
                        finish_reason=finish_reason,
                    )
                ],
                usage=usage_obj,
            )

            return new_response
        except Exception as e:
            raise ValueError(f"Error transforming response: {e}")

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list,
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        """
        Validates the environment by checking for the ZAI_API_KEY.
        Returns the headers with the Authorization token.
        """
        api_key = api_key or os.getenv("ZAI_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing ZAI_API_KEY. Please set the ZAI_API_KEY environment variable or pass the api_key parameter."
            )
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
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
        Returns the complete URL for the Zai chat completions endpoint.
        """
        if api_base:
            return f"{api_base}/chat/completions"
        return "https://open.bigmodel.cn/api/paas/v4/chat/completions"

    def get_error_class(self, error_message: str, status_code: int, headers: dict) -> BaseLLMException:
        """
        Returns a specific exception class based on the error message and status code.
        """
        if status_code == 401:
            return ZaiAuthenticationError(
                status_code=status_code,
                message=error_message,
                headers=headers
            )
        elif status_code == 400:
            return ZaiBadRequestError(
                status_code=status_code,
                message=error_message,
                headers=headers
            )
        elif status_code == 429:
            return ZaiRateLimitError(
                status_code=status_code,
                message=error_message,
                headers=headers
            )
        else:
            return ZaiError(
                status_code=status_code,
                message=error_message,
                headers=headers
            )