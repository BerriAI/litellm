"""
Handles transforming requests for `bedrock/invoke/{qwen3} models`

Inherits from `AmazonInvokeConfig`

Qwen3 + Invoke API Tutorial: https://docs.aws.amazon.com/bedrock/latest/userguide/invoke-imported-model.html
"""

from typing import Any, List, Optional

import httpx

from litellm.llms.base_llm.chat.transformation import BaseConfig
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    AmazonInvokeConfig,
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse


class AmazonQwen3Config(AmazonInvokeConfig, BaseConfig):
    """
    Config for sending `qwen3` requests to `/bedrock/invoke/`
    
    Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/invoke-imported-model.html
    """

    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    stop: Optional[List[str]] = None

    def __init__(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        stop: Optional[List[str]] = None,
    ) -> None:
        locals_ = locals().copy()
        for key, value in locals_.items():
            if key != "self" and value is not None:
                setattr(self.__class__, key, value)
        AmazonInvokeConfig.__init__(self)

    def get_supported_openai_params(self, model: str) -> List[str]:
        return [
            "max_tokens",
            "temperature",
            "top_p",
            "top_k",
            "stop",
            "stream",
        ]

    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        for k, v in non_default_params.items():
            if k == "max_tokens":
                optional_params["max_tokens"] = v
            if k == "temperature":
                optional_params["temperature"] = v
            if k == "top_p":
                optional_params["top_p"] = v
            if k == "top_k":
                optional_params["top_k"] = v
            if k == "stop":
                optional_params["stop"] = v
            if k == "stream":
                optional_params["stream"] = v
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
        Transform OpenAI format to Qwen3 Bedrock invoke format
        """
        # Convert messages to prompt format
        prompt = self._convert_messages_to_prompt(messages)
        
        # Build the request body
        request_body = {
            "prompt": prompt,
        }
        
        # Add optional parameters
        if "max_tokens" in optional_params:
            request_body["max_gen_len"] = optional_params["max_tokens"]
        if "temperature" in optional_params:
            request_body["temperature"] = optional_params["temperature"]
        if "top_p" in optional_params:
            request_body["top_p"] = optional_params["top_p"]
        if "top_k" in optional_params:
            request_body["top_k"] = optional_params["top_k"]
        if "stop" in optional_params:
            request_body["stop"] = optional_params["stop"]
            
        return request_body

    def _convert_messages_to_prompt(self, messages: List[AllMessageValues]) -> str:
        """
        Convert OpenAI messages format to Qwen3 prompt format
        Supports tool calls, multimodal content, and various message types
        """
        prompt_parts = []
        
        for message in messages:
            role = message.get("role", "")
            content = message.get("content", "")
            tool_calls = message.get("tool_calls", [])
            
            if role == "system":
                prompt_parts.append(f"<|im_start|>system\n{content}<|im_end|>")
            elif role == "user":
                # Handle multimodal content
                if isinstance(content, list):
                    text_content = []
                    for item in content:
                        if item.get("type") == "text":
                            text_content.append(item.get("text", ""))
                        elif item.get("type") == "image_url":
                            # For Qwen3, we can include image placeholders
                            text_content.append("<|vision_start|><|image_pad|><|vision_end|>")
                    content = "".join(text_content)
                prompt_parts.append(f"<|im_start|>user\n{content}<|im_end|>")
            elif role == "assistant":
                if tool_calls and isinstance(tool_calls, list):
                    # Handle tool calls
                    for tool_call in tool_calls:
                        function_name = tool_call.get("function", {}).get("name", "")
                        function_args = tool_call.get("function", {}).get("arguments", "")
                        prompt_parts.append(f"<|im_start|>assistant\n<tool_call>\n{{\"name\": \"{function_name}\", \"arguments\": \"{function_args}\"}}\n</tool_call><|im_end|>")
                else:
                    prompt_parts.append(f"<|im_start|>assistant\n{content}<|im_end|>")
            elif role == "tool":
                # Handle tool responses
                prompt_parts.append(f"<|im_start|>tool\n{content}<|im_end|>")
        
        # Add assistant start token for response generation
        prompt_parts.append("<|im_start|>assistant\n")
        
        return "\n".join(prompt_parts)

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        """
        Transform Qwen3 Bedrock response to OpenAI format
        """
        try:
            if hasattr(raw_response, 'json'):
                response_data = raw_response.json()
            else:
                response_data = raw_response
            
            # Extract the generated text - Qwen3 uses "generation" field
            generated_text = response_data.get("generation", "")
            
            # Clean up the response (remove assistant start token if present)
            if generated_text.startswith("<|im_start|>assistant\n"):
                generated_text = generated_text[len("<|im_start|>assistant\n"):]
            if generated_text.endswith("<|im_end|>"):
                generated_text = generated_text[:-len("<|im_end|>")]
            
            # Set the content in the existing model_response structure
            if hasattr(model_response, 'choices') and len(model_response.choices) > 0:
                choice = model_response.choices[0]
                if hasattr(choice, 'message'):
                    choice.message.content = generated_text
                    choice.finish_reason = "stop"
                else:
                    # Handle streaming choices
                    choice.delta.content = generated_text
                    choice.finish_reason = "stop"
            
            # Set usage information if available in response
            if "usage" in response_data:
                usage_data = response_data["usage"]
                if hasattr(model_response, 'usage'):
                    model_response.usage.prompt_tokens = usage_data.get("prompt_tokens", 0)
                    model_response.usage.completion_tokens = usage_data.get("completion_tokens", 0)
                    model_response.usage.total_tokens = usage_data.get("total_tokens", 0)
            
            return model_response
            
        except Exception as e:
            if logging_obj:
                logging_obj.post_call(
                    input=messages,
                    api_key=api_key,
                    original_response=raw_response,
                    additional_args={"error": str(e)},
                )
            raise e
