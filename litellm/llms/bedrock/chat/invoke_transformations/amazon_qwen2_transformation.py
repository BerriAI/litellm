"""
Handles transforming requests for `bedrock/invoke/{qwen2} models`

Inherits from `AmazonQwen3Config` since Qwen2 and Qwen3 architectures are mostly similar.
The main difference is in the response format: Qwen2 uses "text" field while Qwen3 uses "generation" field.

Qwen2 + Invoke API Tutorial: https://docs.aws.amazon.com/bedrock/latest/userguide/invoke-imported-model.html
"""

from typing import Any, List, Optional

import httpx

from litellm.llms.bedrock.chat.invoke_transformations.amazon_qwen3_transformation import (
    AmazonQwen3Config,
)
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse


class AmazonQwen2Config(AmazonQwen3Config):
    """
    Config for sending `qwen2` requests to `/bedrock/invoke/`
    
    Inherits from AmazonQwen3Config since Qwen2 and Qwen3 architectures are mostly similar.
    The main difference is in the response format: Qwen2 uses "text" field while Qwen3 uses "generation" field.
    
    Reference: https://docs.aws.amazon.com/bedrock/latest/userguide/invoke-imported-model.html
    """

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
        Transform Qwen2 Bedrock response to OpenAI format
        
        Qwen2 uses "text" field, but we also support "generation" field for compatibility.
        """
        try:
            if hasattr(raw_response, 'json'):
                response_data = raw_response.json()
            else:
                response_data = raw_response
            
            # Extract the generated text - Qwen2 uses "text" field, but also support "generation" for compatibility
            generated_text = response_data.get("generation", "") or response_data.get("text", "")
            
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

