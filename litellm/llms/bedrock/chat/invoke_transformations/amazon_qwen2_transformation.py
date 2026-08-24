"""
Handles transforming requests for `bedrock/invoke/{qwen2} models`

Inherits from `AmazonQwen3Config` since Qwen2 and Qwen3 architectures are mostly similar.
The main difference is in the response format: Qwen2 uses "text" field while Qwen3 uses "generation" field.

Qwen2 + Invoke API Tutorial: https://docs.aws.amazon.com/bedrock/latest/userguide/invoke-imported-model.html
"""

from typing import Any, Final

import httpx

from litellm.llms.bedrock.chat.invoke_transformations.amazon_qwen3_transformation import (
    AmazonQwen3Config,
)
from litellm.llms.bedrock.chat.invoke_transformations.base_invoke_transformation import (
    LiteLLMLoggingObj,
)
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ModelResponse, Usage


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
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ModelResponse:
        """
        Transform Qwen2 Bedrock response to OpenAI format

        Qwen2 uses "text" field, but we also support "generation" field for compatibility.
        """
        try:
            response_data: Final = raw_response.json()

            # Extract the generated text - Qwen2 uses "text" field, but also support "generation" for compatibility
            generated_text = response_data.get("generation", "") or response_data.get("text", "")

            # Clean up the response (remove assistant start token if present)
            generated_text = generated_text.removeprefix("<|im_start|>assistant\n")
            generated_text = generated_text.removesuffix("<|im_end|>")

            # Set the content in the existing model_response structure
            if hasattr(model_response, "choices") and len(model_response.choices) > 0:
                choice: Final = model_response.choices[0]
                choice.message.content = generated_text
                choice.finish_reason = "stop"

            # Set usage information if available in response
            if "usage" in response_data:
                usage_data: Final = response_data["usage"]
                setattr(
                    model_response,
                    "usage",
                    Usage(
                        prompt_tokens=usage_data.get("prompt_tokens", 0),
                        completion_tokens=usage_data.get("completion_tokens", 0),
                        total_tokens=usage_data.get("total_tokens", 0),
                    ),
                )

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
