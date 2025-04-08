"""
Translate from OpenAI's `/v1/chat/completions` to Sagemaker's `/invocations` API

Called if Sagemaker endpoint supports HF Messages API.

LiteLLM Docs: https://docs.litellm.ai/docs/providers/aws_sagemaker#sagemaker-messages-api
Huggingface Docs: https://huggingface.co/docs/text-generation-inference/en/messages_api
"""

from typing import Union

from httpx._models import Headers

from litellm.llms.base_llm.chat.transformation import BaseLLMException

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import SagemakerError


class SagemakerChatConfig(OpenAIGPTConfig):
    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return SagemakerError(
            status_code=status_code, message=error_message, headers=headers
        )
