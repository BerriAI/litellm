"""
Translate from OpenAI's `/v1/chat/completions` to Sagemaker's `/invocations` API

Called if Sagemaker endpoint supports HF Messages API.

LiteLLM Docs: https://docs.litellm.ai/docs/providers/aws_sagemaker#sagemaker-messages-api
Huggingface Docs: https://huggingface.co/docs/text-generation-inference/en/messages_api
"""

from typing import List, Optional, Union, cast

from httpx._models import Headers

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.types.llms.openai import AllMessageValues

from ...openai.chat.gpt_transformation import OpenAIGPTConfig
from ..common_utils import SagemakerError


class SagemakerChatConfig(OpenAIGPTConfig, BaseAWSLLM):
    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, Headers]
    ) -> BaseLLMException:
        return SagemakerError(
            status_code=status_code, message=error_message, headers=headers
        )

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
        aws_region_name = self._get_aws_region_name(
            optional_params=optional_params,
            model=model,
            model_id=None,
        )
        if stream is True:
            api_base = f"https://runtime.sagemaker.{aws_region_name}.amazonaws.com/endpoints/{model}/invocations-response-stream"
        else:
            api_base = f"https://runtime.sagemaker.{aws_region_name}.amazonaws.com/endpoints/{model}/invocations"

        sagemaker_base_url = cast(
            Optional[str], optional_params.get("sagemaker_base_url")
        )
        if sagemaker_base_url is not None:
            api_base = sagemaker_base_url

        return api_base
