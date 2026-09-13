"""
Native OpenAI Chat Completions on Amazon Bedrock Runtime.

AWS serves this surface at
``https://bedrock-runtime.{region}.amazonaws.com/openai/v1/chat/completions``.
Grok 4.6 on runtime is one of the models that uses it: chat completions stay
chat completions instead of being rewritten to Converse.

Usage: model="us.xai.grok-4.6" or model="bedrock/us.xai.grok-4.6"
Explicit ``bedrock/converse/...`` still uses Converse.
"""

from collections.abc import AsyncIterator, Iterator
from typing import Any, Final

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError, strip_bedrock_routing_prefix
from litellm.llms.openai_like.chat.transformation import OpenAILikeChatConfig
from litellm.types.llms.openai import AllMessageValues


class AmazonBedrockRuntimeChatCompletionsConfig(OpenAILikeChatConfig):
    def __init__(self, aws_signer: BaseAWSLLM | None = None):
        super().__init__()
        self._aws_signer: Final = aws_signer or BaseAWSLLM()

    @property
    def custom_llm_provider(self) -> str | None:
        return "bedrock"

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, object] | httpx.Headers
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: bool | None = None,
    ) -> str:
        if api_base is not None and "chat/completions" in api_base:
            return api_base.rstrip("/")
        aws_region_name: Final = self._aws_signer._get_aws_region_name(optional_params=optional_params, model=model)
        endpoint_url, _ = self._aws_signer.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=optional_params.get("aws_bedrock_runtime_endpoint"),
            aws_region_name=aws_region_name,
        )
        base: Final = endpoint_url.rstrip("/")
        if base.endswith("/openai/v1/chat/completions"):
            return base
        if base.endswith("/openai/v1"):
            return f"{base}/chat/completions"
        return f"{base}/openai/v1/chat/completions"

    def sign_request(
        self,
        headers: dict,
        optional_params: dict,
        request_data: dict,
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict, bytes | None]:
        return self._aws_signer._sign_request(
            service_name="bedrock",
            headers=headers,
            optional_params=optional_params,
            request_data=request_data,
            api_base=api_base,
            api_key=api_key,
            model=model,
            stream=stream,
            fake_stream=fake_stream,
        )

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        inference_params: Final = {
            k: v for k, v in optional_params.items() if k not in self._aws_signer.aws_authentication_params
        }
        return super().transform_request(
            model=strip_bedrock_routing_prefix(model),
            messages=messages,
            optional_params=inference_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    async def async_transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        inference_params: Final = {
            k: v for k, v in optional_params.items() if k not in self._aws_signer.aws_authentication_params
        }
        return await super().async_transform_request(
            model=strip_bedrock_routing_prefix(model),
            messages=messages,
            optional_params=inference_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: list[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        headers = super().validate_environment(
            headers=headers,
            model=model,
            messages=messages,
            optional_params=optional_params,
            litellm_params=litellm_params,
            api_key=api_key,
            api_base=api_base,
        )
        project_id: Final = litellm_params.get("aws_bedrock_project_id")
        if project_id:
            headers["OpenAI-Project"] = project_id
        return headers

    def get_supported_openai_params(self, model: str) -> list:
        base_params: Final = super().get_supported_openai_params(model)
        try:
            if litellm.supports_reasoning(model=model, custom_llm_provider=self.custom_llm_provider):
                if "reasoning_effort" not in base_params:
                    base_params.append("reasoning_effort")
        except Exception as e:
            verbose_logger.debug("AmazonBedrockRuntimeChatCompletionsConfig: error checking reasoning support: %s", e)
        return base_params

    def get_model_response_iterator(
        self,
        streaming_response: Iterator[str] | AsyncIterator[str] | Any,
        sync_stream: bool,
        json_mode: bool | None = False,
    ) -> Any:
        from litellm.llms.openai.chat.gpt_transformation import (
            OpenAIChatCompletionStreamingHandler,
        )

        return OpenAIChatCompletionStreamingHandler(
            streaming_response=streaming_response,
            sync_stream=sync_stream,
            json_mode=json_mode,
        )
