from typing import Final

import httpx

from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.bedrock.request_metadata import (
    bedrock_request_metadata_headers,
    merge_bedrock_invoke_headers,
)
from litellm.llms.openai.chat.gpt_transformation import OpenAIGPTConfig
from litellm.types.llms.openai import AllMessageValues


class AmazonBedrockOpenAIChatCompletionsConfig(OpenAIGPTConfig, BaseAWSLLM):
    def __init__(self, **kwargs: object) -> None:  # kwargs-ok: provider interface
        OpenAIGPTConfig.__init__(self, **kwargs)
        BaseAWSLLM.__init__(self, **kwargs)

    @property
    def custom_llm_provider(self) -> str:
        return "bedrock"

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: provider interface
        litellm_params: dict,  # mutable-ok: provider interface
        stream: bool | None = None,
    ) -> str:
        aws_region_name: Final = self._get_aws_region_name(optional_params=optional_params, model=model)
        endpoint_url: Final = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=optional_params.get("aws_bedrock_runtime_endpoint"),
            aws_region_name=aws_region_name,
        )[0]
        return f"{endpoint_url.rstrip('/')}/openai/v1/chat/completions"

    def transform_request(
        self,
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: provider interface
        optional_params: dict,  # mutable-ok: provider interface
        litellm_params: dict,  # mutable-ok: provider interface
        headers: dict,  # mutable-ok: provider interface
    ) -> dict:  # mutable-ok: provider interface
        inference_params: Final = {  # mutable-ok: provider interface
            key: value for key, value in optional_params.items() if key not in self.aws_authentication_params
        }
        return super().transform_request(
            model=model.removeprefix("bedrock/"),
            messages=messages,
            optional_params=inference_params,
            litellm_params=litellm_params,
            headers=headers,
        )

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: provider interface
        model: str,
        messages: list[AllMessageValues],  # mutable-ok: provider interface
        optional_params: dict,  # mutable-ok: provider interface
        litellm_params: dict,  # mutable-ok: provider interface
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:  # mutable-ok: provider interface
        owned_names, metadata_headers = bedrock_request_metadata_headers(litellm_params)
        return merge_bedrock_invoke_headers(
            headers,
            (),
            metadata_headers,
            owned_names,
        )

    def sign_request(
        self,
        headers: dict,  # mutable-ok: provider interface
        optional_params: dict,  # mutable-ok: provider interface
        request_data: dict,  # mutable-ok: provider interface
        api_base: str,
        api_key: str | None = None,
        model: str | None = None,
        stream: bool | None = None,
        fake_stream: bool | None = None,
    ) -> tuple[dict, bytes | None]:  # mutable-ok: provider interface
        return self._sign_request(
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

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, object] | httpx.Headers,  # mutable-ok: provider interface
    ) -> BedrockError:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)
