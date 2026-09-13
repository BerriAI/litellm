from typing import Final

import httpx

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM
from litellm.llms.bedrock.common_utils import BedrockError
from litellm.llms.openai.responses.transformation import OpenAIResponsesAPIConfig
from litellm.types.llms.openai import ResponseInputParam
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders


class AmazonBedrockResponsesAPIConfig(OpenAIResponsesAPIConfig, BaseAWSLLM):
    def __init__(self, **kwargs: object) -> None:  # kwargs-ok: provider interface
        OpenAIResponsesAPIConfig.__init__(self)
        BaseAWSLLM.__init__(self, **kwargs)

    @property
    def custom_llm_provider(self) -> LlmProviders:
        return LlmProviders.BEDROCK

    def transform_responses_api_request(
        self,
        model: str,
        input: str | ResponseInputParam,
        response_api_optional_request_params: dict,  # mutable-ok: provider interface
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: provider interface
    ) -> dict:  # mutable-ok: provider interface
        request_params: Final = {  # mutable-ok: provider interface
            key: value
            for key, value in response_api_optional_request_params.items()
            if key not in self.aws_authentication_params
        }
        return {  # mutable-ok: provider interface
            **request_params,
            "model": model,
            "input": input,
        }

    def get_complete_url(self, api_base: str | None, litellm_params: dict) -> str:  # mutable-ok: provider interface
        region: Final = self._get_aws_region_name(optional_params=litellm_params, model="")
        endpoint_url: Final = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=litellm_params.get("aws_bedrock_runtime_endpoint"),
            aws_region_name=region,
        )[0]
        return f"{endpoint_url.rstrip('/')}/openai/v1/responses"

    def validate_environment(
        self, headers: dict, model: str, litellm_params: GenericLiteLLMParams | None
    ) -> dict:  # mutable-ok: provider interface
        return {
            **headers,
            "Content-Type": headers.get("Content-Type", "application/json"),
        }  # mutable-ok: provider interface

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
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)
