import json
from collections.abc import Callable
from typing import Final

import httpx

from litellm.llms.anthropic.experimental_pass_through.messages.transformation import (
    DEFAULT_ANTHROPIC_API_VERSION,
    AnthropicMessagesConfig,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.bedrock.base_aws_llm import BaseAWSLLM, bedrock_bearer_token
from litellm.llms.bedrock.common_utils import BedrockError, build_mantle_messages_url
from litellm.types.router import GenericLiteLLMParams

MessagesURLBuilder = Callable[[str | None, str | None, str], str]  # mutable-ok: provider interface  # rebind-ok: test capture
BearerTokenResolver = Callable[[str | None], str | None]  # mutable-ok: provider interface  # rebind-ok: test capture


class AmazonBedrockNativeMessagesConfig(BaseAWSLLM, AnthropicMessagesConfig):
    preserves_request_body: bool = True

    def __init__(
        self,
        custom_llm_provider: str = "bedrock",
        url_builder: MessagesURLBuilder | None = None,
        bearer_token_resolver: BearerTokenResolver = bedrock_bearer_token,
        **kwargs: object,  # kwargs-ok: provider interface
    ) -> None:
        BaseAWSLLM.__init__(self, **kwargs)
        self._custom_llm_provider = custom_llm_provider
        self._url_builder = url_builder
        self._bearer_token_resolver = bearer_token_resolver

    @property
    def custom_llm_provider(self) -> str:
        return self._custom_llm_provider

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: dict,  # mutable-ok: provider interface
        litellm_params: dict,  # mutable-ok: provider interface
        stream: bool | None = None,
    ) -> str:
        region: Final = self._get_aws_region_name(optional_params=optional_params, model=model)
        endpoint: Final = optional_params.get("aws_bedrock_runtime_endpoint")
        if self._url_builder is not None:
            return self._url_builder(api_base, endpoint, region)
        runtime_endpoint: Final = self.get_runtime_endpoint(
            api_base=api_base,
            aws_bedrock_runtime_endpoint=endpoint,
            aws_region_name=region,
        )[0]
        return f"{runtime_endpoint.rstrip('/')}/anthropic/v1/messages"

    def validate_anthropic_messages_environment(
        self,
        headers: dict,  # mutable-ok: provider interface
        model: str,
        messages: list[dict],  # mutable-ok: provider interface
        optional_params: dict,  # mutable-ok: provider interface
        litellm_params: dict,  # mutable-ok: provider interface
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> tuple[dict, str | None]:  # mutable-ok: provider interface
        return (
            {  # mutable-ok: provider interface
                **headers,
                "anthropic-version": headers.get("anthropic-version", DEFAULT_ANTHROPIC_API_VERSION),
                "content-type": headers.get("content-type", "application/json"),
            },
            api_base,
        )

    def transform_anthropic_messages_request(
        self,
        model: str,
        messages: list[dict],  # mutable-ok: provider interface
        anthropic_messages_optional_request_params: dict,  # mutable-ok: provider interface
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: provider interface
    ) -> dict:  # mutable-ok: provider interface
        return {  # mutable-ok: provider interface
            **anthropic_messages_optional_request_params,
            "model": model,
            "messages": messages,
        }

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
        bearer: Final = self._bearer_token_resolver(api_key)
        if bearer:
            return {**headers, "Authorization": f"Bearer {bearer}"}, json.dumps(request_data).encode()  # mutable-ok: provider interface
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
        self, error_message: str, status_code: int, headers: dict[str, object] | httpx.Headers  # mutable-ok: provider interface
    ) -> BaseLLMException:
        return BedrockError(status_code=status_code, message=error_message, headers=headers)


def mantle_native_messages_config() -> AmazonBedrockNativeMessagesConfig:
    from litellm.llms.bedrock_mantle.common_utils import resolve_mantle_bearer_token

    return AmazonBedrockNativeMessagesConfig(
        custom_llm_provider="bedrock_mantle",
        url_builder=build_mantle_messages_url,
        bearer_token_resolver=resolve_mantle_bearer_token,
    )
