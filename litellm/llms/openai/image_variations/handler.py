"""
OpenAI Image Variations Handler
"""

from collections.abc import Callable
from typing import Final

import httpx
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.types.utils import FileTypes, ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager

from ...base_llm.image_variations.transformation import BaseImageVariationConfig
from ...custom_httpx.llm_http_handler import LiteLLMLoggingObj
from ..common_utils import OpenAIError


class OpenAIImageVariationsHandler:
    def get_sync_client(
        self,
        client: OpenAI | None,
        init_client_params: dict,
    ):
        if client is None:
            openai_client = OpenAI(
                **init_client_params,
            )
        else:
            openai_client = client
        return openai_client

    def get_async_client(self, client: AsyncOpenAI | None, init_client_params: dict) -> AsyncOpenAI:
        if client is None:
            openai_client = AsyncOpenAI(
                **init_client_params,
            )
        else:
            openai_client = client
        return openai_client

    async def async_image_variations(
        self,
        api_key: str,
        api_base: str,
        organization: str | None,
        client: AsyncOpenAI | None,
        data: dict,
        headers: dict,
        model: str | None,
        timeout: float | None,
        max_retries: int,
        logging_obj: LiteLLMLoggingObj,
        model_response: ImageResponse,
        optional_params: dict,
        litellm_params: dict,
        image: FileTypes,
        provider_config: BaseImageVariationConfig,
    ) -> ImageResponse:
        try:
            init_client_params: Final = {
                "api_key": api_key,
                "base_url": api_base,
                "http_client": litellm.client_session,
                "timeout": timeout,
                "max_retries": max_retries,
                "organization": organization,
            }

            client = self.get_async_client(client=client, init_client_params=init_client_params)

            raw_response: Final = await client.images.with_raw_response.create_variation(**data)
            response: Final = raw_response.parse()
            response_json: Final = response.model_dump()

            ## LOGGING
            logging_obj.post_call(
                api_key=api_key,
                original_response=response_json,
                additional_args={
                    "headers": headers,
                    "api_base": api_base,
                },
            )

            ## RESPONSE OBJECT
            return provider_config.transform_response_image_variation(
                model=model,
                model_response=ImageResponse(**response_json),
                raw_response=httpx.Response(
                    status_code=200,
                    request=httpx.Request(method="GET", url="https://litellm.ai"),  # mock request object
                ),
                logging_obj=logging_obj,
                request_data=data,
                image=image,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=None,
                api_key=api_key,
            )
        except Exception as e:
            status_code: Final = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text: Final = getattr(e, "text", str(e))
            error_response: Final = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(status_code=status_code, message=error_text, headers=error_headers)

    def image_variations(
        self,
        model_response: ImageResponse,
        api_key: str,
        api_base: str,
        model: str | None,
        image: FileTypes,
        timeout: float | None,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        print_verbose: Callable | None = None,
        logger_fn=None,
        client=None,
        organization: str | None = None,
        headers: dict | None = None,
    ) -> ImageResponse:
        try:
            provider_config: Final = ProviderConfigManager.get_provider_image_variation_config(
                model=model or "",  # openai defaults to dall-e-2
                provider=LlmProviders.OPENAI,
            )

            if provider_config is None:
                raise ValueError(f"image variation provider not found: {custom_llm_provider}.")

            max_retries: Final = optional_params.pop("max_retries", 2)

            data: Final = provider_config.transform_request_image_variation(
                model=model,
                image=image,
                optional_params=optional_params,
                headers=headers or {},
            )
            json_data: Final = data.get("data")
            if not json_data:
                raise ValueError(f"data field is required, for openai image variations. Got={data}")
            ## LOGGING
            logging_obj.pre_call(
                input="",
                api_key=api_key,
                additional_args={
                    "headers": headers,
                    "api_base": api_base,
                    "complete_input_dict": data,
                },
            )
            if litellm_params.get("async_call", False):
                return self.async_image_variations(
                    api_base=api_base,
                    data=json_data,
                    headers=headers or {},
                    model_response=model_response,
                    api_key=api_key,
                    logging_obj=logging_obj,
                    model=model,
                    timeout=timeout,
                    max_retries=max_retries,
                    organization=organization,
                    client=client,
                    provider_config=provider_config,
                    image=image,
                    optional_params=optional_params,
                    litellm_params=litellm_params,
                )

            init_client_params: Final = {
                "api_key": api_key,
                "base_url": api_base,
                "http_client": litellm.client_session,
                "timeout": timeout,
                "max_retries": max_retries,
                "organization": organization,
            }

            client = self.get_sync_client(client=client, init_client_params=init_client_params)

            raw_response: Final = client.images.with_raw_response.create_variation(**json_data)
            response: Final = raw_response.parse()
            response_json: Final = response.model_dump()

            ## LOGGING
            logging_obj.post_call(
                api_key=api_key,
                original_response=response_json,
                additional_args={
                    "headers": headers,
                    "api_base": api_base,
                },
            )

            ## RESPONSE OBJECT
            return provider_config.transform_response_image_variation(
                model=model,
                model_response=ImageResponse(**response_json),
                raw_response=httpx.Response(
                    status_code=200,
                    request=httpx.Request(method="GET", url="https://litellm.ai"),  # mock request object
                ),
                logging_obj=logging_obj,
                request_data=json_data,
                image=image,
                optional_params=optional_params,
                litellm_params=litellm_params,
                encoding=None,
                api_key=api_key,
            )
        except Exception as e:
            status_code: Final = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text: Final = getattr(e, "text", str(e))
            error_response: Final = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(status_code=status_code, message=error_text, headers=error_headers)
