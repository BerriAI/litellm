"""
OpenAI Image Variations Handler
"""

from typing import Callable, Optional

import httpx
from openai import AsyncOpenAI, OpenAI

import litellm
from litellm.types.utils import FileTypes, ImageResponse, LlmProviders
from litellm.utils import ProviderConfigManager

from ...custom_httpx.llm_http_handler import LiteLLMLoggingObj
from ..common_utils import OpenAIError


class OpenAIImageVariationsHandler:
    def get_sync_client(
        self,
        client: Optional[OpenAI],
        init_client_params: dict,
    ):
        if client is None:
            openai_client = OpenAI(
                **init_client_params,
            )
        else:
            openai_client = client
        return openai_client

    async def get_async_client(
        self, client: Optional[AsyncOpenAI], init_client_params: dict
    ):
        if client is None:
            openai_client = AsyncOpenAI(
                **init_client_params,
            )
        else:
            openai_client = client
        return openai_client

    async def async_image_variations(self, *args, **kwargs):
        pass

    def image_variations(
        self,
        model_response: ImageResponse,
        api_key: str,
        model: Optional[str],
        image: FileTypes,
        timeout: float,
        custom_llm_provider: str,
        logging_obj: LiteLLMLoggingObj,
        optional_params: dict,
        litellm_params: dict,
        print_verbose: Optional[Callable] = None,
        api_base: Optional[str] = None,
        aimage_variation: bool = False,
        logger_fn=None,
        client=None,
        organization: Optional[str] = None,
        headers: Optional[dict] = None,
    ):
        try:
            provider_config = ProviderConfigManager.get_provider_image_variation_config(
                model=model or "",  # openai defaults to dall-e-2
                provider=LlmProviders.OPENAI,
            )

            if provider_config is None:
                raise ValueError(
                    f"image variation provider not found: {custom_llm_provider}."
                )

            max_retries = optional_params.pop("max_retries", 2)

            data = provider_config.transform_request_image_variation(
                model=model,
                image=image,
                optional_params=optional_params,
                headers=headers,
            )
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
            if aimage_variation is True:
                return self.async_image_variations(api_base=api_base, data=data, headers=headers, model_response=model_response, api_key=api_key, logging_obj=logging_obj, model=model, timeout=timeout, max_retries=max_retries, organization=organization, client=client)  # type: ignore

            init_client_params = {
                "api_key": api_key,
                "base_url": api_base,
                "http_client": litellm.client_session,
                "timeout": timeout,
                "max_retries": max_retries,  # type: ignore
                "organization": organization,
            }

            client = self.get_sync_client(
                client=client, init_client_params=init_client_params
            )

            raw_response = client.images.with_raw_response.create_variation(**data)  # type: ignore
            response = raw_response.parse()
            response_json = response.model_dump()

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
                    status_code=raw_response.status_code,
                    headers=raw_response.headers,
                    text="dummy_text",
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
            status_code = getattr(e, "status_code", 500)
            error_headers = getattr(e, "headers", None)
            error_text = getattr(e, "text", str(e))
            error_response = getattr(e, "response", None)
            if error_headers is None and error_response:
                error_headers = getattr(error_response, "headers", None)
            raise OpenAIError(
                status_code=status_code, message=error_text, headers=error_headers
            )
