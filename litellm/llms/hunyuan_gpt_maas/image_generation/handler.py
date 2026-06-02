"""
Tencent Hunyuan GPT-Maas Image Generation Handler (Text-to-Image)

The GPT-Maas API is synchronous: a single POST returns the result directly,
no submit+poll required.  Data transformation is delegated to
HunyuanGptMaasImageGenerationConfig.
"""

from typing import Any, Dict, Optional, Union

import httpx

import litellm
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

from .transformation import HunyuanGptMaasImageGenerationConfig


class HunyuanGptMaasImageGeneration:
    """
    Synchronous Hunyuan GPT-Maas text-to-image handler.

    A single POST call returns the result; no polling is required.
    """

    def __init__(self) -> None:
        self.config = HunyuanGptMaasImageGenerationConfig()

    def image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: Dict,
        litellm_params: Union[GenericLiteLLMParams, Dict],
        logging_obj: LiteLLMLoggingObj,
        timeout: Optional[Union[float, httpx.Timeout]],
        extra_headers: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        aimg_generation: bool = False,
    ) -> Union[ImageResponse, Any]:
        if aimg_generation:
            return self.async_image_generation(
                model=model,
                prompt=prompt,
                model_response=model_response,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                timeout=timeout,
                extra_headers=extra_headers,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        litellm_params_dict = (
            litellm_params if isinstance(litellm_params, dict) else dict(litellm_params)
        )
        api_key = litellm_params_dict.get("api_key")
        api_base = litellm_params_dict.get("api_base")

        sync_client: HTTPHandler = (
            client  # type: ignore[assignment]
            if isinstance(client, HTTPHandler)
            else _get_httpx_client()
        )

        headers = self.config.validate_environment(
            api_key=api_key,
            headers={},
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )
        if extra_headers:
            headers.update(extra_headers)

        complete_url = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )

        data = self.config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            headers=headers,
        )

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": complete_url,
                "headers": headers,
            },
        )

        response = sync_client.post(
            url=complete_url,
            headers=headers,
            json=data,
            timeout=timeout,
        )
        response.raise_for_status()

        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response.text,
        )

        return self.config.transform_image_generation_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=data,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            encoding=None,
            api_key=api_key,
        )

    async def async_image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: Dict,
        litellm_params: Union[GenericLiteLLMParams, Dict],
        logging_obj: LiteLLMLoggingObj,
        timeout: Optional[Union[float, httpx.Timeout]],
        extra_headers: Optional[Dict[str, Any]] = None,
        client: Optional[AsyncHTTPHandler] = None,
    ) -> ImageResponse:
        litellm_params_dict = (
            litellm_params if isinstance(litellm_params, dict) else dict(litellm_params)
        )
        api_key = litellm_params_dict.get("api_key")
        api_base = litellm_params_dict.get("api_base")

        async_client: AsyncHTTPHandler = (
            client
            if isinstance(client, AsyncHTTPHandler)
            else get_async_httpx_client(
                llm_provider=litellm.LlmProviders.HUNYUAN_GPT_MAAS
            )
        )

        headers = self.config.validate_environment(
            api_key=api_key,
            headers={},
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )
        if extra_headers:
            headers.update(extra_headers)

        complete_url = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )

        data = self.config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            headers=headers,
        )

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": complete_url,
                "headers": headers,
            },
        )

        response = await async_client.post(
            url=complete_url,
            headers=headers,
            json=data,
            timeout=timeout,
        )
        response.raise_for_status()

        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=response.text,
        )

        return self.config.transform_image_generation_response(
            model=model,
            raw_response=response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=data,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            encoding=None,
            api_key=api_key,
        )


hunyuan_gpt_maas_image_generation = HunyuanGptMaasImageGeneration()
