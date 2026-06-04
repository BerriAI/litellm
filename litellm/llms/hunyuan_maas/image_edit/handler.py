"""
Tencent Hunyuan Maas Image Edit Handler

The Maas API is synchronous: a single POST to /v1/aiart/gtimage returns the
result directly, no polling required.  Data transformation is delegated to
HunyuanMaasImageEditConfig.
"""

from typing import Any, Dict, List, Optional, Union

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
from litellm.types.utils import FileTypes, ImageResponse

from .transformation import HunyuanMaasImageEditConfig


class HunyuanMaasImageEdit:
    """
    Synchronous Hunyuan Maas image edit handler.

    A single POST call returns the result; no polling is required.
    """

    def __init__(self) -> None:
        self.config = HunyuanMaasImageEditConfig()

    def image_edit(
        self,
        model: str,
        image: Union[FileTypes, List[FileTypes]],
        prompt: Optional[str],
        image_edit_optional_request_params: Dict,
        litellm_params: Union[GenericLiteLLMParams, Dict],
        logging_obj: LiteLLMLoggingObj,
        timeout: Optional[Union[float, httpx.Timeout]],
        extra_headers: Optional[Dict[str, Any]] = None,
        client: Optional[Union[HTTPHandler, AsyncHTTPHandler]] = None,
        aimage_edit: bool = False,
    ) -> Union[ImageResponse, Any]:
        if aimage_edit:
            return self.async_image_edit(
                model=model,
                image=image,
                prompt=prompt,
                image_edit_optional_request_params=image_edit_optional_request_params,
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
            headers=image_edit_optional_request_params.get("extra_headers", {}) or {},
            model=model,
            api_key=api_key,
        )
        if extra_headers:
            headers.update(extra_headers)

        complete_url = self.config.get_complete_url(
            model=model,
            api_base=api_base,
            litellm_params=litellm_params_dict,
        )

        data, _ = self.config.transform_image_edit_request(
            model=model,
            prompt=prompt,
            image=image,
            image_edit_optional_request_params=image_edit_optional_request_params,
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

        return self.config.transform_image_edit_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )

    async def async_image_edit(
        self,
        model: str,
        image: Union[FileTypes, List[FileTypes]],
        prompt: Optional[str],
        image_edit_optional_request_params: Dict,
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
                llm_provider=litellm.LlmProviders.HUNYUAN_MAAS
            )
        )

        headers = self.config.validate_environment(
            headers=image_edit_optional_request_params.get("extra_headers", {}) or {},
            model=model,
            api_key=api_key,
        )
        if extra_headers:
            headers.update(extra_headers)

        complete_url = self.config.get_complete_url(
            model=model,
            api_base=api_base,
            litellm_params=litellm_params_dict,
        )

        data, _ = self.config.transform_image_edit_request(
            model=model,
            prompt=prompt,
            image=image,
            image_edit_optional_request_params=image_edit_optional_request_params,
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

        return self.config.transform_image_edit_response(
            model=model,
            raw_response=response,
            logging_obj=logging_obj,
        )


hunyuan_maas_image_edit = HunyuanMaasImageEdit()
