"""
Tencent Hunyuan Image Edit Handler

Handles the submit → poll lifecycle for image editing, delegating data
transformation to HunyuanImageEditConfig.  Follows the same pattern as
the HunyuanImageGeneration handler.
"""

import asyncio
import time
from typing import Any, Dict, List, Optional, Union

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

from ..image_generation.transformation import (
    HUNYUAN_BASE_URL,
    HUNYUAN_QUERY_ENDPOINT,
)
from .transformation import HunyuanImageEditConfig

HUNYUAN_EDIT_POLLING_INTERVAL = 1.5
HUNYUAN_EDIT_MAX_POLLING_TIME = 600


class HunyuanImageEdit:
    """
    Hunyuan image edit handler.

    Manages the two-step submit + poll flow, reusing the HTTP client on every
    poll iteration.  Data transformation is delegated to HunyuanImageEditConfig.
    """

    def __init__(self) -> None:
        self.config = HunyuanImageEditConfig()

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

        submit_response = sync_client.post(
            url=complete_url,
            headers=headers,
            json=data,
            timeout=timeout,
        )
        submit_response.raise_for_status()

        final_response = self._poll_for_result_sync(
            submit_response=submit_response,
            api_key=api_key,
            litellm_params=litellm_params_dict,
            sync_client=sync_client,
        )

        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=final_response.text,
        )

        return self.config.transform_image_edit_response(
            model=model,
            raw_response=final_response,
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
            else get_async_httpx_client(llm_provider=litellm.LlmProviders.HUNYUAN)
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

        submit_response = await async_client.post(
            url=complete_url,
            headers=headers,
            json=data,
            timeout=timeout,
        )
        submit_response.raise_for_status()

        final_response = await self._poll_for_result_async(
            submit_response=submit_response,
            api_key=api_key,
            litellm_params=litellm_params_dict,
            async_client=async_client,
        )

        logging_obj.post_call(
            input=prompt,
            api_key=api_key,
            additional_args={"complete_input_dict": data},
            original_response=final_response.text,
        )

        return self.config.transform_image_edit_response(
            model=model,
            raw_response=final_response,
            logging_obj=logging_obj,
        )

    def _extract_poll_context(
        self,
        submit_response: httpx.Response,
        api_key: Optional[str],
        litellm_params: Dict,
    ) -> tuple:
        """Return (job_id, poll_headers, query_url) from the submit response."""
        try:
            submit_data = submit_response.json()
        except Exception as e:
            raise BaseLLMException(
                status_code=submit_response.status_code,
                message=f"Error parsing Hunyuan submit response: {e}",
            )

        api_error = submit_data.get("error")
        if api_error:
            error_msg = api_error.get("message") or str(api_error)
            raise BaseLLMException(
                status_code=400,
                message=f"Hunyuan API error: {error_msg}",
            )

        job_id = submit_data.get("job_id")
        if not job_id:
            raise BaseLLMException(
                status_code=500,
                message=f"Hunyuan submit response missing job_id: {submit_data}",
            )

        resolved_key = api_key or get_secret_str("HUNYUAN_API_KEY") or ""
        poll_headers = {
            "Authorization": resolved_key,
            "Content-Type": "application/json",
        }

        api_base = litellm_params.get("api_base") or HUNYUAN_BASE_URL
        query_url = f"{api_base.rstrip('/')}/{HUNYUAN_QUERY_ENDPOINT}"

        return job_id, poll_headers, query_url

    def _poll_for_result_sync(
        self,
        submit_response: httpx.Response,
        api_key: Optional[str],
        litellm_params: Dict,
        sync_client: HTTPHandler,
        max_wait: float = HUNYUAN_EDIT_MAX_POLLING_TIME,
        interval: float = HUNYUAN_EDIT_POLLING_INTERVAL,
    ) -> httpx.Response:
        job_id, poll_headers, query_url = self._extract_poll_context(
            submit_response, api_key, litellm_params
        )
        verbose_logger.debug(f"Hunyuan image edit polling job_id={job_id} (sync)")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            response = sync_client.post(
                url=query_url,
                json={"job_id": job_id},
                headers=poll_headers,
            )
            response.raise_for_status()

            data = response.json()
            status = data.get("status", "").upper()
            verbose_logger.debug(f"Hunyuan image edit task status: {status}")

            if status == "DONE":
                return response
            elif status == "FAIL":
                raise BaseLLMException(
                    status_code=400,
                    message=f"Hunyuan image edit failed: {data.get('message', 'Unknown error')}",
                )

            time.sleep(interval)

        raise TimeoutError(
            f"Hunyuan image edit polling timed out after {max_wait} seconds"
        )

    async def _poll_for_result_async(
        self,
        submit_response: httpx.Response,
        api_key: Optional[str],
        litellm_params: Dict,
        async_client: AsyncHTTPHandler,
        max_wait: float = HUNYUAN_EDIT_MAX_POLLING_TIME,
        interval: float = HUNYUAN_EDIT_POLLING_INTERVAL,
    ) -> httpx.Response:
        job_id, poll_headers, query_url = self._extract_poll_context(
            submit_response, api_key, litellm_params
        )
        verbose_logger.debug(f"Hunyuan image edit polling job_id={job_id} (async)")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            response = await async_client.post(
                url=query_url,
                json={"job_id": job_id},
                headers=poll_headers,
            )
            response.raise_for_status()

            data = response.json()
            status = data.get("status", "").upper()
            verbose_logger.debug(f"Hunyuan image edit task status: {status}")

            if status == "DONE":
                return response
            elif status == "FAIL":
                raise BaseLLMException(
                    status_code=400,
                    message=f"Hunyuan image edit failed: {data.get('message', 'Unknown error')}",
                )

            await asyncio.sleep(interval)

        raise TimeoutError(
            f"Hunyuan image edit polling timed out after {max_wait} seconds"
        )


hunyuan_image_edit = HunyuanImageEdit()
