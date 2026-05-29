"""
Tencent Hunyuan Image Generation Handler

Handles the submit → poll lifecycle, delegating data transformation
to HunyuanImageGenerationConfig.  Follows the same pattern as the BFL handler.
"""

import asyncio
import time
from typing import Any, Dict, Optional, Union

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
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

from .transformation import (
    HUNYUAN_BASE_URL,
    HUNYUAN_QUERY_ENDPOINT,
    HunyuanImageGenerationConfig,
    extract_hunyuan_extra_params,
)

HUNYUAN_POLLING_INTERVAL = 1.5  # seconds
HUNYUAN_MAX_POLLING_TIME = 600  # seconds


class HunyuanImageGeneration:
    """
    Hunyuan image generation handler.

    Manages the two-step submit + poll flow, reusing the HTTP client on every
    poll iteration.  Data transformation is delegated to
    HunyuanImageGenerationConfig.
    """

    def __init__(self) -> None:
        self.config = HunyuanImageGenerationConfig()

    # ------------------------------------------------------------------
    # Public entry points
    # ------------------------------------------------------------------

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
        data.setdefault("logo_add", 0)
        extra_params = extract_hunyuan_extra_params(litellm_params_dict)
        if extra_params:
            data.update(extra_params)

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

        return self.config.transform_image_generation_response(
            model=model,
            raw_response=final_response,
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
            else get_async_httpx_client(llm_provider=litellm.LlmProviders.HUNYUAN)
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
        data.setdefault("logo_add", 0)
        extra_params = extract_hunyuan_extra_params(litellm_params_dict)
        if extra_params:
            data.update(extra_params)

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

        return self.config.transform_image_generation_response(
            model=model,
            raw_response=final_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=data,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            encoding=None,
            api_key=api_key,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

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
            raise self.config.get_error_class(
                error_message=f"Error parsing Hunyuan submit response: {e}",
                status_code=submit_response.status_code,
                headers=submit_response.headers,
            )

        api_error = submit_data.get("error")
        if api_error:
            error_msg = api_error.get("message") or str(api_error)
            raise self.config.get_error_class(
                error_message=f"Hunyuan API error: {error_msg}",
                status_code=400,
                headers=submit_response.headers,
            )

        job_id = submit_data.get("job_id")
        if not job_id:
            raise self.config.get_error_class(
                error_message=f"Hunyuan submit response missing job_id: {submit_data}",
                status_code=500,
                headers=submit_response.headers,
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
        max_wait: float = HUNYUAN_MAX_POLLING_TIME,
        interval: float = HUNYUAN_POLLING_INTERVAL,
    ) -> httpx.Response:
        job_id, poll_headers, query_url = self._extract_poll_context(
            submit_response, api_key, litellm_params
        )
        verbose_logger.debug(f"Hunyuan polling job_id={job_id} (sync)")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            response = sync_client.post(
                url=query_url,
                json={"job_id": job_id},
                headers=poll_headers,
            )
            response.raise_for_status()

            status = self.config._check_task_status(response.json())
            if status == "done":
                return response

            time.sleep(interval)

        raise TimeoutError(f"Hunyuan task polling timed out after {max_wait} seconds")

    async def _poll_for_result_async(
        self,
        submit_response: httpx.Response,
        api_key: Optional[str],
        litellm_params: Dict,
        async_client: AsyncHTTPHandler,
        max_wait: float = HUNYUAN_MAX_POLLING_TIME,
        interval: float = HUNYUAN_POLLING_INTERVAL,
    ) -> httpx.Response:
        job_id, poll_headers, query_url = self._extract_poll_context(
            submit_response, api_key, litellm_params
        )
        verbose_logger.debug(f"Hunyuan polling job_id={job_id} (async)")

        start_time = time.time()
        while time.time() - start_time < max_wait:
            response = await async_client.post(
                url=query_url,
                json={"job_id": job_id},
                headers=poll_headers,
            )
            response.raise_for_status()

            status = self.config._check_task_status(response.json())
            if status == "done":
                return response

            await asyncio.sleep(interval)

        raise TimeoutError(f"Hunyuan task polling timed out after {max_wait} seconds")


hunyuan_image_generation = HunyuanImageGeneration()
