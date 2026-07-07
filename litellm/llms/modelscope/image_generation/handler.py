"""
ModelScope Image Generation Handler

ModelScope image generation only supports async mode: the submit call returns
a task_id, then the caller polls GET /v1/tasks/{task_id} until task_status is
SUCCEED (image URLs in output_images) or FAILED.

API Reference: https://modelscope.cn/docs/model-service/API-Inference/intro
"""

import asyncio
import json
import time
from typing import Any

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
from litellm.llms.modelscope.common_utils import (
    DEFAULT_MAX_POLLING_TIME,
    DEFAULT_POLLING_INTERVAL,
    TASK_STATUS_FAILED,
    TASK_STATUS_SUCCEED,
    ModelScopeError,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

from .transformation import ModelScopeImageGenerationConfig


class ModelScopeImageGeneration:
    """
    ModelScope image generation handler.

    Owns the submit + poll HTTP flow; request/response shaping is delegated to
    ModelScopeImageGenerationConfig.
    """

    def __init__(self):
        self.config = ModelScopeImageGenerationConfig()

    def image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: dict,
        litellm_params: GenericLiteLLMParams | dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        extra_headers: dict[str, Any] | None = None,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        aimg_generation: bool = False,
    ) -> ImageResponse | Any:
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

        if isinstance(litellm_params, dict):
            api_key = litellm_params.get("api_key")
            api_base = litellm_params.get("api_base")
            litellm_params_dict = litellm_params
        else:
            api_key = litellm_params.api_key
            api_base = litellm_params.api_base
            litellm_params_dict = dict(litellm_params)

        sync_client = client if isinstance(client, HTTPHandler) else _get_httpx_client()

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

        try:
            response = sync_client.post(
                url=complete_url,
                headers=headers,
                json=data,
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            raise ModelScopeError(
                status_code=500,
                message=f"Request failed: {e}",
            )

        final_response = self._poll_for_result_sync(
            initial_response=response,
            api_base=api_base,
            headers=headers,
            sync_client=sync_client,
            timeout=timeout,
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
        )

    async def async_image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: dict,
        litellm_params: GenericLiteLLMParams | dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        extra_headers: dict[str, Any] | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ImageResponse:
        if isinstance(litellm_params, dict):
            api_key = litellm_params.get("api_key")
            api_base = litellm_params.get("api_base")
            litellm_params_dict = litellm_params
        else:
            api_key = litellm_params.api_key
            api_base = litellm_params.api_base
            litellm_params_dict = dict(litellm_params)

        async_client = client or get_async_httpx_client(
            llm_provider=litellm.LlmProviders.MODELSCOPE,
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

        try:
            response = await async_client.post(
                url=complete_url,
                headers=headers,
                json=data,
                timeout=timeout,
            )
        except httpx.HTTPError as e:
            raise ModelScopeError(
                status_code=500,
                message=f"Request failed: {e}",
            )

        final_response = await self._poll_for_result_async(
            initial_response=response,
            api_base=api_base,
            headers=headers,
            async_client=async_client,
            timeout=timeout,
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
        )

    def _poll_for_result_sync(
        self,
        initial_response: httpx.Response,
        api_base: str | None,
        headers: dict,
        sync_client: HTTPHandler,
        max_wait: float = DEFAULT_MAX_POLLING_TIME,
        interval: float = DEFAULT_POLLING_INTERVAL,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response:
        if initial_response.status_code >= 400:
            raise ModelScopeError(
                status_code=initial_response.status_code,
                message=f"ModelScope submit failed: {initial_response.text}",
            )

        try:
            response_data = initial_response.json()
        except json.JSONDecodeError as e:
            raise ModelScopeError(
                status_code=initial_response.status_code,
                message=f"Error parsing submit response: {e}",
            )

        if "errors" in response_data:
            raise ModelScopeError(
                status_code=initial_response.status_code,
                message=f"ModelScope error: {response_data['errors']}",
            )

        task_id = response_data.get("task_id")
        if not task_id:
            raise ModelScopeError(
                status_code=500,
                message="No task_id in ModelScope submit response",
            )

        polling_url = self.config.get_task_status_url(api_base, task_id)
        polling_headers = self.config.get_polling_headers(headers)

        start_time = time.time()
        verbose_logger.debug(f"ModelScope starting sync polling at {polling_url}")

        while time.time() - start_time < max_wait:
            response = sync_client.get(
                url=polling_url,
                headers=polling_headers,
                timeout=timeout,
            )

            if response.status_code >= 400:
                raise ModelScopeError(
                    status_code=response.status_code,
                    message=f"Polling failed: {response.text}",
                )

            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise ModelScopeError(
                    status_code=response.status_code,
                    message=f"Error parsing poll response: {e}",
                )
            status = data.get("task_status")
            verbose_logger.debug(f"ModelScope poll status: {status}")

            if status == TASK_STATUS_SUCCEED:
                return response
            elif status == TASK_STATUS_FAILED:
                raise ModelScopeError(
                    status_code=400,
                    message="ModelScope image generation task failed",
                )

            time.sleep(interval)

        raise ModelScopeError(
            status_code=408,
            message=f"Polling timed out after {max_wait} seconds",
        )

    async def _poll_for_result_async(
        self,
        initial_response: httpx.Response,
        api_base: str | None,
        headers: dict,
        async_client: AsyncHTTPHandler,
        max_wait: float = DEFAULT_MAX_POLLING_TIME,
        interval: float = DEFAULT_POLLING_INTERVAL,
        timeout: float | httpx.Timeout | None = None,
    ) -> httpx.Response:
        if initial_response.status_code >= 400:
            raise ModelScopeError(
                status_code=initial_response.status_code,
                message=f"ModelScope submit failed: {initial_response.text}",
            )

        try:
            response_data = initial_response.json()
        except json.JSONDecodeError as e:
            raise ModelScopeError(
                status_code=initial_response.status_code,
                message=f"Error parsing submit response: {e}",
            )

        if "errors" in response_data:
            raise ModelScopeError(
                status_code=initial_response.status_code,
                message=f"ModelScope error: {response_data['errors']}",
            )

        task_id = response_data.get("task_id")
        if not task_id:
            raise ModelScopeError(
                status_code=500,
                message="No task_id in ModelScope submit response",
            )

        polling_url = self.config.get_task_status_url(api_base, task_id)
        polling_headers = self.config.get_polling_headers(headers)

        start_time = time.time()
        verbose_logger.debug(f"ModelScope starting async polling at {polling_url}")

        while time.time() - start_time < max_wait:
            response = await async_client.get(
                url=polling_url,
                headers=polling_headers,
                timeout=timeout,
            )

            if response.status_code >= 400:
                raise ModelScopeError(
                    status_code=response.status_code,
                    message=f"Polling failed: {response.text}",
                )

            try:
                data = response.json()
            except json.JSONDecodeError as e:
                raise ModelScopeError(
                    status_code=response.status_code,
                    message=f"Error parsing poll response: {e}",
                )
            status = data.get("task_status")
            verbose_logger.debug(f"ModelScope poll status: {status}")

            if status == TASK_STATUS_SUCCEED:
                return response
            elif status == TASK_STATUS_FAILED:
                raise ModelScopeError(
                    status_code=400,
                    message="ModelScope image generation task failed",
                )

            await asyncio.sleep(interval)

        raise ModelScopeError(
            status_code=408,
            message=f"Polling timed out after {max_wait} seconds",
        )


# Singleton instance for use in images/main.py
modelscope_image_generation = ModelScopeImageGeneration()
