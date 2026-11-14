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
from collections.abc import Coroutine, Mapping
from typing import Final

import httpx
from pydantic import ConfigDict, JsonValue, TypeAdapter

import litellm
from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # litellm client factory is untyped
    get_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # litellm client factory is untyped
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

_JSON_RESPONSE: Final = TypeAdapter(dict[str, JsonValue], config=ConfigDict(hide_input_in_errors=True))


def _str_or_none(value: JsonValue) -> str | None:
    return value if isinstance(value, str) else None


def _extract_params(
    litellm_params: GenericLiteLLMParams | dict[str, JsonValue],  # mutable-ok: litellm passes a mutable dict
) -> tuple[str | None, str | None, dict[str, JsonValue]]:  # mutable-ok: returns the caller's dict unchanged
    if isinstance(litellm_params, dict):
        return (
            _str_or_none(litellm_params.get("api_key")),
            _str_or_none(litellm_params.get("api_base")),
            litellm_params,
        )
    return litellm_params.api_key, litellm_params.api_base, dict(litellm_params)


class ModelScopeImageGeneration:
    """ModelScope image generation handler."""

    def __init__(self) -> None:
        self.config: Final = ModelScopeImageGenerationConfig()

    def image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: dict[str, JsonValue],  # mutable-ok: litellm override signature
        litellm_params: GenericLiteLLMParams | dict[str, JsonValue],  # mutable-ok: litellm override signature
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        extra_headers: Mapping[str, str] | None = None,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        aimg_generation: bool = False,
    ) -> ImageResponse | Coroutine[object, object, ImageResponse]:
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

        api_key, api_base, litellm_params_dict = _extract_params(litellm_params)

        sync_client: Final = client if isinstance(client, HTTPHandler) else get_httpx_client()  # pyright: ignore[reportUnknownVariableType]  # litellm client factory is untyped

        headers: Final = self.config.validate_environment(
            api_key=api_key,
            headers={},
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )
        if extra_headers:
            headers.update(extra_headers)

        complete_url: Final = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )

        data: Final = self.config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            headers=headers,
        )

        logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # litellm logging API is untyped
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": complete_url,
                "headers": headers,
            },
        )

        try:
            response: Final = sync_client.post(  # pyright: ignore[reportUnknownMemberType]  # untyped client post
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

        final_response: Final = self._poll_for_result_sync(
            initial_response=response,  # pyright: ignore[reportArgumentType]  # post() returns Response | None; None raises HTTPError above
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
        optional_params: dict[str, JsonValue],  # mutable-ok: litellm override signature
        litellm_params: GenericLiteLLMParams | dict[str, JsonValue],  # mutable-ok: litellm override signature
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        extra_headers: Mapping[str, str] | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ImageResponse:
        api_key, api_base, litellm_params_dict = _extract_params(litellm_params)

        async_client: Final = client or get_async_httpx_client(  # pyright: ignore[reportUnknownVariableType]  # litellm client factory is untyped
            llm_provider=litellm.LlmProviders.MODELSCOPE,
        )

        headers: Final = self.config.validate_environment(
            api_key=api_key,
            headers={},
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )
        if extra_headers:
            headers.update(extra_headers)

        complete_url: Final = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )

        data: Final = self.config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            headers=headers,
        )

        logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # litellm logging API is untyped
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": complete_url,
                "headers": headers,
            },
        )

        try:
            response: Final = await async_client.post(  # pyright: ignore[reportUnknownMemberType]  # untyped client post
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

        final_response: Final = await self._poll_for_result_async(
            initial_response=response,  # pyright: ignore[reportArgumentType]  # post() returns Response | None; None raises HTTPError above
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
        headers: dict[str, str],  # mutable-ok: caller passes a mutable dict
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
            response_data: Final = _JSON_RESPONSE.validate_python(initial_response.json())
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

        task_id: Final[str | None] = _str_or_none(response_data.get("task_id"))
        if not task_id:
            raise ModelScopeError(
                status_code=500,
                message="No task_id in ModelScope submit response",
            )

        polling_url: Final = self.config.get_task_status_url(api_base, task_id)
        polling_headers: Final = self.config.get_polling_headers(headers)

        start_time: Final = time.time()
        verbose_logger.debug("ModelScope starting sync polling at %s", polling_url)

        while time.time() - start_time < max_wait:
            response = sync_client.get(  # pyright: ignore[reportUnknownMemberType]  # untyped client get
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
                data = _JSON_RESPONSE.validate_python(response.json())
            except json.JSONDecodeError as e:
                raise ModelScopeError(
                    status_code=response.status_code,
                    message=f"Error parsing poll response: {e}",
                )
            status = data.get("task_status")
            verbose_logger.debug("ModelScope poll status: %s", status)

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
        headers: dict[str, str],  # mutable-ok: caller passes a mutable dict
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
            response_data: Final = _JSON_RESPONSE.validate_python(initial_response.json())
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

        task_id: Final[str | None] = _str_or_none(response_data.get("task_id"))
        if not task_id:
            raise ModelScopeError(
                status_code=500,
                message="No task_id in ModelScope submit response",
            )

        polling_url: Final = self.config.get_task_status_url(api_base, task_id)
        polling_headers: Final = self.config.get_polling_headers(headers)

        start_time: Final = time.time()
        verbose_logger.debug("ModelScope starting async polling at %s", polling_url)

        while time.time() - start_time < max_wait:
            response = await async_client.get(  # pyright: ignore[reportUnknownMemberType]  # untyped client get
                url=polling_url,
                headers=polling_headers,  # pyright: ignore[reportArgumentType]  # untyped client headers param
                timeout=timeout,
            )

            if response.status_code >= 400:
                raise ModelScopeError(
                    status_code=response.status_code,
                    message=f"Polling failed: {response.text}",
                )

            try:
                data = _JSON_RESPONSE.validate_python(response.json())
            except json.JSONDecodeError as e:
                raise ModelScopeError(
                    status_code=response.status_code,
                    message=f"Error parsing poll response: {e}",
                )
            status = data.get("task_status")
            verbose_logger.debug("ModelScope poll status: %s", status)

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


modelscope_image_generation: Final = ModelScopeImageGeneration()
