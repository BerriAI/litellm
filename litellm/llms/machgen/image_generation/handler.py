"""
MachGen image generation handler.

Submits a text-to-image task, polls it to completion, and returns an OpenAI shaped
`ImageResponse`. With `response_format="b64_json"` the asset is downloaded with the
caller's MachGen key and inlined, since MachGen asset URLs require authentication.
"""

from __future__ import annotations

import asyncio
import base64
import time
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import TYPE_CHECKING
from urllib.parse import quote

import httpx

import litellm
from litellm._logging import verbose_logger
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import (
    DEFAULT_MAX_POLLING_TIME,
    DEFAULT_POLLING_INTERVAL,
    STATUS_COMPLETED,
    STATUS_FAILED,
    TASKS_PATH,
    MachGenError,
)
from .transformation import MachGenImageGenerationConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


@dataclass(frozen=True, slots=True)
class PreparedRequest:
    url: str
    headers: dict[str, str]
    body: dict[str, object]
    api_base: str


class MachGenImageGeneration:
    def __init__(
        self,
        config: MachGenImageGenerationConfig | None = None,
        polling_interval: float = DEFAULT_POLLING_INTERVAL,
        max_polling_time: float = DEFAULT_MAX_POLLING_TIME,
    ) -> None:
        self.config = config or MachGenImageGenerationConfig()
        self.polling_interval = polling_interval
        self.max_polling_time = max_polling_time

    def image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: dict,
        litellm_params: GenericLiteLLMParams | dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict | None = None,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        aimg_generation: bool = False,
    ) -> ImageResponse | Coroutine[None, None, ImageResponse]:
        if aimg_generation:
            return self.async_image_generation(
                model=model,
                prompt=prompt,
                model_response=model_response,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                timeout=timeout,
                api_key=api_key,
                api_base=api_base,
                extra_headers=extra_headers,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )

        sync_client = client if isinstance(client, HTTPHandler) else _get_httpx_client()
        request = self._prepare_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            extra_headers=extra_headers,
        )

        submit_response = sync_client.post(
            url=request.url,
            headers=request.headers,
            json=request.body,
            timeout=timeout,
        )
        task_url = self._task_url(request, submit_response)

        deadline = time.time() + self.max_polling_time
        while True:
            task_response = sync_client.get(url=task_url, headers=request.headers)
            if self._is_terminal(task_response):
                break
            if time.time() >= deadline:
                raise self._timeout_error()
            time.sleep(self.polling_interval)

        response = self._transform_response(
            model=model,
            task_response=task_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request=request,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        if self._wants_b64(optional_params):
            asset = sync_client.get(url=self.config.get_asset_url(task_response), headers=request.headers)
            return self._inline_asset(response, asset)
        return response

    async def async_image_generation(
        self,
        model: str,
        prompt: str,
        model_response: ImageResponse,
        optional_params: dict,
        litellm_params: GenericLiteLLMParams | dict,
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        api_key: str | None = None,
        api_base: str | None = None,
        extra_headers: dict | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ImageResponse:
        async_client = client or get_async_httpx_client(llm_provider=litellm.LlmProviders.MACHGEN)
        request = self._prepare_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params,
            logging_obj=logging_obj,
            api_key=api_key,
            api_base=api_base,
            extra_headers=extra_headers,
        )

        submit_response = await async_client.post(
            url=request.url,
            headers=request.headers,
            json=request.body,
            timeout=timeout,
        )
        task_url = self._task_url(request, submit_response)

        deadline = time.time() + self.max_polling_time
        while True:
            task_response = await async_client.get(url=task_url, headers=request.headers)
            if self._is_terminal(task_response):
                break
            if time.time() >= deadline:
                raise self._timeout_error()
            await asyncio.sleep(self.polling_interval)

        response = self._transform_response(
            model=model,
            task_response=task_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request=request,
            optional_params=optional_params,
            litellm_params=litellm_params,
        )

        if self._wants_b64(optional_params):
            asset = await async_client.get(url=self.config.get_asset_url(task_response), headers=request.headers)
            return self._inline_asset(response, asset)
        return response

    def _prepare_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: GenericLiteLLMParams | dict,
        logging_obj: LiteLLMLoggingObj,
        api_key: str | None,
        api_base: str | None,
        extra_headers: dict | None,
    ) -> PreparedRequest:
        litellm_params_dict = litellm_params if isinstance(litellm_params, dict) else dict(litellm_params)
        resolved_api_key = api_key or litellm_params_dict.get("api_key")
        resolved_api_base = self.config.get_api_base(api_base or litellm_params_dict.get("api_base"))

        headers = self.config.validate_environment(
            headers=dict(extra_headers or {}),
            model=model,
            messages=[],
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            api_key=resolved_api_key,
            api_base=resolved_api_base,
        )
        body = self.config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            headers=headers,
        )
        url = self.config.get_complete_url(
            api_base=resolved_api_base,
            api_key=resolved_api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )

        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={"complete_input_dict": body, "api_base": url, "headers": headers},
        )
        return PreparedRequest(url=url, headers=headers, body=body, api_base=resolved_api_base)

    def _task_url(self, request: PreparedRequest, submit_response: httpx.Response) -> str:
        if submit_response.status_code >= 400:
            raise MachGenError(
                status_code=submit_response.status_code,
                message=f"MachGen task submission failed: {submit_response.text}",
            )

        try:
            task_id = submit_response.json().get("task_id")
        except ValueError as e:
            raise MachGenError(
                status_code=submit_response.status_code,
                message=f"Error parsing MachGen submit response: {e}",
            ) from e

        if not task_id:
            raise MachGenError(status_code=500, message="No task_id in MachGen submit response")

        verbose_logger.debug("MachGen polling task %s", task_id)
        return f"{request.api_base}{TASKS_PATH}/{quote(str(task_id), safe='')}"

    def _timeout_error(self) -> MachGenError:
        return MachGenError(
            status_code=408,
            message=f"MachGen task polling timed out after {self.max_polling_time} seconds",
        )

    @staticmethod
    def _is_terminal(task_response: httpx.Response) -> bool:
        if task_response.status_code >= 400:
            raise MachGenError(
                status_code=task_response.status_code,
                message=f"MachGen task polling failed: {task_response.text}",
            )
        return task_response.json().get("status") in (STATUS_COMPLETED, STATUS_FAILED)

    def _transform_response(
        self,
        model: str,
        task_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request: PreparedRequest,
        optional_params: dict,
        litellm_params: GenericLiteLLMParams | dict,
    ) -> ImageResponse:
        return self.config.transform_image_generation_response(
            model=model,
            raw_response=task_response,
            model_response=model_response,
            logging_obj=logging_obj,
            request_data=dict(request.body),
            optional_params=optional_params,
            litellm_params=litellm_params if isinstance(litellm_params, dict) else dict(litellm_params),
            encoding=None,
        )

    @staticmethod
    def _wants_b64(optional_params: dict) -> bool:
        return optional_params.get("response_format") == "b64_json"

    @staticmethod
    def _inline_asset(response: ImageResponse, asset: httpx.Response) -> ImageResponse:
        if asset.status_code >= 400:
            raise MachGenError(
                status_code=asset.status_code,
                message=f"MachGen asset download failed: {asset.text}",
            )
        response.data = [ImageObject(b64_json=base64.b64encode(asset.content).decode("utf-8"))]
        return response


machgen_image_generation = MachGenImageGeneration()
