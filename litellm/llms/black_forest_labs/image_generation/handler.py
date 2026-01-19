"""
Black Forest Labs Image Generation Handler

Handles image generation requests for Black Forest Labs models.
BFL uses an async polling pattern - the initial request returns a task ID,
then we poll until the result is ready.
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
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse

from ..common_utils import (
    DEFAULT_MAX_POLLING_TIME,
    DEFAULT_POLLING_INTERVAL,
    BlackForestLabsError,
)
from .transformation import BlackForestLabsImageGenerationConfig


class BlackForestLabsImageGeneration:
    """
    Black Forest Labs Image Generation handler.

    Handles the HTTP requests and polling logic, delegating data transformation
    to the BlackForestLabsImageGenerationConfig class.
    """

    def __init__(self):
        self.config = BlackForestLabsImageGenerationConfig()

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
        """
        Main entry point for image generation requests.

        Args:
            model: The model to use (e.g., "black_forest_labs/flux-pro-1.1")
            prompt: The text prompt for image generation
            model_response: ImageResponse object to populate
            optional_params: Optional parameters for the request
            litellm_params: LiteLLM parameters including api_key, api_base
            logging_obj: Logging object
            timeout: Request timeout
            extra_headers: Additional headers
            client: HTTP client to use
            aimg_generation: If True, return async coroutine

        Returns:
            ImageResponse or coroutine if aimg_generation=True
        """
        # Handle litellm_params as dict or object
        if isinstance(litellm_params, dict):
            api_key = litellm_params.get("api_key")
            api_base = litellm_params.get("api_base")
            litellm_params_dict = litellm_params
        else:
            api_key = litellm_params.api_key
            api_base = litellm_params.api_base
            litellm_params_dict = dict(litellm_params)

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

        # Sync version
        if client is None or not isinstance(client, HTTPHandler):
            sync_client = _get_httpx_client()
        else:
            sync_client = client

        # Validate environment and get headers
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

        # Get complete URL
        complete_url = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )

        # Transform request
        data = self.config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            headers=headers,
        )

        # Logging
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": complete_url,
                "headers": headers,
            },
        )

        # Make initial request
        try:
            response = sync_client.post(
                url=complete_url,
                headers=headers,
                json=data,
                timeout=timeout,
            )
        except Exception as e:
            raise BlackForestLabsError(
                status_code=500,
                message=f"Request failed: {str(e)}",
            )

        # Poll for result
        final_response = self._poll_for_result_sync(
            initial_response=response,
            headers=headers,
            sync_client=sync_client,
        )

        # Transform response
        return self.config.transform_image_generation_response(
            model=model,
            raw_response=final_response,
            model_response=model_response,
            logging_obj=logging_obj,
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
        """
        Async version of image generation.
        """
        # Handle litellm_params as dict or object
        if isinstance(litellm_params, dict):
            api_key = litellm_params.get("api_key")
            api_base = litellm_params.get("api_base")
            litellm_params_dict = litellm_params
        else:
            api_key = litellm_params.api_key
            api_base = litellm_params.api_base
            litellm_params_dict = dict(litellm_params)

        if client is None:
            async_client = get_async_httpx_client(
                llm_provider=litellm.LlmProviders.BLACK_FOREST_LABS,
            )
        else:
            async_client = client

        # Validate environment and get headers
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

        # Get complete URL
        complete_url = self.config.get_complete_url(
            api_base=api_base,
            api_key=api_key,
            model=model,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
        )

        # Transform request
        data = self.config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=litellm_params_dict,
            headers=headers,
        )

        # Logging
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": complete_url,
                "headers": headers,
            },
        )

        # Make initial request
        try:
            response = await async_client.post(
                url=complete_url,
                headers=headers,
                json=data,
                timeout=timeout,
            )
        except Exception as e:
            raise BlackForestLabsError(
                status_code=500,
                message=f"Request failed: {str(e)}",
            )

        # Poll for result
        final_response = await self._poll_for_result_async(
            initial_response=response,
            headers=headers,
            async_client=async_client,
        )

        # Transform response
        return self.config.transform_image_generation_response(
            model=model,
            raw_response=final_response,
            model_response=model_response,
            logging_obj=logging_obj,
        )

    def _poll_for_result_sync(
        self,
        initial_response: httpx.Response,
        headers: dict,
        sync_client: HTTPHandler,
        max_wait: float = DEFAULT_MAX_POLLING_TIME,
        interval: float = DEFAULT_POLLING_INTERVAL,
    ) -> httpx.Response:
        """
        Poll BFL API until result is ready (sync version).
        """
        # Parse initial response to get polling URL
        try:
            response_data = initial_response.json()
        except Exception as e:
            raise BlackForestLabsError(
                status_code=initial_response.status_code,
                message=f"Error parsing initial response: {e}",
            )

        # Check for immediate errors
        if "errors" in response_data:
            raise BlackForestLabsError(
                status_code=initial_response.status_code,
                message=f"BFL error: {response_data['errors']}",
            )

        polling_url = response_data.get("polling_url")
        if not polling_url:
            raise BlackForestLabsError(
                status_code=500,
                message="No polling_url in BFL response",
            )

        # Get just the auth header for polling
        polling_headers = {"x-key": headers.get("x-key", "")}

        start_time = time.time()
        verbose_logger.debug(f"BFL starting sync polling at {polling_url}")

        while time.time() - start_time < max_wait:
            response = sync_client.get(
                url=polling_url,
                headers=polling_headers,
            )

            if response.status_code != 200:
                raise BlackForestLabsError(
                    status_code=response.status_code,
                    message=f"Polling failed: {response.text}",
                )

            data = response.json()
            status = data.get("status")

            verbose_logger.debug(f"BFL poll status: {status}")

            if status == "Ready":
                return response
            elif status in ["Error", "Failed", "Content Moderated", "Request Moderated"]:
                raise BlackForestLabsError(
                    status_code=400,
                    message=f"Image generation failed: {status}",
                )

            time.sleep(interval)

        raise BlackForestLabsError(
            status_code=408,
            message=f"Polling timed out after {max_wait} seconds",
        )

    async def _poll_for_result_async(
        self,
        initial_response: httpx.Response,
        headers: dict,
        async_client: AsyncHTTPHandler,
        max_wait: float = DEFAULT_MAX_POLLING_TIME,
        interval: float = DEFAULT_POLLING_INTERVAL,
    ) -> httpx.Response:
        """
        Poll BFL API until result is ready (async version).
        """
        # Parse initial response to get polling URL
        try:
            response_data = initial_response.json()
        except Exception as e:
            raise BlackForestLabsError(
                status_code=initial_response.status_code,
                message=f"Error parsing initial response: {e}",
            )

        # Check for immediate errors
        if "errors" in response_data:
            raise BlackForestLabsError(
                status_code=initial_response.status_code,
                message=f"BFL error: {response_data['errors']}",
            )

        polling_url = response_data.get("polling_url")
        if not polling_url:
            raise BlackForestLabsError(
                status_code=500,
                message="No polling_url in BFL response",
            )

        # Get just the auth header for polling
        polling_headers = {"x-key": headers.get("x-key", "")}

        start_time = time.time()
        verbose_logger.debug(f"BFL starting async polling at {polling_url}")

        while time.time() - start_time < max_wait:
            response = await async_client.get(
                url=polling_url,
                headers=polling_headers,
            )

            if response.status_code != 200:
                raise BlackForestLabsError(
                    status_code=response.status_code,
                    message=f"Polling failed: {response.text}",
                )

            data = response.json()
            status = data.get("status")

            verbose_logger.debug(f"BFL poll status: {status}")

            if status == "Ready":
                return response
            elif status in ["Error", "Failed", "Content Moderated", "Request Moderated"]:
                raise BlackForestLabsError(
                    status_code=400,
                    message=f"Image generation failed: {status}",
                )

            await asyncio.sleep(interval)

        raise BlackForestLabsError(
            status_code=408,
            message=f"Polling timed out after {max_wait} seconds",
        )


# Singleton instance for use in images/main.py
bfl_image_generation = BlackForestLabsImageGeneration()
