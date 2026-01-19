"""
Black Forest Labs Image Edit Handler

Handles image edit requests for Black Forest Labs models.
BFL uses an async polling pattern - the initial request returns a task ID,
then we poll until the result is ready.
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
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import FileTypes, ImageResponse

from ..common_utils import (
    DEFAULT_MAX_POLLING_TIME,
    DEFAULT_POLLING_INTERVAL,
    BlackForestLabsError,
)
from .transformation import BlackForestLabsImageEditConfig


class BlackForestLabsImageEdit:
    """
    Black Forest Labs Image Edit handler.

    Handles the HTTP requests and polling logic, delegating data transformation
    to the BlackForestLabsImageEditConfig class.
    """

    def __init__(self):
        self.config = BlackForestLabsImageEditConfig()

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
        """
        Main entry point for image edit requests.

        Args:
            model: The model to use (e.g., "black_forest_labs/flux-kontext-pro")
            image: The image(s) to edit
            prompt: The edit instruction
            image_edit_optional_request_params: Optional parameters for the request
            litellm_params: LiteLLM parameters including api_key, api_base
            logging_obj: Logging object
            timeout: Request timeout
            extra_headers: Additional headers
            client: HTTP client to use
            aimage_edit: If True, return async coroutine

        Returns:
            ImageResponse or coroutine if aimage_edit=True
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

        # Sync version
        if client is None or not isinstance(client, HTTPHandler):
            sync_client = _get_httpx_client()
        else:
            sync_client = client

        # Validate environment and get headers
        headers = self.config.validate_environment(
            api_key=api_key,
            headers=image_edit_optional_request_params.get("extra_headers", {}) or {},
            model=model,
        )
        if extra_headers:
            headers.update(extra_headers)

        # Get complete URL
        complete_url = self.config.get_complete_url(
            model=model,
            api_base=api_base,
            litellm_params=litellm_params_dict,
        )

        # Transform request
        # Handle image list vs single image
        image_input = image[0] if isinstance(image, list) and image else image
        data, _ = self.config.transform_image_edit_request(
            model=model,
            prompt=prompt or "",
            image=image_input,
            image_edit_optional_request_params=image_edit_optional_request_params,
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
        """
        Async version of image edit.
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
            headers=image_edit_optional_request_params.get("extra_headers", {}) or {},
            model=model,
        )
        if extra_headers:
            headers.update(extra_headers)

        # Get complete URL
        api_base = self.config.get_complete_url(
            model=model,
            api_base=api_base,
            litellm_params=litellm_params_dict,
        )

        # Transform request
        image_input = image[0] if isinstance(image, list) and image else image
        data, _ = self.config.transform_image_edit_request(
            model=model,
            prompt=prompt or "",
            image=image_input,
            image_edit_optional_request_params=image_edit_optional_request_params,
            litellm_params=litellm_params_dict,
            headers=headers,
        )

        # Logging
        logging_obj.pre_call(
            input=prompt,
            api_key="",
            additional_args={
                "complete_input_dict": data,
                "api_base": api_base,
                "headers": headers,
            },
        )

        # Make initial request
        try:
            response = await async_client.post(
                url=api_base,
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
        return self.config.transform_image_edit_response(
            model=model,
            raw_response=final_response,
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

        Args:
            initial_response: The initial response containing polling_url
            headers: Headers to use for polling (must include x-key)
            sync_client: HTTP client
            max_wait: Maximum time to wait in seconds
            interval: Polling interval in seconds

        Returns:
            Final response with completed result
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
bfl_image_edit = BlackForestLabsImageEdit()
