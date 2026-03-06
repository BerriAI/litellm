import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import httpx

from litellm._logging import verbose_logger
from litellm.constants import (
    RUNWAYML_DEFAULT_API_VERSION,
    RUNWAYML_POLLING_TIMEOUT,
)
from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class RunwayMLImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for RunwayML image generation models.
    """
    DEFAULT_BASE_URL: str = "https://api.dev.runwayml.com"
    IMAGE_GENERATION_ENDPOINT: str = "v1/text_to_image"

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        complete_url: str = (
            api_base 
            or get_secret_str("RUNWAYML_API_BASE") 
            or self.DEFAULT_BASE_URL
        )

        complete_url = complete_url.rstrip("/")
        if self.IMAGE_GENERATION_ENDPOINT:
            complete_url = f"{complete_url}/{self.IMAGE_GENERATION_ENDPOINT}"
        return complete_url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        final_api_key: Optional[str] = (
            api_key or 
            get_secret_str("RUNWAYML_API_SECRET") or
            get_secret_str("RUNWAYML_API_KEY")
        )
        if not final_api_key:
            raise ValueError("RUNWAYML_API_SECRET or RUNWAYML_API_KEY is not set")
        
        headers["Authorization"] = f"Bearer {final_api_key}"    
        headers["X-Runway-Version"] = RUNWAYML_DEFAULT_API_VERSION
        return headers

    @staticmethod
    def _transform_runwayml_response_to_openai(
        response_data: Dict[str, Any],
        model_response: ImageResponse,
    ) -> ImageResponse:
        """
        Transform RunwayML response format to OpenAI ImageResponse format.
        
        RunwayML response format (after polling):
        {
            "id": "task_123...",
            "status": "SUCCEEDED",
            "output": ["https://cloudfront.net/.../image.png"],
            "completedAt": "2025-11-13T..."
        }
        
        OpenAI ImageResponse format:
        {
            "data": [
                {
                    "url": "https://cloudfront.net/.../image.png",
                    "b64_json": null
                }
            ]
        }
        
        Args:
            response_data: JSON response from RunwayML (after polling completes)
            model_response: ImageResponse object to populate
            
        Returns:
            Populated ImageResponse in OpenAI format
        """
        if not model_response.data:
            model_response.data = []
        
        # Handle RunwayML response format
        # Response contains task.output with image URL(s)
        output = response_data.get("output", [])
        
        if isinstance(output, list):
            for image_item in output:
                if isinstance(image_item, str):
                    # If output is a list of URL strings
                    model_response.data.append(ImageObject(
                        url=image_item,
                        b64_json=None,
                    ))
                elif isinstance(image_item, dict):
                    # If output contains dict with url/b64_json
                    model_response.data.append(ImageObject(
                        url=image_item.get("url", None),
                        b64_json=image_item.get("b64_json", None),
                    ))
        
        return model_response

    @staticmethod
    def _check_timeout(start_time: float, timeout_secs: float) -> None:
        """
        Check if operation has timed out.
        
        Args:
            start_time: Start time of the operation
            timeout_secs: Timeout duration in seconds
            
        Raises:
            TimeoutError: If operation has exceeded timeout
        """
        if time.time() - start_time > timeout_secs:
            raise TimeoutError(
                f"RunwayML task polling timed out after {timeout_secs} seconds"
            )

    @staticmethod
    def _check_task_status(response_data: Dict[str, Any]) -> str:
        """
        Check RunwayML task status from response.
        
        RunwayML statuses: PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED, THROTTLED
        
        Args:
            response_data: JSON response from RunwayML task endpoint
            
        Returns:
            Normalized status string: "running", "succeeded", or raises on failure
            
        Raises:
            ValueError: If task failed or status is unknown
        """
        status = response_data.get("status", "").upper()
        
        verbose_logger.debug(f"RunwayML task status: {status}")
        
        if status == "SUCCEEDED":
            return "succeeded"
        elif status == "FAILED":
            failure_reason = response_data.get("failure", "Unknown error")
            failure_code = response_data.get("failureCode", "unknown")
            raise ValueError(
                f"RunwayML image generation failed: {failure_reason} (code: {failure_code})"
            )
        elif status == "CANCELLED":
            raise ValueError("RunwayML image generation was cancelled")
        elif status in ["PENDING", "RUNNING", "THROTTLED"]:
            return "running"
        else:
            raise ValueError(f"Unknown RunwayML task status: {status}")

    def _poll_task_sync(
        self,
        task_id: str,
        api_base: str,
        headers: Dict[str, str],
        timeout_secs: float = 600,
    ) -> httpx.Response:
        """
        Poll RunwayML task until completion (sync).
        
        RunwayML POST returns immediately with a task that has status PENDING/RUNNING.
        We need to poll GET /v1/tasks/{task_id} until status is SUCCEEDED or FAILED.
        
        Args:
            task_id: The task ID to poll
            api_base: Base URL for RunwayML API
            headers: Request headers (including auth)
            timeout_secs: Total timeout in seconds (default: 600s = 10 minutes)
            
        Returns:
            Final response with completed task
        """
        from litellm.llms.custom_httpx.http_handler import _get_httpx_client

        client = _get_httpx_client()
        start_time = time.time()
        
        # Build task status URL
        api_base = api_base.rstrip("/")
        task_url = f"{api_base}/v1/tasks/{task_id}"
        
        verbose_logger.debug(f"Polling RunwayML task: {task_url}")
        
        while True:
            self._check_timeout(start_time=start_time, timeout_secs=timeout_secs)
            
            # Poll the task status
            response = client.get(url=task_url, headers=headers)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Check task status
            status = self._check_task_status(response_data=response_data)
            
            if status == "succeeded":
                return response
            elif status == "running":
                # Wait before polling again (RunwayML recommends 1-2 second intervals)
                time.sleep(2)

    async def _poll_task_async(
        self,
        task_id: str,
        api_base: str,
        headers: Dict[str, str],
        timeout_secs: float = 600,
    ) -> httpx.Response:
        """
        Poll RunwayML task until completion (async).
        
        Args:
            task_id: The task ID to poll
            api_base: Base URL for RunwayML API
            headers: Request headers (including auth)
            timeout_secs: Total timeout in seconds (default: 600s = 10 minutes)
            
        Returns:
            Final response with completed task
        """
        import litellm
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        client = get_async_httpx_client(llm_provider=litellm.LlmProviders.RUNWAYML)
        start_time = time.time()
        
        # Build task status URL
        api_base = api_base.rstrip("/")
        task_url = f"{api_base}/v1/tasks/{task_id}"
        
        verbose_logger.debug(f"Polling RunwayML task (async): {task_url}")
        
        while True:
            self._check_timeout(start_time=start_time, timeout_secs=timeout_secs)
            
            # Poll the task status
            response = await client.get(url=task_url, headers=headers)
            response.raise_for_status()
            
            response_data = response.json()
            
            # Check task status
            status = self._check_task_status(response_data=response_data)
            
            if status == "succeeded":
                return response
            elif status == "running":
                # Wait before polling again (RunwayML recommends 1-2 second intervals)
                await asyncio.sleep(2)

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        """
        Transform the image generation response to the litellm image response.
        
        RunwayML returns a task immediately with status PENDING/RUNNING.
        We need to poll the task until it completes (status SUCCEEDED).
        
        Initial response:
        {
            "id": "task_123...",
            "status": "PENDING" | "RUNNING",
            "createdAt": "2025-11-13T..."
        }
        
        After polling:
        {
            "id": "task_123...",
            "status": "SUCCEEDED",
            "output": ["https://cloudfront.net/.../image.png"],
            "completedAt": "2025-11-13T..."
        }
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        

        verbose_logger.debug(
            "RunwayML starting polling..."
        )
        
        # Get task ID
        task_id = response_data.get("id")
        if not task_id:
            raise ValueError("RunwayML response missing task ID")
                
        # Get headers for polling (need auth)
        poll_headers = {
            "Authorization": raw_response.request.headers.get("Authorization", ""),
            "X-Runway-Version": raw_response.request.headers.get("X-Runway-Version", RUNWAYML_DEFAULT_API_VERSION),
        }
        
        # Poll until task completes
        raw_response = self._poll_task_sync(
            task_id=task_id,
            api_base=self.DEFAULT_BASE_URL,
            headers=poll_headers,
            timeout_secs=RUNWAYML_POLLING_TIMEOUT,
        )
        
        # Update response_data with polled result
        response_data = raw_response.json()
        
        verbose_logger.debug("RunwayML polling complete, transforming to OpenAI format")
        
        # Transform RunwayML response to OpenAI format
        return self._transform_runwayml_response_to_openai(
            response_data=response_data,
            model_response=model_response,
        )

    async def async_transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        """
        Async transform the image generation response to the litellm image response.
        
        RunwayML returns a task immediately with status PENDING/RUNNING.
        We need to poll the task until it completes (status SUCCEEDED) using async polling.
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        
        verbose_logger.debug(
            "RunwayML starting polling (async)..."
        )
        
        # Get task ID
        task_id = response_data.get("id")
        if not task_id:
            raise ValueError("RunwayML response missing task ID")
        
        # Get headers for polling (need auth)
        poll_headers = {
            "Authorization": raw_response.request.headers.get("Authorization", ""),
            "X-Runway-Version": raw_response.request.headers.get("X-Runway-Version", RUNWAYML_DEFAULT_API_VERSION),
        }
        
        # Poll until task completes (async)
        raw_response = await self._poll_task_async(
            task_id=task_id,
            api_base=self.DEFAULT_BASE_URL,
            headers=poll_headers,
            timeout_secs=RUNWAYML_POLLING_TIMEOUT,
        )
        
        # Update response_data with polled result
        response_data = raw_response.json()
        
        verbose_logger.debug("RunwayML polling complete (async), transforming to OpenAI format")
        
        # Transform RunwayML response to OpenAI format
        return self._transform_runwayml_response_to_openai(
            response_data=response_data,
            model_response=model_response,
        )
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Get supported OpenAI parameters for RunwayML image generation
        """
        return [
            "size",
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        
        # Map OpenAI 'size' parameter to RunwayML 'ratio' parameter
        if "size" in non_default_params:
            size = non_default_params["size"]
            # Map common OpenAI sizes to RunwayML ratios
            size_to_ratio_map = {
                "1024x1024": "1024:1024",
                "1792x1024": "1792:1024",
                "1024x1792": "1024:1792",
                "1920x1080": "1920:1080",
                "1080x1920": "1080:1920",
            }
            optional_params["ratio"] = size_to_ratio_map.get(size, "1920:1080")
        
        for k in non_default_params.keys():
            if k not in optional_params.keys():
                if k in supported_params:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to the RunwayML image generation request body
        
        RunwayML expects:
        - model: The model to use (e.g., 'gen4_image')
        - promptText: The text prompt
        - ratio: The aspect ratio (e.g., '1920:1080', '1080:1920', '1024:1024')
        """
        runwayml_request_body = {
            "model": model or "gen4_image",
            "promptText": prompt,
        }
        
        # Add any RunwayML-specific parameters
        if "ratio" in optional_params:
            runwayml_request_body["ratio"] = optional_params["ratio"]
        else:
            # Set default ratio if not provided
            runwayml_request_body["ratio"] = "1920:1080"

        
        # Add any other optional parameters
        for k, v in optional_params.items():
            if k not in runwayml_request_body and k not in ["size"]:
                runwayml_request_body[k] = v
        
        return runwayml_request_body

