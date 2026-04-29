"""
ModelsLab video generation transformation for LiteLLM.

NOTE: ModelsLab uses key-in-body authentication. The MODELSLAB_API_KEY
will appear in the request body (not headers). LiteLLM's logging pipeline
may log this — treat the key accordingly.
"""
import asyncio
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

MODELSLAB_VIDEO_BASE_URL = "https://modelslab.com/api/v6/video"
MODELSLAB_POLL_INTERVAL_SECONDS = 5
MODELSLAB_POLL_TIMEOUT_SECONDS = 300


class ModelsLabVideoConfig(BaseVideoConfig):
    """
    Configuration class for ModelsLab video generation.

    ModelsLab uses an async pattern:
    1. POST /api/v6/video/text2video (or img2video) creates a job
    2. Response is either {status: success, output: [...]} or {status: processing, request_id: ...}
    3. When processing, poll POST /api/v6/video/fetch/{request_id} with {key} body until done
    """

    def __init__(self):
        super().__init__()
        self._api_key: Optional[str] = None

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped_params: Dict[str, Any] = {}

        # Parse size "WxH" → width, height
        if "size" in video_create_optional_params:
            size = video_create_optional_params["size"]
            if isinstance(size, str) and "x" in size:
                try:
                    w, h = size.split("x", 1)
                    mapped_params["width"] = int(w)
                    mapped_params["height"] = int(h)
                except (ValueError, TypeError):
                    pass

        # input_reference → init_image (for img2video)
        if "input_reference" in video_create_optional_params:
            mapped_params["init_image"] = video_create_optional_params["input_reference"]

        # seconds → num_frames (approximate at ~8fps default)
        if "seconds" in video_create_optional_params:
            seconds = video_create_optional_params["seconds"]
            try:
                mapped_params["num_frames"] = max(8, int(float(str(seconds))) * 8)
            except (ValueError, TypeError):
                pass

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        """
        Validate environment. ModelsLab uses key-in-body — only Content-Type goes in headers.
        """
        # Priority: explicit api_key arg > litellm_params.api_key > MODELSLAB_API_KEY > litellm.api_key
        # Provider-specific env var takes precedence over litellm.api_key (which is commonly
        # set to an OpenAI key globally and should not be forwarded to ModelsLab).
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = (
            api_key
            or get_secret_str("MODELSLAB_API_KEY")
            or litellm.api_key
        )

        if not api_key:
            raise ValueError(
                "ModelsLab API key is required. Set MODELSLAB_API_KEY environment variable "
                "or pass api_key parameter."
            )

        self._api_key = api_key
        # Key-in-body: DO NOT set Authorization header
        headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        if api_base:
            return api_base.rstrip("/")
        # Default to text2video; endpoint selection is handled in 
        # transform_video_create_request based on mapped params
        return f"{MODELSLAB_VIDEO_BASE_URL}/text2video"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        request_data: Dict[str, Any] = {
            "key": self._api_key,
            "model_id": model,
            "prompt": prompt,
        }
        request_data.update(video_create_optional_request_params)

        # Route to img2video endpoint if init_image is present
        if "init_image" in video_create_optional_request_params:
            api_base = f"{MODELSLAB_VIDEO_BASE_URL}/img2video"

        files_list: List[Tuple[str, Any]] = []
        return request_data, files_list, api_base

    def _build_video_object(
        self,
        response_data: Dict,
        request_id: str,
        model: str,
        raw_response_status_code: int,
        raw_response_headers: dict,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        """
        Shared result-building logic for sync and async video transform paths.
        Both paths share identical success/error handling after polling completes.
        """
        status = response_data.get("status", "")

        if status == "error":
            raise BaseLLMException(
                status_code=500,
                message=response_data.get("message", "ModelsLab video generation failed"),
                headers={},
            )

        if status == "success":
            output = response_data.get("output", [])
            output_url = output[0] if output else None
            video_obj = VideoObject(
                id=request_id,
                object="video",
                status="completed",
                created_at=int(time.time()),
            )  # type: ignore[arg-type]
            if output_url:
                video_obj._hidden_params["output_url"] = output_url
            if custom_llm_provider and video_obj.id:
                video_obj.id = encode_video_id_with_provider(
                    video_obj.id, custom_llm_provider, model
                )
            return video_obj

        raise BaseLLMException(
            status_code=raw_response_status_code,
            message=f"Unexpected ModelsLab video status: {status}",
            headers=raw_response_headers,
        )

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        status = response_data.get("status", "")
        # ModelsLab may return request_id, id, or neither on immediate success.
        # Fall back to a timestamp-based placeholder so VideoObject.id is never empty.
        request_id = str(
            response_data.get("request_id")
            or response_data.get("id")
            or f"modelslab-{int(time.time())}"
        )

        if status == "error":
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=response_data.get("message", "ModelsLab video generation failed"),
                headers=dict(raw_response.headers),
            )

        if status == "processing":
            # Poll until done
            response_data = self._poll_sync(request_id)

        return self._build_video_object(
            response_data=response_data,
            request_id=request_id,
            model=model,
            raw_response_status_code=raw_response.status_code,
            raw_response_headers=dict(raw_response.headers),
            custom_llm_provider=custom_llm_provider,
        )

    def _poll_sync(
        self,
        request_id: str,
        timeout: int = MODELSLAB_POLL_TIMEOUT_SECONDS,
        interval: int = MODELSLAB_POLL_INTERVAL_SECONDS,
    ) -> Dict:
        """Poll the ModelsLab fetch endpoint until status is success or error (sync path)."""
        fetch_url = f"{MODELSLAB_VIDEO_BASE_URL}/fetch/{request_id}"
        body = {"key": self._api_key}
        client: HTTPHandler = _get_httpx_client()
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                resp = client.post(fetch_url, json=body)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise BaseLLMException(
                    status_code=e.response.status_code,
                    message=f"ModelsLab poll request failed: {e.response.text}",
                    headers=dict(e.response.headers),
                ) from e
            except httpx.RequestError as e:
                # Wrap network-level errors (timeout, DNS failure, connection reset)
                # so they flow through LiteLLM's standard error pipeline
                raise BaseLLMException(
                    status_code=503,
                    message=f"ModelsLab poll network error: {e}",
                    headers={},
                ) from e
            data = resp.json()
            status = data.get("status", "")
            if status in ("success", "error"):
                return data
            # Still processing — sleep before next poll (not before the first attempt)
            time.sleep(interval)

        raise BaseLLMException(
            status_code=408,
            message=f"ModelsLab video generation timed out after {timeout}s (request_id={request_id})",
            headers={},
        )

    async def _poll_async(
        self,
        request_id: str,
        timeout: int = MODELSLAB_POLL_TIMEOUT_SECONDS,
        interval: int = MODELSLAB_POLL_INTERVAL_SECONDS,
    ) -> Dict:
        """Poll the ModelsLab fetch endpoint using asyncio.sleep (async path)."""
        fetch_url = f"{MODELSLAB_VIDEO_BASE_URL}/fetch/{request_id}"
        body = {"key": self._api_key}
        async_client = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.MODELSLAB,
        )
        deadline = time.time() + timeout

        while time.time() < deadline:
            try:
                resp = await async_client.post(fetch_url, json=body)
                resp.raise_for_status()
            except httpx.HTTPStatusError as e:
                raise BaseLLMException(
                    status_code=e.response.status_code,
                    message=f"ModelsLab poll request failed: {e.response.text}",
                    headers=dict(e.response.headers),
                ) from e
            except httpx.RequestError as e:
                # Wrap network-level errors so they flow through LiteLLM's error pipeline
                raise BaseLLMException(
                    status_code=503,
                    message=f"ModelsLab async poll network error: {e}",
                    headers={},
                ) from e
            data = resp.json()
            status = data.get("status", "")
            if status in ("success", "error"):
                return data
            # Still processing — sleep before next poll (not before the first attempt)
            await asyncio.sleep(interval)

        raise BaseLLMException(
            status_code=408,
            message=f"ModelsLab video generation timed out after {timeout}s (request_id={request_id})",
            headers={},
        )

    async def async_transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        """
        Async version of transform_video_create_response.
        Uses asyncio.sleep in the poll loop instead of time.sleep, so it does
        not block the event loop during long-running video generation jobs.
        Shared result-building is delegated to _build_video_object to avoid duplication.
        """
        response_data = raw_response.json()
        status = response_data.get("status", "")
        request_id = str(
            response_data.get("request_id")
            or response_data.get("id")
            or f"modelslab-{int(time.time())}"
        )

        if status == "error":
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=response_data.get("message", "ModelsLab video generation failed"),
                headers=dict(raw_response.headers),
            )

        if status == "processing":
            response_data = await self._poll_async(request_id)

        return self._build_video_object(
            response_data=response_data,
            request_id=request_id,
            model=model,
            raw_response_status_code=raw_response.status_code,
            raw_response_headers=dict(raw_response.headers),
            custom_llm_provider=custom_llm_provider,
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_id = extract_original_video_id(video_id)
        url = f"{MODELSLAB_VIDEO_BASE_URL}/fetch/{original_id}"
        body = {"key": self._api_key}
        return url, body

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        status = response_data.get("status", "")
        request_id = str(response_data.get("request_id", ""))

        status_map = {
            "processing": "in_progress",
            "success": "completed",
            "error": "failed",
        }
        mapped_status = status_map.get(status, "queued")

        output = response_data.get("output", [])
        output_url = output[0] if output else None

        video_obj = VideoObject(
            id=request_id,
            object="video",
            status=mapped_status,
            created_at=int(time.time()),
        )  # type: ignore[arg-type]
        if output_url:
            video_obj._hidden_params["output_url"] = output_url

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, None
            )

        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video content download not supported by ModelsLab API")

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        raise NotImplementedError("Video content download not supported by ModelsLab API")

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        raise NotImplementedError("Video content download not supported by ModelsLab API")

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video remix not supported by ModelsLab API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix not supported by ModelsLab API")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video listing not supported by ModelsLab API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video listing not supported by ModelsLab API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video deletion not supported by ModelsLab API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video deletion not supported by ModelsLab API")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Union[dict, httpx.Headers],
    ) -> BaseLLMException:
        # Return the exception object — callers use `raise provider.get_error_class(...)`
        return BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
