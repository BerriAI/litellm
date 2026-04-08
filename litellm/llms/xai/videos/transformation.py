"""
xAI (Grok Imagine) video generation configuration for LiteLLM.

xAI uses an async task-based API:
1. POST /v1/videos/generations creates a generation request
2. Returns a request_id immediately
3. GET /v1/videos/{request_id} polls for completion
4. Response includes video URL when status is "done"

API docs: https://docs.x.ai/developers/rest-api-reference/inference/videos
"""

from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.constants import XAI_API_BASE
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
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


_XAI_STATUS_MAP = {
    "pending": "in_progress",
    "done": "completed",
    "failed": "failed",
    "expired": "failed",
}

_SIZE_TO_ASPECT_RATIO = {
    (16, 9): "16:9",
    (9, 16): "9:16",
    (1, 1): "1:1",
    (4, 3): "4:3",
    (3, 4): "3:4",
    (3, 2): "3:2",
    (2, 3): "2:3",
}

_ASPECT_RATIO_TO_SIZE = {
    "16:9": "1280x720",
    "9:16": "720x1280",
    "1:1": "720x720",
    "4:3": "960x720",
    "3:4": "720x960",
    "3:2": "1080x720",
    "2:3": "720x1080",
}


def _size_to_aspect_ratio(size: str) -> Optional[str]:
    """Convert "WIDTHxHEIGHT" to the closest xAI aspect ratio string."""
    try:
        w, h = size.lower().split("x")
        width, height = int(w), int(h)
    except (ValueError, AttributeError):
        return None

    if width == 0 or height == 0:
        return None

    actual_ratio = width / height
    best_match = None
    best_diff = float("inf")
    for (rw, rh), label in _SIZE_TO_ASPECT_RATIO.items():
        diff = abs(actual_ratio - rw / rh)
        if diff < best_diff:
            best_diff = diff
            best_match = label

    return best_match


class XAIVideoConfig(BaseVideoConfig):
    """
    Configuration for xAI (Grok Imagine) video generation.

    xAI video API endpoints:
    - POST /v1/videos/generations  -> {request_id}
    - POST /v1/videos/edits        -> {request_id}
    - POST /v1/videos/extensions   -> {request_id}
    - GET  /v1/videos/{request_id} -> {status, video: {url, duration}, usage}
    """

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

        if "input_reference" in video_create_optional_params:
            input_reference = video_create_optional_params["input_reference"]
            if input_reference is not None:
                mapped_params["image"] = {"url": str(input_reference)}

        if "seconds" in video_create_optional_params:
            seconds = video_create_optional_params["seconds"]
            if seconds is not None:
                try:
                    mapped_params["duration"] = (
                        int(float(seconds))
                        if isinstance(seconds, str)
                        else int(seconds)
                    )
                except (ValueError, TypeError):
                    pass

        if "size" in video_create_optional_params:
            size = video_create_optional_params["size"]
            if isinstance(size, str) and "x" in size:
                aspect_ratio = _size_to_aspect_ratio(size)
                if aspect_ratio:
                    mapped_params["aspect_ratio"] = aspect_ratio

        # Pass through provider-specific parameters (e.g., reference_images)
        supported_openai_params = self.get_supported_openai_params(model)
        for key, value in video_create_optional_params.items():
            if key not in supported_openai_params:
                mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = api_key or litellm.api_key or get_secret_str("XAI_API_KEY")

        if api_key is None:
            raise ValueError(
                "xAI API key is required. Set XAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        if api_base is None:
            api_base = XAI_API_BASE
        return api_base.rstrip("/")

    # -- Create ---------------------------------------------------------------

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        request_data: Dict[str, Any] = {"prompt": prompt}
        if model:
            request_data["model"] = model
        request_data.update(video_create_optional_request_params)

        files_list: List[Tuple[str, Any]] = []
        url = f"{api_base}/videos/generations"
        return request_data, files_list, url

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        response_data = raw_response.json()

        video_data: Dict[str, Any] = {
            "id": response_data.get("request_id", ""),
            "object": "video",
            "status": "queued",
        }

        if model:
            video_data["model"] = model
        if request_data:
            if "duration" in request_data:
                video_data["seconds"] = str(request_data["duration"])
            if "aspect_ratio" in request_data:
                size = _ASPECT_RATIO_TO_SIZE.get(request_data["aspect_ratio"])
                if size:
                    video_data["size"] = size

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )

        return video_obj

    # -- Status ---------------------------------------------------------------

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/videos/{original_video_id}"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        response_data = raw_response.json()

        xai_status = response_data.get("status", "pending")
        status = _XAI_STATUS_MAP.get(xai_status, "in_progress")

        video_data: Dict[str, Any] = {
            "id": "",
            "object": "video",
            "status": status,
        }

        if "model" in response_data:
            video_data["model"] = response_data["model"]

        if "progress" in response_data:
            video_data["progress"] = response_data["progress"]

        video_info = response_data.get("video") or {}
        if "duration" in video_info:
            video_data["seconds"] = str(video_info["duration"])

        if "error" in response_data:
            error = response_data["error"]
            video_data["error"] = {
                "code": error.get("code", "unknown"),
                "message": error.get("message", "Video generation failed"),
            }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        usage = response_data.get("usage")
        if usage:
            video_obj.usage = usage

        return video_obj

    # -- Content download -----------------------------------------------------

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/videos/{original_video_id}"
        return url, {}

    def _extract_video_url(self, response_data: Dict[str, Any]) -> str:
        video_info = response_data.get("video") or {}
        video_url = video_info.get("url")

        if not video_url:
            status = response_data.get("status", "unknown")
            if status == "pending":
                raise ValueError(
                    "Video is still processing. Please wait and try again."
                )
            elif status == "failed":
                error = response_data.get("error", {})
                message = error.get("message", "Unknown error")
                raise ValueError(f"Video generation failed: {message}")
            else:
                raise ValueError(
                    "Video URL not found in response. Video may not be ready yet."
                )
        return video_url

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        video_url = self._extract_video_url(response_data)

        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        video_url = self._extract_video_url(response_data)

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.XAI,
        )
        video_response = await async_httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    # -- Edit -----------------------------------------------------------------

    def transform_video_edit_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/videos/edits"
        data: Dict[str, Any] = {
            "prompt": prompt,
            "video": {"url": original_video_id},
        }
        if extra_body:
            data.update(extra_body)
        return url, data

    def transform_video_edit_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        video_obj = VideoObject(
            id=response_data.get("request_id", ""),
            object="video",
            status="queued",
        )
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, ""
            )
        return video_obj

    # -- Extension ------------------------------------------------------------

    def transform_video_extension_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        seconds: Optional[str] = None,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        url = f"{api_base}/videos/extensions"
        data: Dict[str, Any] = {
            "prompt": prompt,
            "video": {"url": original_video_id},
        }
        if seconds is not None:
            try:
                data["duration"] = int(float(seconds))
            except (ValueError, TypeError):
                pass
        if extra_body:
            data.update(extra_body)
        return url, data

    def transform_video_extension_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        video_obj = VideoObject(
            id=response_data.get("request_id", ""),
            object="video",
            status="queued",
        )
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, ""
            )
        return video_obj

    # -- Unsupported operations -----------------------------------------------

    def transform_video_remix_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video remix is not supported by xAI API")

    def transform_video_remix_response(self, *args: Any, **kwargs: Any) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by xAI API")

    def transform_video_list_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video listing is not supported by xAI API")

    def transform_video_list_response(self, *args: Any, **kwargs: Any) -> Dict[str, str]:
        raise NotImplementedError("Video listing is not supported by xAI API")

    def transform_video_delete_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video deletion is not supported by xAI API")

    def transform_video_delete_response(self, *args: Any, **kwargs: Any) -> VideoObject:
        raise NotImplementedError("Video deletion is not supported by xAI API")

    def transform_video_create_character_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video character creation is not supported by xAI API")

    def transform_video_create_character_response(self, *args: Any, **kwargs: Any) -> VideoObject:
        raise NotImplementedError("Video character creation is not supported by xAI API")

    def transform_video_get_character_request(self, *args: Any, **kwargs: Any) -> Tuple[str, Dict]:
        raise NotImplementedError("Video character retrieval is not supported by xAI API")

    def transform_video_get_character_response(self, *args: Any, **kwargs: Any) -> VideoObject:
        raise NotImplementedError("Video character retrieval is not supported by xAI API")
