import time
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.constants import PIXVERSE_DEFAULT_API_VERSION
from litellm.llms.base_llm.chat.transformation import BaseLLMException
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


class PixverseVideoConfig(BaseVideoConfig):
    """
    Configuration class for Pixverse video generation.

    Pixverse uses a task-based API where:
    1. POST /v1/text-to-video, /v1/image-to-video, or /v1/video-to-video creates a task
    2. The task returns immediately with a task ID
    3. Client must poll or wait for task completion
    """

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported OpenAI parameters for video generation.
        Maps OpenAI params to Pixverse equivalents:
        - prompt -> prompt
        - input_reference -> image or video (auto-detected)
        - size -> resolution
        - seconds -> duration
        """
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "extra_body",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """
        Map OpenAI parameters to Pixverse format.

        Mappings:
        - prompt -> prompt
        - input_reference -> image or video (based on file type detection)
        - size -> aspect_ratio + quality (e.g., "1280x720" -> "16:9" + "720p")
        - seconds -> duration (convert to int)
        - model -> model (Pixverse model version like "v5.6")
        """
        mapped_params: Dict[str, Any] = {}

        # Handle input_reference parameter - map to image or video
        if "input_reference" in video_create_optional_params:
            input_reference = video_create_optional_params["input_reference"]
            # Store it as input_reference for now, we'll determine the endpoint
            # in transform_video_create_request
            mapped_params["input_reference"] = input_reference

        # Handle size parameter - Pixverse uses aspect_ratio + quality
        if "size" in video_create_optional_params:
            size = video_create_optional_params["size"]
            if isinstance(size, str):
                # Parse size like "1280x720" or "720x1280"
                aspect_ratio, quality = self._parse_size_to_pixverse_format(size)
                mapped_params["aspect_ratio"] = aspect_ratio
                mapped_params["quality"] = quality

        # Handle seconds parameter - convert to int
        if "seconds" in video_create_optional_params:
            seconds = video_create_optional_params["seconds"]
            if seconds is not None:
                try:
                    # Pixverse requires integer duration
                    mapped_params["duration"] = int(
                        float(seconds) if isinstance(seconds, str) else seconds
                    )
                except (ValueError, TypeError):
                    # If conversion fails, skip duration
                    pass

        # Handle model parameter - forward Pixverse model version
        if "model" in video_create_optional_params:
            model_version = video_create_optional_params["model"]
            # Only forward bare version strings like "v5.6", not LiteLLM routing keys
            if (
                model_version
                and isinstance(model_version, str)
                and "/" not in model_version
            ):
                mapped_params["model"] = model_version

        # Pass through extra_body parameters that might be Pixverse-specific
        if "extra_body" in video_create_optional_params:
            extra_body = video_create_optional_params["extra_body"]
            if extra_body and isinstance(extra_body, dict):
                # Merge extra_body params
                mapped_params.update(extra_body)

        # Pass through other parameters that aren't OpenAI-specific
        supported_openai_params = self.get_supported_openai_params(model)
        for key, value in video_create_optional_params.items():
            if key not in supported_openai_params and key != "extra_body":
                mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        """
        Validate environment and set up authentication headers.
        Pixverse uses Bearer token authentication via PIXVERSE_API_KEY.
        """
        # Use api_key from litellm_params if available, otherwise fall back to other sources
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("PIXVERSE_API_KEY")
            or get_secret_str("PIXVERSE_API_SECRET")
        )

        if api_key is None:
            raise ValueError(
                "Pixverse API key is required. Set PIXVERSE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        headers.update(
            {
                "API-KEY": api_key,
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
        """
        Get the base URL for Pixverse API.
        The specific endpoint path will be added in the transform methods.
        """
        if api_base is None:
            api_base = (
                f"https://app-api.pixverse.ai/openapi/{PIXVERSE_DEFAULT_API_VERSION}"
            )

        return api_base.rstrip("/")

    def _parse_size_to_pixverse_format(self, size: str) -> tuple[str, str]:
        """
        Parse OpenAI size format to Pixverse aspect_ratio and quality.

        Args:
            size: Size string like "1280x720", "720x1280", "1920x1080"

        Returns:
            tuple: (aspect_ratio, quality)
                aspect_ratio: "16:9", "9:16", etc.
                quality: "720p", "1080p", etc.

        Examples:
            "1280x720" -> ("16:9", "720p")
            "720x1280" -> ("9:16", "720p")
            "1920x1080" -> ("16:9", "1080p")
        """
        try:
            parts = size.split("x")
            if len(parts) == 2:
                width = int(parts[0])
                height = int(parts[1])

                # Determine aspect ratio
                if width == height:
                    # Square
                    aspect_ratio = "1:1"
                elif width > height:
                    # Landscape
                    aspect_ratio = "16:9"
                else:
                    # Portrait
                    aspect_ratio = "9:16"

                # Determine quality based on height
                if height >= 1080:
                    quality = "1080p"
                else:
                    quality = "720p"

                return aspect_ratio, quality
        except (ValueError, IndexError):
            pass

        # Default values
        return "16:9", "720p"

    def _pixverse_format_to_size(self, aspect_ratio: str, quality: str) -> str:
        """
        Convert Pixverse aspect_ratio and quality back to size format.

        Args:
            aspect_ratio: "16:9", "9:16", "1:1", etc.
            quality: "720p", "1080p", etc.

        Returns:
            Size string like "1280x720", "720x1280", "720x720", etc.
        """
        # Extract short edge from quality
        short_edge = 720
        if "1080" in quality:
            short_edge = 1080

        # Determine dimensions based on aspect ratio
        if aspect_ratio == "16:9":
            # Landscape: short edge is height
            return f"{int(short_edge * 16 / 9)}x{short_edge}"
        elif aspect_ratio == "9:16":
            # Portrait: short edge is width
            return f"{short_edge}x{int(short_edge * 16 / 9)}"
        elif aspect_ratio == "1:1":
            # Square: both edges are equal
            return f"{short_edge}x{short_edge}"
        else:
            # Default to 16:9 landscape
            return f"{int(short_edge * 16 / 9)}x{short_edge}"

    def _determine_endpoint(self, input_reference: Optional[str]) -> str:
        """
        Determine the appropriate Pixverse endpoint based on input_reference type.

        Returns:
            - "/video/text/generate" for text-only
            - "/video/image/generate" for image input
            - "/video/video/generate" for video input (fusion)
        """
        if not input_reference:
            return "/video/text/generate"

        # Check if it's a video file (common video extensions or content type indicators)
        input_lower = input_reference.lower()
        video_extensions = (".mp4", ".mov", ".avi", ".webm", ".mkv", ".flv")
        video_mimetypes = ("video/", "application/x-mpegurl")

        # Check data URI scheme
        if input_lower.startswith("data:"):
            if any(mime in input_lower for mime in video_mimetypes):
                return "/video/video/generate"
            else:
                return "/video/image/generate"

        # For URLs, parse and check the path extension only (ignore query params)
        try:
            from urllib.parse import urlparse

            parsed = urlparse(input_reference)
            path = parsed.path.lower()
            if any(path.endswith(ext) for ext in video_extensions):
                return "/video/video/generate"
        except Exception:
            # If parsing fails, fall back to simple check
            if any(ext in input_lower for ext in video_extensions):
                return "/video/video/generate"

        # Default to image-to-video for URLs without clear video indicators
        return "/video/image/generate"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        """
        Transform the video creation request for Pixverse API.

        Pixverse expects:
        {
            "prompt": "description",
            "image": "https://... or data:image/...",  (for image-to-video)
            "video": "https://... or data:video/...",  (for video-to-video)
            "resolution": "1280x720",
            "duration": 5.0
        }
        """
        # Build the request data
        request_data: Dict[str, Any] = {
            "prompt": prompt,
        }

        # Determine the endpoint based on input_reference
        input_reference = video_create_optional_request_params.pop(
            "input_reference", None
        )
        endpoint = self._determine_endpoint(input_reference)

        # Add input_reference to the appropriate field
        if input_reference:
            if endpoint == "/video/video/generate":
                request_data["video"] = input_reference
            else:  # /video/image/generate
                request_data["image"] = input_reference

        # Add mapped parameters (excluding input_reference which we already handled)
        for key, value in video_create_optional_request_params.items():
            if key not in ["extra_headers", "extra_body"]:
                request_data[key] = value

        # Pixverse uses JSON body, no files multipart
        files_list: List[Tuple[str, Any]] = []

        # Append the specific endpoint for video generation
        full_api_base = f"{api_base}{endpoint}"

        return request_data, files_list, full_api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        """
        Transform the Pixverse video creation response.

        Pixverse returns:
        {
            "ErrCode": 0,
            "ErrMsg": "success",
            "Resp": {
                "video_id": 391504480090022,
                "credits": 45
            }
        }

        We map this to OpenAI VideoObject format.
        """
        response_data = raw_response.json()

        # Check for error
        if response_data.get("ErrCode", 0) != 0:
            raise ValueError(
                f"Pixverse API error: {response_data.get('ErrMsg', 'Unknown error')}"
            )

        # Extract video_id from Resp
        resp_data = response_data.get("Resp", {})
        video_id = resp_data.get("video_id")

        if not video_id:
            raise ValueError("No video_id in Pixverse response")

        # Convert video_id to string for encoding
        video_id_str = str(video_id)

        # Map to VideoObject format
        video_data: Dict[str, Any] = {
            "id": video_id_str,
            "object": "video",
            "status": "queued",  # Video generation is queued/in progress
            "created_at": int(time.time()),  # Current timestamp
        }

        # Add model and size info if available from request
        if request_data:
            if "model" in request_data:
                video_data["model"] = request_data["model"]
            if "aspect_ratio" in request_data and "quality" in request_data:
                # Reconstruct size from aspect_ratio and quality
                video_data["size"] = self._pixverse_format_to_size(
                    request_data["aspect_ratio"], request_data["quality"]
                )
            if "duration" in request_data:
                video_data["seconds"] = str(request_data["duration"])

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )

        # Add usage data for cost tracking
        usage_data = {}
        if video_obj and hasattr(video_obj, "seconds") and video_obj.seconds:
            try:
                usage_data["duration_seconds"] = float(video_obj.seconds)
            except (ValueError, TypeError):
                pass
        video_obj.usage = usage_data

        return video_obj

    def _map_pixverse_status(self, pixverse_status: Union[str, int]) -> str:
        """
        Map Pixverse status to OpenAI status format.

        Pixverse uses numeric status codes:
        - 0: queued/pending
        - 1: completed
        - 2: failed
        - 3: processing/in_progress
        - 5: unknown/other

        OpenAI statuses: queued, in_progress, completed, failed
        """
        # Handle numeric status codes
        if isinstance(pixverse_status, int):
            status_map_int = {
                0: "queued",
                1: "completed",
                2: "failed",
                3: "in_progress",
                5: "in_progress",  # Processing/generating
            }
            return status_map_int.get(pixverse_status, "queued")

        # Handle string status codes (for backwards compatibility)
        status_map = {
            "pending": "queued",
            "processing": "in_progress",
            "completed": "completed",
            "failed": "failed",
            "cancelled": "failed",
        }
        return status_map.get(str(pixverse_status).lower(), "queued")

    def _parse_pixverse_timestamp(self, timestamp_str: Optional[str]) -> int:
        """
        Convert Pixverse ISO 8601 timestamp to Unix timestamp.

        Pixverse returns timestamps like: "2025-01-01T00:00:00Z"
        We need to convert to Unix timestamp (seconds since epoch).
        """
        if not timestamp_str:
            return 0

        try:
            # Parse ISO 8601 timestamp
            dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            # Convert to Unix timestamp
            return int(dt.timestamp())
        except (ValueError, AttributeError):
            return 0

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        """
        Transform the video content request for Pixverse API.

        Pixverse doesn't have a separate content download endpoint.
        The video URL is returned in the /video/result response.
        We'll retrieve the result and extract the video URL.
        """
        original_video_id = extract_original_video_id(video_id)

        # Get video result to retrieve video URL
        url = f"{api_base}/video/result/{original_video_id}"

        params: Dict[str, Any] = {}

        return url, params

    def _extract_video_url_from_response(self, response_data: Dict[str, Any]) -> str:
        """
        Helper method to extract video URL from Pixverse response.
        Shared between sync and async transforms.

        Response format:
        {
            "ErrCode": 0,
            "ErrMsg": "Success",
            "Resp": {
                "id": 391504857968062,
                "status": 1,  # 0=queued, 1=completed, 2=failed, 3=processing
                "url": "https://media.pixverse.ai/...",
                ...
            }
        }
        """
        # Check for API error
        if response_data.get("ErrCode", 0) != 0:
            raise ValueError(
                f"Pixverse API error: {response_data.get('ErrMsg', 'Unknown error')}"
            )

        # Extract data from Resp
        resp_data = response_data.get("Resp", {})

        # Extract video URL from the url field
        video_url = resp_data.get("url")

        if not video_url:
            # Check if the video generation failed or is still processing
            status = resp_data.get("status", 0)
            if status in [0, 3]:  # queued or processing
                raise ValueError(
                    f"Video is still processing (status: {status}). Please wait and try again."
                )
            elif status == 2:  # failed
                raise ValueError("Video generation failed")
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
        """
        Transform the Pixverse video content download response (synchronous).

        Pixverse's task endpoint returns JSON with a video URL in the video_url field.
        We need to extract the URL and download the video.

        Example response:
        {
            "task_id":"task_123...",
            "created_at":"2025-01-01T00:00:00Z",
            "status":"completed",
            "video_url":"https://cdn.pixverse.ai/.../video.mp4"
        }
        """
        response_data = raw_response.json()
        video_url = self._extract_video_url_from_response(response_data)

        # Download the video from the URL synchronously
        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()

        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """
        Transform the Pixverse video content download response (asynchronous).

        Pixverse's task endpoint returns JSON with a video URL in the video_url field.
        We need to extract the URL and download the video asynchronously.

        Example response:
        {
            "task_id":"task_123...",
            "created_at":"2025-01-01T00:00:00Z",
            "status":"completed",
            "video_url":"https://cdn.pixverse.ai/.../video.mp4"
        }
        """
        response_data = raw_response.json()
        video_url = self._extract_video_url_from_response(response_data)

        # Download the video from the URL asynchronously
        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.PIXVERSE,
        )
        video_response = await async_httpx_client.get(video_url)
        video_response.raise_for_status()

        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """
        Transform the video remix request for Pixverse API.

        Pixverse doesn't have a direct remix endpoint in their current API.
        This would need to be implemented when/if they add this feature.
        """
        raise NotImplementedError("Video remix is not yet supported by Pixverse API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        """Transform the Pixverse video remix response."""
        raise NotImplementedError("Video remix is not yet supported by Pixverse API")

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
        """
        Transform the video list request for Pixverse API.

        Pixverse doesn't expose a list endpoint in their public API yet.
        """
        raise NotImplementedError("Video listing is not yet supported by Pixverse API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        """Transform the Pixverse video list response."""
        raise NotImplementedError("Video listing is not yet supported by Pixverse API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video delete request for Pixverse API.

        Pixverse uses task cancellation.
        """
        original_video_id = extract_original_video_id(video_id)

        # Construct the URL for task cancellation
        url = f"{api_base}/video/tasks/{original_video_id}/cancel"

        data: Dict[str, Any] = {}

        return url, data

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """Transform the Pixverse video delete/cancel response."""
        response_data = raw_response.json()

        # Check for error
        if response_data.get("ErrCode", 0) != 0:
            raise ValueError(
                f"Pixverse API error: {response_data.get('ErrMsg', 'Unknown error')}"
            )

        # Extract data from Resp
        resp_data = response_data.get("Resp", {})

        video_obj = VideoObject(
            id=str(resp_data.get("id", resp_data.get("task_id", ""))),
            object="video",
            status="cancelled",
            created_at=self._parse_pixverse_timestamp(resp_data.get("created_at")),
        )  # type: ignore[arg-type]

        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the Pixverse video status retrieve request.

        Pixverse uses GET /openapi/v2/video/result/{video_id} to retrieve video status.
        """
        original_video_id = extract_original_video_id(video_id)

        # Construct the full URL for video status retrieval
        url = f"{api_base}/video/result/{original_video_id}"

        # Empty dict for GET request (no body)
        data: Dict[str, Any] = {}

        return url, data

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        """
        Transform the Pixverse video status retrieve response.

        Response format:
        {
            "ErrCode": 0,
            "ErrMsg": "Success",
            "Resp": {
                "id": 391504857968062,
                "status": 1,  # 0=queued, 1=completed, 2=failed, 3=processing
                "url": "https://media.pixverse.ai/...",
                "create_time": "2026-03-12T03:40:35Z",
                "modify_time": "2026-03-12T03:41:04Z",
                "outputWidth": 1280,
                "outputHeight": 720,
                ...
            }
        }
        """
        response_data = raw_response.json()

        # Check for error
        if response_data.get("ErrCode", 0) != 0:
            raise ValueError(
                f"Pixverse API error: {response_data.get('ErrMsg', 'Unknown error')}"
            )

        # Extract data from Resp
        resp_data = response_data.get("Resp", {})

        # Map to VideoObject format
        video_data: Dict[str, Any] = {
            "id": str(resp_data.get("id", "")),
            "object": "video",
            "status": self._map_pixverse_status(resp_data.get("status", 0)),
            "created_at": self._parse_pixverse_timestamp(resp_data.get("create_time")),
        }

        # Add optional fields if present
        if "url" in resp_data and resp_data["url"]:
            video_data["output_url"] = resp_data["url"]

        if "modify_time" in resp_data:
            video_data["completed_at"] = self._parse_pixverse_timestamp(
                resp_data.get("modify_time")
            )

        # Add size information
        if "outputWidth" in resp_data and "outputHeight" in resp_data:
            video_data[
                "size"
            ] = f"{resp_data['outputWidth']}x{resp_data['outputHeight']}"

        # Store video URL in hidden params for content download
        if "url" in resp_data:
            video_data["_hidden_params"] = {"video_url": resp_data["url"]}

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, None
            )

        return video_obj

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
