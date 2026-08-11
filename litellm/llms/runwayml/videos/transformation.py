from datetime import datetime
from typing import TYPE_CHECKING, Any, Final

import httpx
from httpx._types import FileContent, RequestFiles
from pydantic import BaseModel, ConfigDict, Field

import litellm
from litellm.constants import RUNWAYML_DEFAULT_API_VERSION
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
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


class RunwayMLTaskResponse(BaseModel):
    """RunwayML task object returned by ``/v1/image_to_video`` and ``/v1/tasks/{id}``."""

    model_config = ConfigDict(populate_by_name=True)

    id: str = ""
    status: str = "pending"
    created_at: str | None = Field(default=None, alias="createdAt")
    completed_at: str | None = Field(default=None, alias="completedAt")
    output: tuple[str, ...] | str | None = None
    progress: int | None = None
    failure: str | None = None
    failure_code: str | None = Field(default=None, alias="failureCode")


class RunwayMLTaskRequest(BaseModel):
    """The RunwayML-shaped request fields this module reads back when building a ``VideoObject``."""

    model: str | None = None
    ratio: str | None = None
    duration: int | None = None


class RunwayMLVideoConfig(BaseVideoConfig):
    """
    Configuration class for RunwayML video generation.

    RunwayML uses a task-based API where:
    1. POST /v1/image_to_video creates a task
    2. The task returns immediately with a task ID
    3. Client must poll or wait for task completion
    """

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported OpenAI parameters for video generation.
        Maps OpenAI params to RunwayML equivalents:
        - prompt -> promptText
        - input_reference -> promptImage
        - size -> ratio (e.g., "1280x720" -> "1280:720")
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
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to RunwayML format.

        Mappings:
        - prompt -> promptText
        - input_reference -> promptImage
        - size -> ratio (convert "WIDTHxHEIGHT" to "WIDTH:HEIGHT")
        - seconds -> duration (convert to integer)
        """
        mapped_params: Final[dict[str, object]] = {}

        # Handle input_reference parameter - map to promptImage
        if "input_reference" in video_create_optional_params:
            input_reference: Final = video_create_optional_params["input_reference"]
            # RunwayML supports URLs and data URIs directly
            mapped_params["promptImage"] = input_reference

        # Handle size parameter - convert "1280x720" to "1280:720"
        if "size" in video_create_optional_params:
            size: Final = video_create_optional_params["size"]
            if isinstance(size, str) and "x" in size:
                mapped_params["ratio"] = size.replace("x", ":")

        # Handle seconds parameter - convert to integer
        if "seconds" in video_create_optional_params:
            seconds: Final = video_create_optional_params["seconds"]
            if seconds is not None:
                try:
                    mapped_params["duration"] = int(float(seconds)) if isinstance(seconds, str) else int(seconds)
                except (ValueError, TypeError):
                    # If conversion fails, use default duration
                    pass

        # Pass through other parameters that aren't OpenAI-specific
        supported_openai_params: Final = self.get_supported_openai_params(model)
        for key, value in video_create_optional_params.items():
            if key not in supported_openai_params:
                mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        """
        Validate environment and set up authentication headers.
        RunwayML uses Bearer token authentication via RUNWAYML_API_SECRET.
        """
        # Use api_key from litellm_params if available, otherwise fall back to other sources
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = (
            api_key or litellm.api_key or get_secret_str("RUNWAYML_API_SECRET") or get_secret_str("RUNWAYML_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "RunwayML API key is required. Set RUNWAYML_API_SECRET environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "X-Runway-Version": RUNWAYML_DEFAULT_API_VERSION,
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Get the base URL for RunwayML API.
        The specific endpoint path will be added in the transform methods.
        """
        if api_base is None:
            api_base = "https://api.dev.runwayml.com/v1"

        return api_base.rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        """
        Transform the video creation request for RunwayML API.

        RunwayML expects:
        {
            "model": "gen4_turbo",
            "promptImage": "https://... or data:image/...",
            "promptText": "description",
            "ratio": "1280:720",
            "duration": 5
        }
        """
        # Build the request data
        request_data: Final[dict[str, object]] = {
            "model": model,
            "promptText": prompt,
        }

        # Add mapped parameters
        request_data.update(video_create_optional_request_params)

        # RunwayML uses JSON body, no files multipart
        files_list: Final[RequestFiles] = []

        # Append the specific endpoint for video generation
        full_api_base: Final = f"{api_base}/image_to_video"

        return request_data, files_list, full_api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        """
        Transform the RunwayML video creation response.

        RunwayML returns a task object that looks like:
        {
            "id": "task_123...",
            "status": "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED",
            "output": ["https://...video.mp4"] (when succeeded)
        }

        We map this to OpenAI VideoObject format.
        """
        task: Final = RunwayMLTaskResponse.model_validate(raw_response.json())
        request_params: Final = RunwayMLTaskRequest.model_validate(request_data or {})
        seconds: Final = str(request_params.duration) if request_params.duration is not None else None

        return VideoObject(
            id=self._encoded_video_id(task.id, custom_llm_provider, model),
            object="video",
            status=self._map_runway_status(task.status),
            created_at=self._parse_runway_timestamp(task.created_at),
            completed_at=self._completed_at(task),
            error=self._build_error(task),
            model=request_params.model,
            size=self._ratio_to_size(request_params.ratio),
            seconds=seconds,
            usage=self._usage_from_seconds(seconds),
        )

    @staticmethod
    def _encoded_video_id(video_id: str, custom_llm_provider: str | None, model: str | None) -> str:
        if not (custom_llm_provider and video_id):
            return video_id
        return encode_video_id_with_provider(video_id, custom_llm_provider, model)

    @staticmethod
    def _ratio_to_size(ratio: str | None) -> str | None:
        if not ratio or ":" not in ratio:
            return None
        return ratio.replace(":", "x")

    @staticmethod
    def _usage_from_seconds(seconds: str | None) -> dict[str, float]:
        if not seconds:
            return {}
        try:
            return {"duration_seconds": float(seconds)}
        except ValueError:
            return {}

    def _completed_at(self, task: RunwayMLTaskResponse) -> int | None:
        if "completed_at" not in task.model_fields_set:
            return None
        return self._parse_runway_timestamp(task.completed_at)

    @staticmethod
    def _build_error(task: RunwayMLTaskResponse) -> dict[str, str] | None:
        if not {"failure", "failure_code"} & task.model_fields_set:
            return None
        return {
            "code": task.failure_code if task.failure_code is not None else "unknown",
            "message": task.failure if task.failure is not None else "Video generation failed",
        }

    def _map_runway_status(self, runway_status: str) -> str:
        """
        Map RunwayML status to OpenAI status format.

        RunwayML statuses: PENDING, RUNNING, SUCCEEDED, FAILED, CANCELLED
        OpenAI statuses: queued, in_progress, completed, failed
        """
        status_map: Final = {
            "PENDING": "queued",
            "RUNNING": "in_progress",
            "SUCCEEDED": "completed",
            "FAILED": "failed",
            "CANCELLED": "failed",
            "THROTTLED": "queued",
        }
        return status_map.get(runway_status.upper(), "queued")

    def _parse_runway_timestamp(self, timestamp_str: str | None) -> int:
        """
        Convert RunwayML ISO 8601 timestamp to Unix timestamp.

        RunwayML returns timestamps like: "2025-11-11T21:48:50.448Z"
        We need to convert to Unix timestamp (seconds since epoch).
        """
        if not timestamp_str:
            return 0

        try:
            # Parse ISO 8601 timestamp
            dt: Final = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
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
        variant: str | None = None,
    ) -> tuple[str, dict]:
        """
        Transform the video content request for RunwayML API.

        RunwayML doesn't have a separate content download endpoint.
        The video URL is returned in the task output field.
        We'll retrieve the task and extract the video URL.
        """
        original_video_id: Final = extract_original_video_id(video_id)
        encoded_video_id: Final = encode_url_path_segment(original_video_id, field_name="video_id")

        # Get task status to retrieve video URL
        url: Final = f"{api_base}/tasks/{encoded_video_id}"

        params: Final[dict[str, str]] = {}

        return url, params

    def _extract_video_url_from_response(self, task: RunwayMLTaskResponse) -> str:
        """
        Helper method to extract video URL from RunwayML response.
        Shared between sync and async transforms.
        """
        video_url: Final = self._first_output_url(task.output)
        if video_url:
            return video_url

        status: Final = task.status
        if status in ("PENDING", "RUNNING", "THROTTLED"):
            raise ValueError(f"Video is still processing (status: {status}). Please wait and try again.")
        if status == "FAILED":
            failure_reason: Final = task.failure if task.failure is not None else "Unknown error"
            raise ValueError(f"Video generation failed: {failure_reason}")
        raise ValueError("Video URL not found in response. Video may not be ready yet.")

    @staticmethod
    def _first_output_url(output: tuple[str, ...] | str | None) -> str | None:
        if isinstance(output, tuple):
            return output[0] if output else None
        return output

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """
        Transform the RunwayML video content download response (synchronous).

        RunwayML's task endpoint returns JSON with a video URL in the output field.
        We need to extract the URL and download the video.

        Example response:
        {
            "id":"63fd0f13-f29d-4e58-99d3-1cb9efa14a5b",
            "createdAt":"2025-11-11T21:48:50.448Z",
            "status":"SUCCEEDED",
            "output":["https://dnznrvs05pmza.cloudfront.net/.../video.mp4?_jwt=..."]
        }
        """
        task: Final = RunwayMLTaskResponse.model_validate(raw_response.json())
        video_url: Final = self._extract_video_url_from_response(task)

        # Download the video from the CloudFront URL synchronously
        httpx_client: Final[HTTPHandler] = _get_httpx_client()
        video_response: Final = httpx_client.get(video_url)
        video_response.raise_for_status()

        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """
        Transform the RunwayML video content download response (asynchronous).

        RunwayML's task endpoint returns JSON with a video URL in the output field.
        We need to extract the URL and download the video asynchronously.

        Example response:
        {
            "id":"63fd0f13-f29d-4e58-99d3-1cb9efa14a5b",
            "createdAt":"2025-11-11T21:48:50.448Z",
            "status":"SUCCEEDED",
            "output":["https://dnznrvs05pmza.cloudfront.net/.../video.mp4?_jwt=..."]
        }
        """
        task: Final = RunwayMLTaskResponse.model_validate(raw_response.json())
        video_url: Final = self._extract_video_url_from_response(task)

        # Download the video from the CloudFront URL asynchronously
        async_httpx_client: Final[AsyncHTTPHandler] = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.RUNWAYML,
        )
        video_response: Final = await async_httpx_client.get(video_url)
        video_response.raise_for_status()

        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: dict[str, object] | None = None,
    ) -> tuple[str, dict]:
        """
        Transform the video remix request for RunwayML API.

        RunwayML doesn't have a direct remix endpoint in their current API.
        This would need to be implemented when/if they add this feature.
        """
        raise NotImplementedError("Video remix is not yet supported by RunwayML API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        """Transform the RunwayML video remix response."""
        raise NotImplementedError("Video remix is not yet supported by RunwayML API")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, object] | None = None,
    ) -> tuple[str, dict]:
        """
        Transform the video list request for RunwayML API.

        RunwayML doesn't expose a list endpoint in their public API yet.
        """
        raise NotImplementedError("Video listing is not yet supported by RunwayML API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        """Transform the RunwayML video list response."""
        raise NotImplementedError("Video listing is not yet supported by RunwayML API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """
        Transform the video delete request for RunwayML API.

        RunwayML uses task cancellation.
        """
        original_video_id: Final = extract_original_video_id(video_id)
        encoded_video_id: Final = encode_url_path_segment(original_video_id, field_name="video_id")

        # Construct the URL for task cancellation
        url: Final = f"{api_base}/tasks/{encoded_video_id}/cancel"

        data: Final[dict[str, str]] = {}

        return url, data

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """Transform the RunwayML video delete/cancel response."""
        task: Final = RunwayMLTaskResponse.model_validate(raw_response.json())

        return VideoObject(
            id=task.id,
            object="video",
            status="cancelled",
            created_at=self._parse_runway_timestamp(task.created_at),
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """
        Transform the RunwayML video status retrieve request.

        RunwayML uses GET /v1/tasks/{task_id} to retrieve task status.
        """
        original_video_id: Final = extract_original_video_id(video_id)
        encoded_video_id: Final = encode_url_path_segment(original_video_id, field_name="video_id")

        # Construct the full URL for task status retrieval
        url: Final = f"{api_base}/tasks/{encoded_video_id}"

        # Empty dict for GET request (no body)
        data: Final[dict[str, str]] = {}

        return url, data

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        """
        Transform the RunwayML video status retrieve response.
        """
        task: Final = RunwayMLTaskResponse.model_validate(raw_response.json())

        return VideoObject(
            id=self._encoded_video_id(task.id, custom_llm_provider, None),
            object="video",
            status=self._map_runway_status(task.status),
            created_at=self._parse_runway_timestamp(task.created_at),
            completed_at=self._completed_at(task),
            progress=task.progress,
            error=self._build_error(task),
        )

    def transform_video_create_character_request(self, name, video: FileContent, api_base, litellm_params, headers):
        raise NotImplementedError("video create character is not supported for RunwayML")

    def transform_video_create_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video create character is not supported for RunwayML")

    def transform_video_get_character_request(self, character_id, api_base, litellm_params, headers):
        raise NotImplementedError("video get character is not supported for RunwayML")

    def transform_video_get_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video get character is not supported for RunwayML")

    def transform_video_edit_request(
        self,
        prompt,
        video_id,
        api_base,
        litellm_params,
        headers,
        extra_body=None,
        prefetched_source_data=None,
    ):
        raise NotImplementedError("video edit is not supported for RunwayML")

    def transform_video_edit_response(
        self,
        raw_response,
        logging_obj,
        custom_llm_provider=None,
        request_data=None,
    ):
        raise NotImplementedError("video edit is not supported for RunwayML")

    def transform_video_extension_request(
        self,
        prompt,
        video_id,
        seconds,
        api_base,
        litellm_params,
        headers,
        extra_body=None,
    ):
        raise NotImplementedError("video extension is not supported for RunwayML")

    def transform_video_extension_response(self, raw_response, logging_obj, custom_llm_provider=None):
        raise NotImplementedError("video extension is not supported for RunwayML")

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
