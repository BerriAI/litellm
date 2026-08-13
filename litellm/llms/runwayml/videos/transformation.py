from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import TYPE_CHECKING, Any, Final, Literal, TypedDict

import httpx
from httpx._types import FileTypes, RequestFiles
from typing_extensions import NotRequired, ReadOnly

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


class _RunwayTaskResponse(TypedDict, total=False):
    id: ReadOnly[str]
    status: ReadOnly[str]
    createdAt: ReadOnly[str]
    completedAt: ReadOnly[str]
    output: ReadOnly[Sequence[str] | str]
    progress: ReadOnly[int]
    failureCode: ReadOnly[str]
    failure: ReadOnly[str]


class _RunwayVideoData(TypedDict):
    id: ReadOnly[str]
    object: ReadOnly[Literal["video"]]
    status: ReadOnly[str]
    created_at: ReadOnly[int]
    output_url: ReadOnly[NotRequired[str]]
    completed_at: ReadOnly[NotRequired[int]]
    progress: ReadOnly[NotRequired[int]]
    error: ReadOnly[NotRequired[Mapping[str, str]]]
    model: ReadOnly[NotRequired[str]]
    size: ReadOnly[NotRequired[str]]
    seconds: ReadOnly[NotRequired[str]]


class _RunwayVideoOptionalData(TypedDict, total=False):
    output_url: ReadOnly[str]
    completed_at: ReadOnly[int]
    progress: ReadOnly[int]
    error: ReadOnly[Mapping[str, str]]
    model: ReadOnly[str]
    size: ReadOnly[str]
    seconds: ReadOnly[str]


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

    @staticmethod
    def _parse_task_response(raw_response: httpx.Response) -> _RunwayTaskResponse:
        return raw_response.json()

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
    ) -> dict[str, object]:
        """
        Map OpenAI parameters to RunwayML format.

        Mappings:
        - prompt -> promptText
        - input_reference -> promptImage
        - size -> ratio (convert "WIDTHxHEIGHT" to "WIDTH:HEIGHT")
        - seconds -> duration (convert to integer)
        """
        supported_openai_params: Final = self.get_supported_openai_params(model)
        return {
            **self._prompt_image_param(video_create_optional_params),
            **self._ratio_param(video_create_optional_params),
            **self._duration_param(video_create_optional_params),
            # Pass through other parameters that aren't OpenAI-specific
            **{key: value for key, value in video_create_optional_params.items() if key not in supported_openai_params},
        }

    @staticmethod
    def _prompt_image_param(video_create_optional_params: VideoCreateOptionalRequestParams) -> Mapping[str, object]:
        # Handle input_reference parameter - map to promptImage
        # RunwayML supports URLs and data URIs directly
        if "input_reference" in video_create_optional_params:
            return {"promptImage": video_create_optional_params["input_reference"]}
        return {}

    @staticmethod
    def _ratio_param(video_create_optional_params: VideoCreateOptionalRequestParams) -> Mapping[str, str]:
        # Handle size parameter - convert "1280x720" to "1280:720"
        if "size" in video_create_optional_params:
            size: Final = video_create_optional_params["size"]
            if isinstance(size, str) and "x" in size:
                return {"ratio": size.replace("x", ":")}
        return {}

    @staticmethod
    def _duration_param(video_create_optional_params: VideoCreateOptionalRequestParams) -> Mapping[str, int]:
        # Handle seconds parameter - convert to integer
        if "seconds" in video_create_optional_params:
            seconds: Final = video_create_optional_params["seconds"]
            if seconds is not None:
                try:
                    return {"duration": int(float(seconds)) if isinstance(seconds, str) else int(seconds)}
                except (ValueError, TypeError):
                    # If conversion fails, use default duration
                    pass
        return {}

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
        video_create_optional_request_params: dict[str, object],
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
        # Build the request data with the mapped parameters merged in
        request_data: Final = {
            "model": model,
            "promptText": prompt,
            **video_create_optional_request_params,
        }

        # RunwayML uses JSON body, no files multipart
        files_list: Final[Sequence[tuple[str, FileTypes]]] = []

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
        response_data: Final[_RunwayTaskResponse] = self._parse_task_response(raw_response)

        # RunwayML returns output as array of URLs when task succeeds
        output: Final = response_data.get("output")
        output_kwargs: Final = (
            _RunwayVideoOptionalData(output_url=output if isinstance(output, str) else output[0])
            if output
            else _RunwayVideoOptionalData()
        )
        completed_at_kwargs: Final = (
            _RunwayVideoOptionalData(completed_at=self._parse_runway_timestamp(response_data.get("completedAt")))
            if "completedAt" in response_data
            else _RunwayVideoOptionalData()
        )
        error_kwargs: Final = (
            _RunwayVideoOptionalData(
                error={
                    "code": response_data.get("failureCode", "unknown"),
                    "message": response_data.get("failure", "Video generation failed"),
                }
            )
            if "failureCode" in response_data or "failure" in response_data
            else _RunwayVideoOptionalData()
        )

        # Add model and size info if available from request
        model_kwargs: Final = (
            _RunwayVideoOptionalData(model=request_data["model"])
            if request_data and "model" in request_data
            else _RunwayVideoOptionalData()
        )
        # Convert ratio back to size format
        ratio: Final = request_data.get("ratio") if request_data else None
        size_kwargs: Final = (
            _RunwayVideoOptionalData(size=ratio.replace(":", "x"))
            if isinstance(ratio, str) and ":" in ratio
            else _RunwayVideoOptionalData()
        )
        seconds_kwargs: Final = (
            _RunwayVideoOptionalData(seconds=str(request_data["duration"]))
            if request_data and "duration" in request_data
            else _RunwayVideoOptionalData()
        )

        # Map RunwayML task response to VideoObject format
        video_data: Final[_RunwayVideoData] = {
            "id": response_data.get("id", ""),
            "object": "video",
            "status": self._map_runway_status(response_data.get("status", "pending")),
            "created_at": self._parse_runway_timestamp(response_data.get("createdAt")),
            **output_kwargs,
            **completed_at_kwargs,
            **error_kwargs,
            **model_kwargs,
            **size_kwargs,
            **seconds_kwargs,
        }

        video_obj: Final = VideoObject.model_validate(video_data)

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model)

        # Add usage data for cost tracking
        usage_data: Final = {}
        if video_obj and hasattr(video_obj, "seconds") and video_obj.seconds:
            try:
                usage_data["duration_seconds"] = float(video_obj.seconds)
            except (ValueError, TypeError):
                pass
        video_obj.usage = usage_data

        return video_obj

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

        return url, dict[str, str]()

    def _extract_video_url_from_response(self, response_data: _RunwayTaskResponse) -> str:
        """
        Helper method to extract video URL from RunwayML response.
        Shared between sync and async transforms.
        """
        # Extract video URL from the output field
        video_url = None
        raw_output: Final = response_data.get("output")
        if raw_output:
            video_url = raw_output if isinstance(raw_output, str) else raw_output[0]

        if not video_url:
            # Check if the video generation failed or is still processing
            status: Final = response_data.get("status", "UNKNOWN")
            if status in ["PENDING", "RUNNING", "THROTTLED"]:
                raise ValueError(f"Video is still processing (status: {status}). Please wait and try again.")
            elif status == "FAILED":
                failure_reason: Final = response_data.get("failure", "Unknown error")
                raise ValueError(f"Video generation failed: {failure_reason}")
            else:
                raise ValueError("Video URL not found in response. Video may not be ready yet.")

        return video_url

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
        response_data: Final[_RunwayTaskResponse] = self._parse_task_response(raw_response)
        video_url: Final = self._extract_video_url_from_response(response_data)

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
        response_data: Final[_RunwayTaskResponse] = self._parse_task_response(raw_response)
        video_url: Final = self._extract_video_url_from_response(response_data)

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
        extra_body: Mapping[str, object] | None = None,
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
        extra_query: Mapping[str, object] | None = None,
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

        return url, dict[str, str]()

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """Transform the RunwayML video delete/cancel response."""
        response_data: Final[_RunwayTaskResponse] = self._parse_task_response(raw_response)

        video_obj: Final = VideoObject(
            id=response_data.get("id", ""),
            object="video",
            status="cancelled",
            created_at=self._parse_runway_timestamp(response_data.get("createdAt")),
        )

        return video_obj

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
        return url, dict[str, str]()

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        """
        Transform the RunwayML video status retrieve response.
        """
        response_data: Final[_RunwayTaskResponse] = self._parse_task_response(raw_response)

        output: Final = response_data.get("output")
        output_kwargs: Final = (
            _RunwayVideoOptionalData(output_url=output if isinstance(output, str) else output[0])
            if output
            else _RunwayVideoOptionalData()
        )
        completed_at_kwargs: Final = (
            _RunwayVideoOptionalData(completed_at=self._parse_runway_timestamp(response_data.get("completedAt")))
            if "completedAt" in response_data
            else _RunwayVideoOptionalData()
        )
        progress_kwargs: Final = (
            _RunwayVideoOptionalData(progress=response_data["progress"])
            if "progress" in response_data
            else _RunwayVideoOptionalData()
        )
        error_kwargs: Final = (
            _RunwayVideoOptionalData(
                error={
                    "code": response_data.get("failureCode", "unknown"),
                    "message": response_data.get("failure", "Video generation failed"),
                }
            )
            if "failureCode" in response_data or "failure" in response_data
            else _RunwayVideoOptionalData()
        )

        # Map RunwayML task response to VideoObject format
        video_data: Final[_RunwayVideoData] = {
            "id": response_data.get("id", ""),
            "object": "video",
            "status": self._map_runway_status(response_data.get("status", "pending")),
            "created_at": self._parse_runway_timestamp(response_data.get("createdAt")),
            **output_kwargs,
            **completed_at_kwargs,
            **progress_kwargs,
            **error_kwargs,
        }

        video_obj: Final = VideoObject.model_validate(video_data)

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, None)

        return video_obj

    def transform_video_create_character_request(self, name, video: object, api_base, litellm_params, headers):
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
