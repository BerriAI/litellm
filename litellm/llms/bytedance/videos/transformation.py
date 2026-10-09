from typing import TYPE_CHECKING, Any

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.constants import DEFAULT_BYTEDANCE_VIDEO_DURATION_SECONDS
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_httpx_client,
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

BYTEDANCE_DEFAULT_API_BASE = "https://ark.ap-southeast.bytepluses.com"
BYTEDANCE_API_PATH_PREFIX = "/api/v3/contents/generations/tasks"

SEEDANCE_STATUS_MAP = {
    "queued": "queued",
    "running": "in_progress",
    "succeeded": "completed",
    "failed": "failed",
    "cancelled": "failed",
    "expired": "failed",
}


class ByteDanceVideoConfig(BaseVideoConfig):
    def __init__(self) -> None:
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "last_frame",
            "reference_images",
            "seconds",
            "size",
            "resolution",
            "seed",
            "generate_audio",
            "watermark",
            "return_last_frame",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:
        mapped_params: dict[str, Any] = {}

        if "input_reference" in video_create_optional_params:
            mapped_params["input_reference"] = video_create_optional_params["input_reference"]

        if "last_frame" in video_create_optional_params:
            mapped_params["last_frame"] = video_create_optional_params["last_frame"]

        if "reference_images" in video_create_optional_params:
            mapped_params["reference_images"] = video_create_optional_params["reference_images"]

        if "size" in video_create_optional_params:
            size = video_create_optional_params["size"]
            if isinstance(size, str) and "x" in size:
                mapped_params["ratio"] = size.replace("x", ":")
            elif isinstance(size, str):
                mapped_params["ratio"] = size

        if "seconds" in video_create_optional_params:
            seconds = video_create_optional_params["seconds"]
            if seconds is not None:
                try:
                    mapped_params["duration"] = int(float(seconds)) if isinstance(seconds, str) else int(seconds)
                except (ValueError, TypeError):
                    pass

        if "resolution" in video_create_optional_params:
            mapped_params["resolution"] = video_create_optional_params["resolution"]

        for key in ("seed", "generate_audio", "watermark", "return_last_frame"):
            if key in video_create_optional_params:
                mapped_params[key] = video_create_optional_params[key]

        supported_openai_params = self.get_supported_openai_params(model)
        for key, value in video_create_optional_params.items():
            if key not in supported_openai_params and key not in mapped_params:
                mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = api_key or litellm.api_key or get_secret_str("BYTEDANCE_API_KEY") or get_secret_str("ARK_API_KEY")

        if api_key is None:
            raise ValueError(
                "ByteDance API key is required. Set BYTEDANCE_API_KEY or ARK_API_KEY "
                "environment variable or pass api_key parameter."
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
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        if api_base is None:
            api_base = BYTEDANCE_DEFAULT_API_BASE
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
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]

        input_reference = video_create_optional_request_params.pop("input_reference", None)
        last_frame = video_create_optional_request_params.pop("last_frame", None)
        reference_images = video_create_optional_request_params.pop("reference_images", None)

        has_frame_mode = input_reference is not None or last_frame is not None
        has_reference_mode = reference_images is not None and len(reference_images) > 0
        if has_frame_mode and has_reference_mode:
            raise ValueError(
                "Seedance API does not allow first_frame/last_frame and reference_images "
                "in the same request. Use either frame mode (input_reference, last_frame) "
                "or reference mode (reference_images), not both."
            )

        if input_reference is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": str(input_reference)},
                    "role": "first_frame",
                }
            )

        if last_frame is not None:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": str(last_frame)},
                    "role": "last_frame",
                }
            )

        if reference_images is not None:
            for ref_url in reference_images:
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": str(ref_url)},
                        "role": "reference_image",
                    }
                )

        request_data: dict[str, Any] = {
            "model": model,
            "content": content,
        }

        if "ratio" in video_create_optional_request_params:
            request_data["ratio"] = video_create_optional_request_params.pop("ratio")
        if "duration" in video_create_optional_request_params:
            request_data["duration"] = video_create_optional_request_params.pop("duration")
        if "resolution" in video_create_optional_request_params:
            request_data["resolution"] = video_create_optional_request_params.pop("resolution")

        for key, value in video_create_optional_request_params.items():
            if key not in request_data:
                request_data[key] = value

        files_list: list[tuple[str, Any]] = []
        full_url = f"{api_base}{BYTEDANCE_API_PATH_PREFIX}"

        return request_data, files_list, full_url

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        response_data = raw_response.json()

        task_id = response_data.get("id", "")

        video_data: dict[str, Any] = {
            "id": task_id,
            "object": "video",
            "status": "queued",
            "created_at": response_data.get("created_at", 0),
        }

        if request_data:
            if "model" in request_data:
                video_data["model"] = request_data["model"]
            if "ratio" in request_data:
                ratio = request_data["ratio"]
                if isinstance(ratio, str) and ":" in ratio:
                    video_data["size"] = ratio.replace(":", "x")
            if "duration" in request_data:
                video_data["seconds"] = str(request_data["duration"])

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model)

        duration = float(video_obj.seconds) if video_obj.seconds else float(DEFAULT_BYTEDANCE_VIDEO_DURATION_SECONDS)
        usage_data: dict[str, Any] = {"duration_seconds": duration}
        if request_data:
            res = request_data.get("resolution")
            if res is not None and str(res).strip() != "":
                usage_data["video_resolution"] = str(res).strip().lower()
        video_obj.usage = usage_data

        return video_obj

    def _map_seedance_status(self, status: str) -> str:
        return SEEDANCE_STATUS_MAP.get(status.lower(), "queued")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        original_video_id = extract_original_video_id(video_id)
        encoded_video_id = encode_url_path_segment(original_video_id, field_name="video_id")
        url = f"{api_base}{BYTEDANCE_API_PATH_PREFIX}/{encoded_video_id}"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response_data = raw_response.json()

        video_data: dict[str, Any] = {
            "id": response_data.get("id", ""),
            "object": "video",
            "status": self._map_seedance_status(response_data.get("status", "queued")),
            "created_at": response_data.get("created_at", 0),
        }

        if "updated_at" in response_data and response_data.get("status") in (
            "succeeded",
            "failed",
            "cancelled",
            "expired",
        ):
            video_data["completed_at"] = response_data["updated_at"]

        if "duration" in response_data:
            video_data["seconds"] = str(response_data["duration"])

        if "ratio" in response_data:
            ratio = response_data["ratio"]
            if isinstance(ratio, str) and ":" in ratio:
                video_data["size"] = ratio.replace(":", "x")

        if "model" in response_data:
            video_data["model"] = response_data["model"]

        if "error" in response_data and response_data["error"]:
            video_data["error"] = response_data["error"]

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, None)

        usage_data: dict[str, Any] = response_data.get("usage", {})
        duration = float(video_obj.seconds) if video_obj.seconds else float(DEFAULT_BYTEDANCE_VIDEO_DURATION_SECONDS)
        usage_data["duration_seconds"] = duration
        res = response_data.get("resolution")
        if res is not None and str(res).strip() != "":
            usage_data["video_resolution"] = str(res).strip().lower()
        video_obj.usage = usage_data

        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        original_video_id = extract_original_video_id(video_id)
        encoded_video_id = encode_url_path_segment(original_video_id, field_name="video_id")
        url = f"{api_base}{BYTEDANCE_API_PATH_PREFIX}/{encoded_video_id}"
        return url, {}

    def _extract_video_url_from_response(self, response_data: dict[str, Any]) -> str:
        content = response_data.get("content")
        if isinstance(content, dict):
            video_url = content.get("video_url")
            if video_url:
                return str(video_url)

        status = response_data.get("status", "unknown")
        if status in ("queued", "running"):
            raise ValueError(f"Video is still processing (status: {status}). Please wait and try again.")
        if status == "failed":
            error = response_data.get("error", {})
            message = error.get("message", "Unknown error") if error else "Unknown error"
            raise ValueError(f"Video generation failed: {message}")

        raise ValueError("Video URL not found in response. Video may not be ready yet.")

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        video_url = self._extract_video_url_from_response(response_data)

        httpx_client: HTTPHandler = get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()

        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        response_data = raw_response.json()
        video_url = self._extract_video_url_from_response(response_data)

        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.BYTEDANCE,
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
        extra_body: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video remix is not supported for ByteDance")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported for ByteDance")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, Any] | None = None,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video listing is not supported for ByteDance")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError("Video listing is not supported for ByteDance")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        original_video_id = extract_original_video_id(video_id)
        encoded_video_id = encode_url_path_segment(original_video_id, field_name="video_id")
        url = f"{api_base}{BYTEDANCE_API_PATH_PREFIX}/{encoded_video_id}"
        return url, {}

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        response_data = raw_response.json()
        return VideoObject(
            id=response_data.get("id", ""),
            object="video",
            status="failed",
            created_at=response_data.get("created_at", 0),
        )  # type: ignore[arg-type]

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict | httpx.Headers
    ) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
