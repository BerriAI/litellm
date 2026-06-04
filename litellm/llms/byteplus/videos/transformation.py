import math
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

import litellm
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


DEFAULT_BASE_URL = "https://ark.ap-southeast.bytepluses.com/api/v3"
VIDEO_TASKS_ENDPOINT = "contents/generations/tasks"


class BytePlusVideoConfig(BaseVideoConfig):
    """
    BytePlus (ByteDance Ark) video generation (seedance / dreamina models).

    Task-based async API:
    1. POST /contents/generations/tasks  -> {"id": "cgt-..."}
    2. GET  /contents/generations/tasks/{id} -> {status, content: {video_url}, usage, ...}

    Reference: https://docs.byteplus.com/en/docs/ModelArk/1520757
    """

    def __init__(self):
        super().__init__()

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
        """
        Map OpenAI video params to BytePlus:
        - size "1280x720" -> ratio "16:9" (pixels reduced to an aspect ratio)
        - seconds -> duration (int)
        - input_reference (image url) -> kept; assembled into the content array
          in transform_video_create_request
        - BytePlus-specific params (generate_audio, watermark, ratio, ...) pass through
        """
        mapped_params: Dict[str, Any] = {}

        if "input_reference" in video_create_optional_params:
            mapped_params["input_reference"] = video_create_optional_params[
                "input_reference"
            ]

        if "size" in video_create_optional_params:
            # OpenAI `size` is pixel dimensions ("1280x720"); BytePlus `ratio` is an
            # aspect ratio ("16:9"). Reduce the dimensions to lowest terms.
            size = video_create_optional_params["size"]
            if isinstance(size, str) and "x" in size:
                try:
                    w, h = (int(p) for p in size.lower().split("x", 1))
                    g = math.gcd(w, h) or 1
                    mapped_params["ratio"] = f"{w // g}:{h // g}"
                except (ValueError, TypeError):
                    pass

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

        # Pass through provider-specific params (generate_audio, watermark, etc.)
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

        api_key = api_key or litellm.api_key or get_secret_str("BYTEPLUS_API_KEY")

        if api_key is None:
            raise ValueError(
                "BYTEPLUS_API_KEY is required. Set BYTEPLUS_API_KEY environment "
                "variable or pass api_key parameter."
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
        api_base = api_base or get_secret_str("BYTEPLUS_API_BASE") or DEFAULT_BASE_URL
        return api_base.rstrip("/")

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
        BytePlus expects:
        {
          "model": "seedance-1-5-pro-251215",
          "content": [
            {"type": "text", "text": "..."},
            {"type": "image_url", "image_url": {"url": "..."}}   # optional
          ],
          "ratio": "16:9", "duration": 5, "generate_audio": true, ...
        }
        """
        params = dict(video_create_optional_request_params)

        content: List[Dict[str, Any]] = []
        if prompt:
            content.append({"type": "text", "text": prompt})

        input_reference = params.pop("input_reference", None)
        if input_reference:
            content.append({"type": "image_url", "image_url": {"url": input_reference}})

        request_data: Dict[str, Any] = {"model": model, "content": content}
        request_data.update(params)

        files_list: List[Tuple[str, Any]] = []
        full_api_base = f"{api_base}/{VIDEO_TASKS_ENDPOINT}"
        return request_data, files_list, full_api_base

    def _map_status(self, status: Optional[str]) -> str:
        """Map BytePlus task status to OpenAI video status."""
        status_map = {
            "queued": "queued",
            "running": "in_progress",
            "succeeded": "completed",
            "failed": "failed",
            "cancelled": "failed",
        }
        return status_map.get((status or "").lower(), "queued")

    def _build_video_object(
        self,
        response_data: Dict[str, Any],
        model: Optional[str],
        custom_llm_provider: Optional[str],
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        video_data: Dict[str, Any] = {
            "id": response_data.get("id", ""),
            "object": "video",
            "status": self._map_status(response_data.get("status")),
            "created_at": int(response_data.get("created_at") or 0),
        }

        content = response_data.get("content") or {}
        if isinstance(content, dict) and content.get("video_url"):
            video_data["output_url"] = content["video_url"]

        if response_data.get("updated_at"):
            video_data["completed_at"] = int(response_data["updated_at"])

        if response_data.get("error"):
            err = response_data["error"]
            video_data["error"] = {
                "code": err.get("code", "unknown"),
                "message": err.get("message", "Video generation failed"),
            }

        if response_data.get("ratio"):
            ratio = response_data["ratio"]
            if isinstance(ratio, str) and ":" in ratio:
                video_data["size"] = ratio.replace(":", "x")
        if response_data.get("duration") is not None:
            video_data["seconds"] = str(response_data["duration"])
        elif request_data and request_data.get("duration") is not None:
            video_data["seconds"] = str(request_data["duration"])

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )

        # Usage for cost tracking (duration-based)
        usage_data: Dict[str, Any] = {}
        if getattr(video_obj, "seconds", None):
            try:
                usage_data["duration_seconds"] = float(video_obj.seconds)
            except (ValueError, TypeError):
                pass
        video_obj.usage = usage_data

        return video_obj

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        return self._build_video_object(
            raw_response.json(), model, custom_llm_provider, request_data
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        encoded = encode_url_path_segment(original_video_id, field_name="video_id")
        url = f"{api_base}/{VIDEO_TASKS_ENDPOINT}/{encoded}"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        return self._build_video_object(raw_response.json(), None, custom_llm_provider)

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        encoded = encode_url_path_segment(original_video_id, field_name="video_id")
        url = f"{api_base}/{VIDEO_TASKS_ENDPOINT}/{encoded}"
        return url, {}

    def _extract_video_url_from_response(self, response_data: Dict[str, Any]) -> str:
        content = response_data.get("content") or {}
        video_url = content.get("video_url") if isinstance(content, dict) else None
        if not video_url:
            status = (response_data.get("status") or "UNKNOWN").lower()
            if status in ("queued", "running"):
                raise ValueError(
                    f"Video is still processing (status: {status}). "
                    "Please wait and try again."
                )
            if status == "failed":
                err = response_data.get("error") or {}
                raise ValueError(
                    f"Video generation failed: {err.get('message', 'Unknown error')}"
                )
            raise ValueError(
                "Video URL not found in response. Video may not be ready yet."
            )
        return video_url

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        video_url = self._extract_video_url_from_response(raw_response.json())
        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        video_url = self._extract_video_url_from_response(raw_response.json())
        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.BYTEPLUS,
        )
        video_response = await async_httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_video_id = extract_original_video_id(video_id)
        encoded = encode_url_path_segment(original_video_id, field_name="video_id")
        url = f"{api_base}/{VIDEO_TASKS_ENDPOINT}/{encoded}"
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
            status="cancelled",
            created_at=int(response_data.get("created_at") or 0),
        )  # type: ignore[arg-type]

    # ---- unsupported operations ----
    def transform_video_remix_request(
        self, video_id, prompt, api_base, litellm_params, headers, extra_body=None
    ):
        raise NotImplementedError("Video remix is not supported for BytePlus")

    def transform_video_remix_response(
        self, raw_response, logging_obj, custom_llm_provider=None
    ):
        raise NotImplementedError("Video remix is not supported for BytePlus")

    def transform_video_list_request(
        self,
        api_base,
        litellm_params,
        headers,
        after=None,
        limit=None,
        order=None,
        extra_query=None,
    ):
        raise NotImplementedError("Video listing is not supported for BytePlus")

    def transform_video_list_response(
        self, raw_response, logging_obj, custom_llm_provider=None
    ):
        raise NotImplementedError("Video listing is not supported for BytePlus")

    def transform_video_create_character_request(
        self, name, video, api_base, litellm_params, headers
    ):
        raise NotImplementedError(
            "video create character is not supported for BytePlus"
        )

    def transform_video_create_character_response(self, raw_response, logging_obj):
        raise NotImplementedError(
            "video create character is not supported for BytePlus"
        )

    def transform_video_get_character_request(
        self, character_id, api_base, litellm_params, headers
    ):
        raise NotImplementedError("video get character is not supported for BytePlus")

    def transform_video_get_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video get character is not supported for BytePlus")

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
        raise NotImplementedError("video edit is not supported for BytePlus")

    def transform_video_edit_response(
        self, raw_response, logging_obj, custom_llm_provider=None, request_data=None
    ):
        raise NotImplementedError("video edit is not supported for BytePlus")

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
        raise NotImplementedError("video extension is not supported for BytePlus")

    def transform_video_extension_response(
        self, raw_response, logging_obj, custom_llm_provider=None
    ):
        raise NotImplementedError("video extension is not supported for BytePlus")

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
