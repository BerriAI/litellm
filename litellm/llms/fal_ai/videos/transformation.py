from json import JSONDecodeError
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.constants import FAL_AI_DEFAULT_API_BASE
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.fal_ai.utils import normalize_fal_model_id as _normalize_fal_model_id
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


_FAL_AI_STATUS_MAP = {
    "IN_QUEUE": "queued",
    "IN_PROGRESS": "in_progress",
    "COMPLETED": "completed",
    "FAILED": "failed",
    "CANCELLED": "failed",
}

_SIZE_TO_ASPECT_RATIO = {
    "1280x720": "16:9",
    "1920x1080": "16:9",
    "720x1280": "9:16",
    "1080x1920": "9:16",
    "1024x1024": "1:1",
    "1280x1280": "1:1",
}


class FalAIVideoConfig(BaseVideoConfig):
    """
    fal.ai uses a queue API: POST to /{model_id}, then poll
    /{model_id}/requests/{id}/status and GET /{model_id}/requests/{id} for the
    result. Video models return {"video": {"url": ...}}.
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
            "extra_body",
        ]

    @staticmethod
    def _image_url_field_for_model(model: str) -> str:
        # Kling v3 image-to-video requires `start_image_url`; Seedance uses `image_url`.
        normalized = model.lower()
        if "kling-video/v3" in normalized:
            return "start_image_url"
        return "image_url"

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped: Dict[str, Any] = {}

        seconds = video_create_optional_params.get("seconds")
        if seconds is not None:
            mapped["duration"] = str(seconds)

        size = video_create_optional_params.get("size")
        if isinstance(size, str):
            aspect = _SIZE_TO_ASPECT_RATIO.get(size)
            if aspect is not None:
                mapped["aspect_ratio"] = aspect
            elif "x" in size:
                mapped["aspect_ratio"] = size.replace("x", ":")

        input_reference = video_create_optional_params.get("input_reference")
        if isinstance(input_reference, str) and input_reference:
            mapped[self._image_url_field_for_model(model)] = input_reference

        supported = self.get_supported_openai_params(model)
        for key, value in video_create_optional_params.items():
            if key not in supported:
                mapped[key] = value

        extra_body = video_create_optional_params.get("extra_body")
        if isinstance(extra_body, dict):
            mapped.update(extra_body)
            mapped.pop("extra_body", None)

        return mapped

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        resolved_key = (
            api_key
            or litellm.api_key
            or get_secret_str("FAL_AI_API_KEY")
            or get_secret_str("FAL_KEY")
        )

        if not resolved_key:
            raise ValueError(
                "fal.ai API key is required. Set FAL_AI_API_KEY (or FAL_KEY) "
                "environment variable or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Key {resolved_key}",
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
        base = api_base or get_secret_str("FAL_AI_API_BASE") or FAL_AI_DEFAULT_API_BASE
        return base.rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        model_id = _normalize_fal_model_id(model)

        request_data: Dict[str, Any] = {"prompt": prompt}
        request_data.update(video_create_optional_request_params)
        request_data.pop("model", None)

        return request_data, [], f"{api_base}/{model_id}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        model_id = _normalize_fal_model_id(model)

        video_data: Dict[str, Any] = {
            "id": response_data.get("request_id", ""),
            "object": "video",
            "status": _FAL_AI_STATUS_MAP.get(
                response_data.get("status", "IN_QUEUE").upper(), "queued"
            ),
            "model": model,
        }

        if request_data:
            if "duration" in request_data:
                video_data["seconds"] = str(request_data["duration"])
            if "aspect_ratio" in request_data:
                video_data["size"] = str(request_data["aspect_ratio"]).replace(":", "x")

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id,
                custom_llm_provider,
                model_id,
            )

        usage: Dict[str, Any] = {}
        if video_obj.seconds:
            try:
                usage["duration_seconds"] = float(video_obj.seconds)
            except (ValueError, TypeError):
                pass
        video_obj.usage = usage

        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_id, model_id = self._extract_request_and_model_id(video_id)
        encoded = encode_url_path_segment(original_id, field_name="video_id")
        namespace = self._queue_request_namespace(model_id)
        return f"{api_base}/{namespace}/requests/{encoded}/status", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        self._raise_for_status(raw_response)
        try:
            response_data = raw_response.json()
        except (ValueError, JSONDecodeError):
            return VideoObject(id="", object="video", status="in_progress")
        status_raw = response_data.get("status", "IN_QUEUE")
        error_payload = response_data.get("error")

        status = _FAL_AI_STATUS_MAP.get(status_raw.upper(), "queued")
        if error_payload:
            status = "failed"

        video_data: Dict[str, Any] = {
            "id": response_data.get("request_id", ""),
            "object": "video",
            "status": status,
        }

        if "queue_position" in response_data:
            video_data["progress"] = response_data["queue_position"]

        if status == "failed":
            video_data["error"] = {
                "code": "failed",
                "message": str(error_payload or "Video generation failed"),
            }

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            model_id = self._model_id_from_request_url(raw_response)
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model_id
            )

        return video_obj

    @staticmethod
    def _model_id_from_request_url(raw_response: httpx.Response) -> Optional[str]:
        request = getattr(raw_response, "request", None)
        if request is None:
            return None
        path = request.url.path
        head = path.split("/requests/", 1)[0].strip("/")
        return head or None

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        original_id, model_id = self._extract_request_and_model_id(video_id)
        encoded = encode_url_path_segment(original_id, field_name="video_id")
        namespace = self._queue_request_namespace(model_id)
        return f"{api_base}/{namespace}/requests/{encoded}", {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        video_url = self._extract_video_url(raw_response.json())
        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        self._raise_for_status(raw_response)
        video_url = self._extract_video_url(raw_response.json())
        async_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.FAL_AI,
        )
        video_response = await async_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    @staticmethod
    def _extract_video_url(response_data: Dict[str, Any]) -> str:
        error_payload = response_data.get("error")
        if error_payload:
            raise ValueError(f"fal.ai video generation failed: {error_payload}")

        video = response_data.get("video")
        if isinstance(video, dict):
            url = video.get("url")
            if isinstance(url, str) and url:
                return url

        top_level = response_data.get("url")
        if isinstance(top_level, str) and top_level:
            return top_level

        raise ValueError(
            "Video URL not found in fal.ai response. The job may still be processing."
        )

    @staticmethod
    def _queue_request_namespace(model_id: str) -> str:
        # Queue submits accept full model subpaths (fal-ai/kling-video/v3/pro/
        # image-to-video), but request status/result routes only exist under the
        # owner/app prefix; deeper paths answer 405 Method Not Allowed.
        segments = [segment for segment in model_id.split("/") if segment]
        return "/".join(segments[:2])

    @staticmethod
    def _extract_request_and_model_id(video_id: str) -> Tuple[str, str]:
        # Queue URLs are always rebuilt from api_base + model_id + request id, never
        # taken from the (caller-supplied, only base64-encoded) video_id. Trusting an
        # embedded URL would let a forged id redirect fal-authenticated requests to an
        # arbitrary host and leak the API key.
        decoded = decode_video_id_with_provider(video_id)
        original_id = decoded.get("video_id") or extract_original_video_id(video_id)
        model_id = decoded.get("model_id")

        if not model_id:
            raise ValueError(
                "fal.ai video status/content lookup requires a model id encoded "
                "in the video_id. Use the id returned by video creation."
            )

        return original_id, model_id

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError(
            "Video remix is not supported by the fal.ai queue API"
        )

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError(
            "Video remix is not supported by the fal.ai queue API"
        )

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
        raise NotImplementedError(
            "Video listing is not supported by the fal.ai queue API"
        )

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError(
            "Video listing is not supported by the fal.ai queue API"
        )

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        # fal cancels jobs via PUT /requests/{id}/cancel, not the DELETE the shared handler issues.
        raise NotImplementedError(
            "Video delete/cancel is not supported by the fal.ai queue API via LiteLLM"
        )

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError(
            "Video delete/cancel is not supported by the fal.ai queue API via LiteLLM"
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _raise_for_status(self, raw_response: httpx.Response) -> None:
        if raw_response.is_success:
            return
        raise self.get_error_class(
            error_message=raw_response.text,
            status_code=raw_response.status_code,
            headers=raw_response.headers,
        )
