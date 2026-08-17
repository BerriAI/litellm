import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.constants import XAI_API_BASE
from litellm.exceptions import AuthenticationError
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,
    get_async_httpx_client,
)
from litellm.llms.xai.common_utils import XAIModelInfo
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any

_SIZE_TO_ASPECT_RATIO = {
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1920x1080": "16:9",
    "1080x1920": "9:16",
}


class XAIVideoConfig(BaseVideoConfig):
    """
    xAI Imagine video generation.

    Create: POST /v1/videos/generations  -> {request_id}
    Status: GET  /v1/videos/{request_id} -> {status, video.url, ...}
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
        mapped: Dict[str, Any] = dict(video_create_optional_params)

        if "seconds" in mapped and "duration" not in mapped:
            try:
                seconds = mapped.pop("seconds")
                duration = int(seconds) if seconds is not None else 6
            except (TypeError, ValueError):
                duration = 6
            mapped["duration"] = duration

        if "size" in mapped and "aspect_ratio" not in mapped:
            size = mapped.pop("size")
            if size:
                mapped["aspect_ratio"] = _SIZE_TO_ASPECT_RATIO.get(str(size), "16:9")

        if "input_reference" in mapped and "image" not in mapped:
            mapped["image"] = mapped.pop("input_reference")

        mapped.pop("user", None)
        mapped.pop("extra_headers", None)
        mapped.pop("model", None)
        return mapped

    def _resolve_api_base(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        litellm_params: Optional[Union[GenericLiteLLMParams, dict]],
    ) -> str:
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        params = (
            litellm_params.model_dump()
            if isinstance(litellm_params, GenericLiteLLMParams)
            else (litellm_params or {})
        )
        if should_use_xai_oauth(params) and not XAIModelInfo.get_api_key(api_key):
            token_file = params.get("xai_oauth_token_file")
            return XAIOAuthAuthenticator(auth_file=token_file).get_api_base().rstrip("/")

        resolved = (
            api_base
            or (params.get("api_base") if isinstance(params, dict) else None)
            or get_secret_str("XAI_API_BASE")
            or get_secret_str("XAI_OAUTH_API_BASE")
            or XAI_API_BASE
        )
        return str(resolved).rstrip("/")

    def _v1_root(self, api_base: str) -> str:
        base = api_base.rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        params = litellm_params.model_dump() if litellm_params is not None else {}
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        dynamic_api_key = XAIModelInfo.get_api_key(api_key)
        if should_use_xai_oauth(params) and not dynamic_api_key:
            token_file = params.get("xai_oauth_token_file")
            try:
                headers["Authorization"] = (
                    f"Bearer {XAIOAuthAuthenticator(auth_file=token_file).get_access_token()}"
                )
            except XAIOAuthError as exc:
                raise AuthenticationError(
                    model=model or "xai-video",
                    llm_provider="xai",
                    message=str(exc),
                ) from exc
        else:
            if not dynamic_api_key:
                raise AuthenticationError(
                    model=model or "xai-video",
                    llm_provider="xai",
                    message=(
                        "Missing xAI credentials for video generation. "
                        "Pass api_key / XAI_API_KEY, or set use_xai_oauth=True."
                    ),
                )
            headers["Authorization"] = f"Bearer {dynamic_api_key}"

        if "content-type" not in headers and "Content-Type" not in headers:
            headers["Content-Type"] = "application/json"
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        resolved = self._resolve_api_base(
            api_base=api_base,
            api_key=litellm_params.get("api_key") if litellm_params else None,
            litellm_params=litellm_params,
        )
        # Empty model used by status/content handlers that only need the root.
        if not model:
            return self._v1_root(resolved)
        return f"{self._v1_root(resolved)}/videos/generations"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        request: Dict[str, Any] = {
            "model": XAIModelInfo.get_base_model(model) or model,
        }
        if prompt:
            request["prompt"] = prompt

        for key in (
            "image",
            "images",
            "duration",
            "resolution_name",
            "aspect_ratio",
            "size",
        ):
            if key in video_create_optional_request_params and video_create_optional_request_params[key] is not None:
                request[key] = video_create_optional_request_params[key]

        # Default duration when neither seconds nor duration provided
        if "duration" not in request:
            request["duration"] = 6

        files_list: List[Tuple[str, Any]] = []
        return request, files_list, api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        request_id = response_data.get("request_id") or response_data.get("id")
        if not request_id:
            raise ValueError(f"xAI video generation response missing request_id: {response_data}")

        video_obj = VideoObject(
            id=str(request_id),
            object="video",
            status="processing",
            created_at=int(time.time()),
            model=XAIModelInfo.get_base_model(model) or model,
            progress=0,
        )
        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )
        usage = response_data.get("usage") or {}
        video_obj.usage = usage if isinstance(usage, dict) else {}
        video_obj._hidden_params["video_url"] = None
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        original_id = extract_original_video_id(video_id)
        root = self._v1_root(api_base)
        # Status is GET /v1/videos/{request_id} (not under /videos/generations)
        url = f"{root}/videos/{original_id}"
        return url, {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        response_data = raw_response.json()
        status_raw = str(response_data.get("status") or "processing").lower()
        status_map = {
            "done": "completed",
            "completed": "completed",
            "succeeded": "completed",
            "failed": "failed",
            "expired": "failed",
            "pending": "processing",
            "processing": "processing",
            "in_progress": "processing",
        }
        status = status_map.get(status_raw, status_raw)

        video_meta = response_data.get("video") or {}
        video_url = None
        seconds = None
        if isinstance(video_meta, dict):
            video_url = video_meta.get("url")
            if video_meta.get("duration") is not None:
                seconds = str(video_meta.get("duration"))

        # Prefer request id from URL path is not available; keep provider id if present
        request_id = (
            response_data.get("request_id")
            or response_data.get("id")
            or (video_url.split("/")[-1].replace(".mp4", "") if video_url else "unknown")
        )

        video_obj = VideoObject(
            id=str(request_id),
            object="video",
            status=status,
            created_at=response_data.get("created_at") or int(time.time()),
            completed_at=int(time.time()) if status == "completed" else None,
            model=response_data.get("model"),
            progress=response_data.get("progress"),
            seconds=seconds,
            usage=response_data.get("usage") if isinstance(response_data.get("usage"), dict) else {},
        )
        video_obj._hidden_params["video_url"] = video_url
        if custom_llm_provider and video_obj.id and video_obj.id != "unknown":
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, response_data.get("model")
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
        original_id = extract_original_video_id(video_id)
        root = self._v1_root(api_base)
        return f"{root}/videos/{original_id}", {}

    def _video_cdn_url(self, raw_response: httpx.Response) -> Optional[str]:
        content_type = (raw_response.headers.get("content-type") or "").lower()
        if "application/json" not in content_type and raw_response.content[:1] != b"{":
            return None
        payload = raw_response.json()
        if not isinstance(payload, dict):
            return None
        video_meta = payload.get("video") or {}
        url = video_meta.get("url") if isinstance(video_meta, dict) else None
        if isinstance(url, str) and url:
            return url
        raise ValueError(
            f"xAI video not ready for download (status={payload.get('status')}): {payload}"
        )

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        url = self._video_cdn_url(raw_response)
        if url is None:
            return raw_response.content
        httpx_client: HTTPHandler = _get_httpx_client()
        video_response = httpx_client.get(url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        url = self._video_cdn_url(raw_response)
        if url is None:
            return raw_response.content
        async_httpx_client: AsyncHTTPHandler = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.XAI,
        )
        video_response = await async_httpx_client.get(url)
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
        raise NotImplementedError("Video remix is not supported by xAI Imagine API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by xAI Imagine API")

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
        raise NotImplementedError("Video listing is not supported by xAI Imagine API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video listing is not supported by xAI Imagine API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video delete is not supported by xAI Imagine API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by xAI Imagine API")

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
