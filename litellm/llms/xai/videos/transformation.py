import time
from typing import TYPE_CHECKING, Any, Final

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

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any

_SIZE_TO_ASPECT_RATIO: Final = {
    "1024x1024": "1:1",
    "1792x1024": "16:9",
    "1024x1792": "9:16",
    "1280x720": "16:9",
    "720x1280": "9:16",
    "1920x1080": "16:9",
    "1080x1920": "9:16",
}
def _duration_from_seconds(seconds: object) -> int:
    try:
        return int(seconds) if seconds is not None else 6
    except (TypeError, ValueError):
        return 6


_STATUS_MAP: Final = {
    "done": "completed",
    "completed": "completed",
    "succeeded": "completed",
    "failed": "failed",
    "expired": "failed",
    "pending": "processing",
    "processing": "processing",
    "in_progress": "processing",
}


class XAIVideoConfig(BaseVideoConfig):
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
    ) -> dict:
        incoming: Final = dict(video_create_optional_params)
        size: Final = incoming.get("size")
        return {
            **{
                key: value
                for key, value in incoming.items()
                if key not in {"seconds", "size", "input_reference", "user", "extra_headers", "model"}
            },
            **(
                {"duration": _duration_from_seconds(incoming.get("seconds"))}
                if "seconds" in incoming and "duration" not in incoming
                else {}
            ),
            **(
                {"aspect_ratio": incoming.get("aspect_ratio") or _SIZE_TO_ASPECT_RATIO.get(str(size), "16:9")}
                if size and "aspect_ratio" not in incoming
                else {}
            ),
            **(
                {"image": incoming.get("image") or incoming.get("input_reference")}
                if incoming.get("input_reference") and "image" not in incoming
                else {}
            ),
        }

    def _resolve_api_base(
        self,
        api_base: str | None,
        api_key: str | None,
        litellm_params: GenericLiteLLMParams | dict | None,
    ) -> str:
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        params: Final = (
            litellm_params.model_dump()
            if isinstance(litellm_params, GenericLiteLLMParams)
            else (litellm_params or {})
        )
        if should_use_xai_oauth(params) and not XAIModelInfo.get_api_key(api_key):
            return XAIOAuthAuthenticator().get_api_base().rstrip("/")

        resolved: Final = (
            api_base
            or (params.get("api_base") if isinstance(params, dict) else None)
            or get_secret_str("XAI_API_BASE")
            or get_secret_str("XAI_OAUTH_API_BASE")
            or XAI_API_BASE
        )
        return str(resolved).rstrip("/")

    def _v1_root(self, api_base: str) -> str:
        base: Final = api_base.rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        params: Final = litellm_params.model_dump() if litellm_params is not None else {}
        resolved_api_key: Final = api_key or (litellm_params.api_key if litellm_params else None)
        dynamic_api_key: Final = XAIModelInfo.get_api_key(resolved_api_key)
        if should_use_xai_oauth(params) and not dynamic_api_key:
            try:
                headers["Authorization"] = f"Bearer {XAIOAuthAuthenticator().get_access_token()}"
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
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        resolved: Final = self._resolve_api_base(
            api_base=api_base,
            api_key=litellm_params.get("api_key") if litellm_params else None,
            litellm_params=litellm_params,
        )
        if not model:
            return self._v1_root(resolved)
        return f"{self._v1_root(resolved)}/videos/generations"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        copied: Final = {
            key: video_create_optional_request_params[key]
            for key in (
                "image",
                "images",
                "duration",
                "resolution_name",
                "aspect_ratio",
                "size",
            )
            if video_create_optional_request_params.get(key) is not None
        }
        return (
            {
                "model": XAIModelInfo.get_base_model(model) or model,
                **({"prompt": prompt} if prompt else {}),
                **copied,
                **({"duration": 6} if "duration" not in copied else {}),
            },
            [],
            api_base,
        )

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        response_data: Final = raw_response.json()
        request_id: Final = response_data.get("request_id") or response_data.get("id")
        if not request_id:
            raise ValueError(f"xAI video generation response missing request_id: {response_data}")

        usage: Final = response_data.get("usage") or {}
        video_obj: Final = VideoObject(
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
        video_obj.usage = usage if isinstance(usage, dict) else {}
        video_obj._hidden_params["video_url"] = None
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        original_id: Final = extract_original_video_id(video_id)
        return f"{self._v1_root(api_base)}/videos/{original_id}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response_data: Final = raw_response.json()
        status_raw: Final = str(response_data.get("status") or "processing").lower()
        status: Final = _STATUS_MAP.get(status_raw, status_raw)
        video_meta: Final = response_data.get("video") or {}
        video_url: Final = video_meta.get("url") if isinstance(video_meta, dict) else None
        seconds: Final = (
            str(video_meta.get("duration"))
            if isinstance(video_meta, dict) and video_meta.get("duration") is not None
            else None
        )
        request_id: Final = (
            response_data.get("request_id")
            or response_data.get("id")
            or (video_url.split("/")[-1].replace(".mp4", "") if video_url else "unknown")
        )
        video_obj: Final = VideoObject(
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
        variant: str | None = None,
    ) -> tuple[str, dict]:
        original_id: Final = extract_original_video_id(video_id)
        return f"{self._v1_root(api_base)}/videos/{original_id}", {}

    def _video_cdn_url(self, raw_response: httpx.Response) -> str | None:
        content_type: Final = (raw_response.headers.get("content-type") or "").lower()
        if "application/json" not in content_type and raw_response.content[:1] != b"{":
            return None
        payload: Final = raw_response.json()
        if not isinstance(payload, dict):
            return None
        video_meta: Final = payload.get("video") or {}
        url: Final = video_meta.get("url") if isinstance(video_meta, dict) else None
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
        url: Final = self._video_cdn_url(raw_response)
        if url is None:
            return raw_response.content
        httpx_client: Final[HTTPHandler] = _get_httpx_client()
        video_response: Final = httpx_client.get(url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        url: Final = self._video_cdn_url(raw_response)
        if url is None:
            return raw_response.content
        async_httpx_client: Final[AsyncHTTPHandler] = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.XAI,
        )
        video_response: Final = await async_httpx_client.get(url)
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
        raise NotImplementedError("Video remix is not supported by xAI Imagine API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by xAI Imagine API")

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
        raise NotImplementedError("Video listing is not supported by xAI Imagine API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        raise NotImplementedError("Video listing is not supported by xAI Imagine API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        raise NotImplementedError("Video delete is not supported by xAI Imagine API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by xAI Imagine API")
