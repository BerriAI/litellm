import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles
from typing_extensions import TypeIs  # noqa: TID251  # narrows untyped wire payloads without a runtime conversion

import litellm
from litellm.constants import XAI_API_BASE
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.url_utils import async_safe_get, encode_url_path_segment, safe_get
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
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
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj


def _is_json_object(value: object) -> TypeIs[Mapping[str, object]]:  # guard-ok: provider JSON objects have str keys
    return isinstance(value, Mapping)


_DROPPED: Final = frozenset(("seconds", "size", "input_reference", "user", "extra_headers", "model"))
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
    def get_supported_openai_params(
        self, model: str
    ) -> list:  # mutable-ok: provider JSON body and base-class dict signature
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
    ) -> dict:  # mutable-ok: provider JSON body and base-class dict signature
        raw: Final = video_create_optional_params
        incoming: Final = dict(raw)
        size: Final = incoming.get("size")
        seconds: Final = incoming.get("seconds")
        use_duration: Final = "seconds" in incoming and "duration" not in incoming
        duration_value: Final = _duration_from_seconds(seconds)
        duration: Final = {"duration": duration_value} if use_duration else None
        mapped_ratio: Final = incoming.get("aspect_ratio") or _SIZE_TO_ASPECT_RATIO.get(str(size), "16:9")
        use_ratio: Final = bool(size) and "aspect_ratio" not in incoming
        ratio: Final = {"aspect_ratio": mapped_ratio} if use_ratio else None
        image_ref: Final = incoming.get("image") or incoming.get("input_reference")
        use_image: Final = bool(incoming.get("input_reference")) and "image" not in incoming
        image: Final = {"image": image_ref} if use_image else None
        return {
            **{key: value for key, value in incoming.items() if key not in _DROPPED},
            **(duration or {}),
            **(ratio or {}),
            **(image or {}),
        }

    def _resolve_api_base(
        self,
        api_base: str | None,
        api_key: str | None,
        litellm_params: GenericLiteLLMParams | dict[str, object] | None,  # mutable-ok: base-class dict
    ) -> str:
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        params: Final = (
            litellm_params.model_dump() if isinstance(litellm_params, GenericLiteLLMParams) else (litellm_params or {})
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
        headers: dict,  # mutable-ok: provider JSON body and base-class dict signature
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:  # mutable-ok: provider JSON body and base-class dict signature
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        dumped: Final = litellm_params.model_dump() if litellm_params is not None else None
        params: Final = dumped or {}
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
        litellm_params: dict[str, object],  # mutable-ok: provider JSON body and base-class dict signature
    ) -> str:
        api_key: Final = litellm_params.get("api_key") if litellm_params else None
        resolved: Final = self._resolve_api_base(
            api_base=api_base,
            api_key=api_key if isinstance(api_key, str) else None,
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
        video_create_optional_request_params: dict,  # mutable-ok: provider JSON body and base-class dict signature
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: provider JSON body and base-class dict signature
    ) -> tuple[dict, RequestFiles, str]:  # mutable-ok: provider JSON body and base-class dict signature
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
        prompt_body: Final = {"prompt": prompt} if prompt else None
        duration_body: Final = {"duration": 6} if "duration" not in copied else None
        return (
            {
                "model": XAIModelInfo.get_base_model(model) or model,
                **(prompt_body or {}),
                **copied,
                **(duration_body or {}),
            },
            [],
            api_base,
        )

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,  # mutable-ok: provider JSON body and base-class dict signature
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
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model)
        usage_body: Final = usage if isinstance(usage, dict) else None
        video_obj.usage = usage_body or {}
        return video_obj

    def _video_resource_url(self, api_base: str, video_id: str) -> str:
        encoded_video_id: Final = encode_url_path_segment(
            extract_original_video_id(video_id),
            field_name="video_id",
        )
        return f"{self._v1_root(api_base)}/videos/{encoded_video_id}"

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: provider JSON body and base-class dict signature
    ) -> tuple[str, dict]:  # mutable-ok: provider JSON body and base-class dict signature
        return self._video_resource_url(api_base, video_id), {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        response_data: Final = raw_response.json()
        status_raw: Final = str(response_data.get("status") or "processing").lower()
        status: Final = _STATUS_MAP.get(status_raw, status_raw)
        video_body: Final[object] = response_data.get("video")
        video_meta: Final[Mapping[str, object]] = video_body if _is_json_object(video_body) else {}
        url_value: Final = video_meta.get("url")
        video_url: Final = url_value if isinstance(url_value, str) else None
        duration: Final = video_meta.get("duration")
        seconds: Final = str(duration) if duration is not None else None
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
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, response_data.get("model"))
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: provider JSON body and base-class dict signature
        variant: str | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: provider JSON body and base-class dict signature
        return self._video_resource_url(api_base, video_id), {}

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
        raise ValueError(f"xAI video not ready for download (status={payload.get('status')}): {payload}")

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> bytes:
        url: Final = self._video_cdn_url(raw_response)
        if url is None:
            return raw_response.content
        video_response: Final = safe_get(litellm.module_level_client, url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> bytes:
        url: Final = self._video_cdn_url(raw_response)
        if url is None:
            return raw_response.content
        async_httpx_client: Final[AsyncHTTPHandler] = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.XAI,
        )
        video_response: Final = await async_safe_get(async_httpx_client, url)
        video_response.raise_for_status()
        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: provider JSON body and base-class dict signature
        extra_body: dict[str, object] | None = None,  # mutable-ok: provider JSON body and base-class dict signature
    ) -> tuple[str, dict]:  # mutable-ok: provider JSON body and base-class dict signature
        raise NotImplementedError("Video remix is not supported by xAI Imagine API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by xAI Imagine API")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: provider JSON body and base-class dict signature
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, object] | None = None,  # mutable-ok: provider JSON body and base-class dict signature
    ) -> tuple[str, dict]:  # mutable-ok: provider JSON body and base-class dict signature
        raise NotImplementedError("Video listing is not supported by xAI Imagine API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: provider JSON body and base-class dict signature
        raise NotImplementedError("Video listing is not supported by xAI Imagine API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: provider JSON body and base-class dict signature
    ) -> tuple[str, dict]:  # mutable-ok: provider JSON body and base-class dict signature
        raise NotImplementedError("Video delete is not supported by xAI Imagine API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by xAI Imagine API")
