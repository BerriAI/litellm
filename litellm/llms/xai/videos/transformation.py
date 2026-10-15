import math
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING, Final, TypeAlias

import httpx
from httpx._types import RequestFiles
from pydantic import TypeAdapter
from typing_extensions import TypeIs  # noqa: TID251  # narrows untyped wire payloads without a runtime conversion

import litellm
from litellm.constants import XAI_API_BASE
from litellm.exceptions import AuthenticationError
from litellm.litellm_core_utils.url_utils import async_safe_get, encode_url_path_segment, safe_get
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory lacks typed params
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

_Params: TypeAlias = dict[str, object]
_Headers: TypeAlias = dict[str, object]
_SupportedParams: TypeAlias = list[str]
_EMPTY: Final[Mapping[str, object]] = {}
_JSON_PAYLOAD: Final = TypeAdapter(object)
_DEFAULT_DURATION: Final = 6


def _is_json_object(value: object) -> TypeIs[Mapping[str, object]]:  # guard-ok: provider JSON objects have str keys
    return isinstance(value, Mapping)


def _json_object(raw_response: httpx.Response) -> Mapping[str, object]:
    payload: Final = _JSON_PAYLOAD.validate_python(raw_response.json())
    return payload if _is_json_object(payload) else _EMPTY


def _optional_str(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


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


def _positive_seconds(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        seconds: Final = float(value)
    except ValueError:
        return None
    return seconds if math.isfinite(seconds) and seconds > 0 else None


def _duration_from_seconds(seconds: object) -> int:
    parsed: Final = _positive_seconds(seconds)
    return int(parsed) if parsed is not None else _DEFAULT_DURATION


def _requested_video_usage(request_data: Mapping[str, object]) -> _Params:
    resolution: Final = request_data.get("resolution") or request_data.get("resolution_name")
    return {
        key: value
        for key, value in (
            ("duration_seconds", _positive_seconds(request_data.get("duration"))),
            ("video_resolution", resolution if isinstance(resolution, str) and resolution else None),
        )
        if value is not None
    }


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
    def get_supported_openai_params(self, model: str) -> _SupportedParams:
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
    ) -> _Params:
        raw: Final = video_create_optional_params
        incoming: Final[Mapping[str, object]] = dict(raw)
        size: Final = incoming.get("size")
        use_duration: Final = "seconds" in incoming and "duration" not in incoming
        duration: Final = {"duration": _duration_from_seconds(incoming.get("seconds"))} if use_duration else None
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
        litellm_params: Mapping[str, object],
    ) -> str:
        from litellm.llms.xai.oauth import XAIOAuthAuthenticator, should_use_xai_oauth

        if should_use_xai_oauth(litellm_params) and not XAIModelInfo.get_api_key(api_key):
            return XAIOAuthAuthenticator().get_api_base().rstrip("/")

        resolved: Final = (
            api_base
            or _optional_str(litellm_params.get("api_base"))
            or get_secret_str("XAI_API_BASE")
            or get_secret_str("XAI_OAUTH_API_BASE")
            or XAI_API_BASE
        )
        return resolved.rstrip("/")

    def _v1_root(self, api_base: str) -> str:
        base: Final = api_base.rstrip("/")
        if base.endswith("/v1"):
            return base
        return f"{base}/v1"

    def _bearer_token(self, model: str, api_key: str | None, litellm_params: GenericLiteLLMParams | None) -> str:
        from litellm.llms.xai.oauth import (
            XAIOAuthAuthenticator,
            XAIOAuthError,
            should_use_xai_oauth,
        )

        dynamic_api_key: Final = XAIModelInfo.get_api_key(
            api_key or (litellm_params.api_key if litellm_params else None)
        )
        if dynamic_api_key:
            return dynamic_api_key
        if not should_use_xai_oauth(litellm_params.model_dump() if litellm_params is not None else {}):
            raise AuthenticationError(
                model=model or "xai-video",
                llm_provider="xai",
                message=(
                    "Missing xAI credentials for video generation. "
                    "Pass api_key / XAI_API_KEY, or set use_xai_oauth=True."
                ),
            )
        try:
            return XAIOAuthAuthenticator().get_access_token()
        except XAIOAuthError as exc:
            raise AuthenticationError(model=model or "xai-video", llm_provider="xai", message=str(exc)) from exc

    def validate_environment(
        self,
        headers: _Headers,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> _Headers:
        has_content_type: Final = "content-type" in headers or "Content-Type" in headers
        return {
            **headers,
            "Authorization": f"Bearer {self._bearer_token(model, api_key, litellm_params)}",
            **({} if has_content_type else {"Content-Type": "application/json"}),
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: _Params,
    ) -> str:
        resolved: Final = self._resolve_api_base(
            api_base=api_base,
            api_key=_optional_str(litellm_params.get("api_key")),
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
        video_create_optional_request_params: _Params,
        litellm_params: GenericLiteLLMParams,
        headers: _Headers,
    ) -> tuple[_Params, RequestFiles, str]:
        copied: Final = {
            key: video_create_optional_request_params[key]
            for key in (
                "image",
                "images",
                "duration",
                "resolution",
                "resolution_name",
                "aspect_ratio",
                "size",
            )
            if video_create_optional_request_params.get(key) is not None
        }
        prompt_body: Final = {"prompt": prompt} if prompt else None
        duration_body: Final = {"duration": _DEFAULT_DURATION} if "duration" not in copied else None
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
        request_data: _Params | None = None,
    ) -> VideoObject:
        response_data: Final = _json_object(raw_response)
        request_id: Final = _optional_str(response_data.get("request_id")) or _optional_str(response_data.get("id"))
        if not request_id:
            raise ValueError(f"xAI video generation response missing request_id: {response_data}")

        provider_usage: Final = response_data.get("usage")
        requested_usage: Final = _requested_video_usage(request_data or _EMPTY)
        duration_seconds: Final = requested_usage.get("duration_seconds")
        return VideoObject(
            id=encode_video_id_with_provider(request_id, custom_llm_provider, model)
            if custom_llm_provider
            else request_id,
            object="video",
            status="processing",
            created_at=int(time.time()),
            model=XAIModelInfo.get_base_model(model) or model,
            progress=0,
            seconds=f"{duration_seconds:g}" if isinstance(duration_seconds, float) else None,
            usage={**requested_usage, **(provider_usage if _is_json_object(provider_usage) else _EMPTY)},
        )

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
        headers: _Headers,
    ) -> tuple[str, _Params]:
        return self._video_resource_url(api_base, video_id), {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        response_data: Final = _json_object(raw_response)
        status_raw: Final = str(response_data.get("status") or "processing").lower()
        status: Final = _STATUS_MAP.get(status_raw, status_raw)
        video_body: Final = response_data.get("video")
        video_meta: Final = video_body if _is_json_object(video_body) else _EMPTY
        video_url: Final = _optional_str(video_meta.get("url"))
        duration: Final = video_meta.get("duration")
        model: Final = _optional_str(response_data.get("model"))
        provider_usage: Final = response_data.get("usage")
        request_id: Final = (
            _optional_str(response_data.get("request_id"))
            or _optional_str(response_data.get("id"))
            or (video_url.split("/")[-1].replace(".mp4", "") if video_url else "unknown")
        )
        video_obj: Final = VideoObject(
            id=encode_video_id_with_provider(request_id, custom_llm_provider, model)
            if custom_llm_provider and request_id != "unknown"
            else request_id,
            object="video",
            status=status,
            created_at=_optional_int(response_data.get("created_at")) or int(time.time()),
            completed_at=int(time.time()) if status == "completed" else None,
            model=model,
            progress=_optional_int(response_data.get("progress")),
            seconds=str(duration) if duration is not None else None,
            usage=dict(provider_usage) if _is_json_object(provider_usage) else {},
        )
        video_obj.hidden_params["video_url"] = video_url
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _Headers,
        variant: str | None = None,
    ) -> tuple[str, _Params]:
        return self._video_resource_url(api_base, video_id), {}

    def _video_cdn_url(self, raw_response: httpx.Response) -> str | None:
        content_type: Final = (raw_response.headers.get("content-type") or "").lower()
        if "application/json" not in content_type and raw_response.content[:1] != b"{":
            return None
        payload: Final = _JSON_PAYLOAD.validate_python(raw_response.json())
        if not _is_json_object(payload):
            return None
        video_body: Final = payload.get("video")
        url: Final = _optional_str(video_body.get("url")) if _is_json_object(video_body) else None
        if url:
            return url
        raise ValueError(f"xAI video not ready for download (status={payload.get('status')}): {payload}")

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        client: HTTPHandler | None = None,
    ) -> bytes:
        url: Final = self._video_cdn_url(raw_response)
        if url is None:
            return raw_response.content
        video_response: Final = safe_get(client or litellm.module_level_client, url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        client: AsyncHTTPHandler | None = None,
    ) -> bytes:
        url: Final = self._video_cdn_url(raw_response)
        if url is None:
            return raw_response.content
        async_httpx_client: Final[AsyncHTTPHandler] = client or get_async_httpx_client(
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
        headers: _Headers,
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, _Params]:
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
        headers: _Headers,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> tuple[str, _Params]:
        raise NotImplementedError("Video listing is not supported by xAI Imagine API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: base-class signature
        raise NotImplementedError("Video listing is not supported by xAI Imagine API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _Headers,
    ) -> tuple[str, _Params]:
        raise NotImplementedError("Video delete is not supported by xAI Imagine API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by xAI Imagine API")
