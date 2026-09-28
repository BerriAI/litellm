import math
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx
from httpx._types import FileContent, RequestFiles
from pydantic import TypeAdapter

from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # shared HTTP factory is private
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory lacks typed params
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import (
    CharacterObject,
    VideoCreateOptionalRequestParams,
    VideoObject,
)
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)


class FalAIVideoError(BaseLLMException):
    pass


_ALLOWED_ASPECT_RATIOS: Final[frozenset[str]] = frozenset({"auto", "16:9", "9:16", "1:1", "4:3", "3:4", "21:9"})


@dataclass(frozen=True, slots=True)
class _ModelProfile:
    resolutions: frozenset[str]
    resolution_tiers: tuple[tuple[int, str], ...]
    default_resolution: str
    integer_duration: bool
    reference_key: str
    reference_as_list: bool


_SEEDANCE_PROFILE: Final[_ModelProfile] = _ModelProfile(
    resolutions=frozenset({"480p", "720p", "1080p", "4k"}),
    resolution_tiers=((480, "480p"), (720, "720p"), (1080, "1080p"), (sys.maxsize, "4k")),
    default_resolution="720p",
    integer_duration=False,
    reference_key="image_url",
    reference_as_list=False,
)
_H3_PROFILE: Final[_ModelProfile] = _ModelProfile(
    resolutions=frozenset({"480P", "768P", "2K", "4K"}),
    resolution_tiers=((480, "480P"), (768, "768P"), (1440, "2K"), (sys.maxsize, "4K")),
    default_resolution="2K",
    integer_duration=True,
    reference_key="reference_image_urls",
    reference_as_list=True,
)
_QUEUE_NAMESPACES: Final[frozenset[str]] = frozenset(("workflows", "comfy"))
_STATUS_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "IN_QUEUE": "queued",
        "IN_PROGRESS": "in_progress",
        "COMPLETED": "completed",
    }
)
_FAL_AI_PROVIDER: Final[str] = LlmProviders.FAL_AI.value
_SupportedParams: TypeAlias = list[str]
_VideoParams: TypeAlias = dict[str, object]
_VideoHeaders: TypeAlias = dict[str, str]
_VideoStringParams: TypeAlias = dict[str, str]
_VideoFiles: TypeAlias = list[object]


def _queue_request_base_path(model: str) -> str:
    segments: Final[tuple[str, ...]] = tuple(model.split("/"))
    segment_count: Final[int] = 3 if segments and segments[0] in _QUEUE_NAMESPACES else 2
    return "/".join(segments[:segment_count])


def _duration_value(value: object) -> str | None:
    if isinstance(value, str) and value == "auto":
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        return str(int(float(value)))
    except (TypeError, ValueError):
        return None


def _profile_for_model(model: str) -> _ModelProfile:
    return _H3_PROFILE if model.startswith("minimax/h3/") else _SEEDANCE_PROFILE


def _resolution_for_short_side(short_side: int, profile: _ModelProfile) -> str:
    return next(
        (resolution for threshold, resolution in profile.resolution_tiers if short_side <= threshold),
        profile.resolution_tiers[-1][1],
    )


def _model_path_from_request_url(raw_response: httpx.Response) -> str | None:
    segments: Final[tuple[str, ...]] = tuple(segment for segment in raw_response.request.url.path.split("/") if segment)
    if "requests" not in segments:
        return None
    model_segments: Final[tuple[str, ...]] = segments[: segments.index("requests")]
    segment_count: Final[int] = 3 if len(model_segments) >= 3 and model_segments[-3] in _QUEUE_NAMESPACES else 2
    return "/".join(model_segments[-segment_count:]) if len(model_segments) >= segment_count else None


def _request_id_from_request_url(raw_response: httpx.Response) -> str | None:
    segments: Final[tuple[str, ...]] = tuple(segment for segment in raw_response.request.url.path.split("/") if segment)
    if "requests" not in segments:
        return None
    request_index: Final[int] = segments.index("requests")
    request_id_index: Final[int] = request_index + 1
    return segments[request_id_index] if len(segments) > request_id_index else None


def _size_params(size: object, profile: _ModelProfile) -> Mapping[str, str]:
    if not isinstance(size, str):
        return MappingProxyType({})
    normalized_size: Final[str] = size.lower()
    canonical_resolution: Final[str | None] = next(
        (resolution for resolution in profile.resolutions if resolution.lower() == normalized_size),
        None,
    )
    if canonical_resolution is not None:
        return MappingProxyType({"resolution": canonical_resolution})
    if normalized_size.count("x") != 1:
        return MappingProxyType({})
    width_text, height_text = normalized_size.split("x")
    if not (width_text.isdigit() and height_text.isdigit()):
        return MappingProxyType({})
    width: Final[int] = int(width_text)
    height: Final[int] = int(height_text)
    if width <= 0 or height <= 0:
        return MappingProxyType({})
    reduced_gcd: Final[int] = math.gcd(width, height)
    aspect_ratio: Final[str] = f"{width // reduced_gcd}:{height // reduced_gcd}"
    resolution: Final[str] = _resolution_for_short_side(min(width, height), profile)
    if aspect_ratio in _ALLOWED_ASPECT_RATIOS:
        return MappingProxyType({"resolution": resolution, "aspect_ratio": aspect_ratio})
    return MappingProxyType({"resolution": resolution})


def _numeric_duration(value: object) -> float | None:
    duration: Final[str | None] = _duration_value(value)
    if duration is None or duration == "auto":
        return None
    return float(duration)


def _response_data(raw_response: httpx.Response) -> Mapping[str, object]:
    return TypeAdapter(Mapping[str, object]).validate_python(raw_response.json())


def _response_data_or_none(raw_response: httpx.Response) -> Mapping[str, object] | None:
    try:
        return _response_data(raw_response)
    except ValueError:
        return None


def _detail_item_text(item: Mapping[str, object]) -> str | None:
    message: Final[object] = item.get("msg")
    if not isinstance(message, str):
        return None
    location: Final[object] = item.get("loc")
    if isinstance(location, str) and location:
        return f"{location}: {message}"
    if isinstance(location, (list, tuple)):
        location_parts: Final[tuple[str, ...]] = tuple(part for part in location if isinstance(part, str))
        if location_parts:
            return f"{'.'.join(location_parts)}: {message}"
    return message


def _error_text(response_data: Mapping[str, object]) -> str | None:
    detail: Final[object] = response_data.get("detail")
    if isinstance(detail, str):
        return detail
    if isinstance(detail, list):
        detail_items: Final[tuple[Mapping[str, object], ...]] = tuple(
            item for item in detail if isinstance(item, Mapping)
        )
        detail_messages: Final[tuple[str, ...]] = tuple(
            message for item in detail_items if (message := _detail_item_text(item)) is not None
        )
        if detail_messages:
            return "; ".join(detail_messages)
    error: Final[object] = response_data.get("error")
    return error if isinstance(error, str) else None


def _result_error(raw_response: httpx.Response) -> str | None:
    if raw_response.is_success:
        return None
    response_data: Final[Mapping[str, object] | None] = _response_data_or_none(raw_response)
    error_text: Final[str | None] = _error_text(response_data) if response_data is not None else None
    if error_text:
        return error_text
    response_text: Final[str] = raw_response.text
    return response_text or f"fal.ai returned HTTP {raw_response.status_code}"


def _terminal_result_error(raw_response: httpx.Response) -> str | None:
    if raw_response.status_code == 429 or raw_response.status_code >= 500:
        return None
    return _result_error(raw_response)


def _get_fal_ai_async_httpx_client() -> AsyncHTTPHandler:
    return get_async_httpx_client(llm_provider=LlmProviders.FAL_AI)


def _response_string(response_data: Mapping[str, object], key: str, default: str = "") -> str:
    value: Final[object] = response_data.get(key)
    return value if isinstance(value, str) else default


_RESULT_HEADERS_NOT_FORWARDED: Final[frozenset[str]] = frozenset({"host", "content-length", "transfer-encoding"})


def _result_request(
    raw_response: httpx.Response,
    response_data: Mapping[str, object],
) -> tuple[str, Mapping[str, str]] | None:
    if _response_string(response_data, "status", "IN_QUEUE") != "COMPLETED":
        return None
    result_url: Final[str] = str(raw_response.request.url).removesuffix("/status")
    encoding: Final[str] = raw_response.request.headers.encoding
    result_headers: Final[Mapping[str, str]] = MappingProxyType(
        {
            key.decode(encoding): value.decode(encoding)
            for key, value in raw_response.request.headers.raw
            if key.decode(encoding).lower() not in _RESULT_HEADERS_NOT_FORWARDED
        }
    )
    return result_url, result_headers


def _status_video_object(
    response_data: Mapping[str, object],
    raw_response: httpx.Response,
    custom_llm_provider: str | None,
    result_error: str | None,
) -> VideoObject:
    raw_status: Final[str] = _response_string(response_data, "status", "IN_QUEUE")
    status: Final[str] = _STATUS_MAP.get(raw_status, "queued")
    status_error: Final[str | None] = _error_text(response_data)
    error: Final[str | None] = result_error if result_error is not None else status_error
    provider: Final[str] = custom_llm_provider or _FAL_AI_PROVIDER
    model_path: Final[str | None] = _model_path_from_request_url(raw_response)
    request_id: Final[str] = _response_string(response_data, "request_id") or (
        _request_id_from_request_url(raw_response) or ""
    )
    return VideoObject(
        id=encode_video_id_with_provider(request_id, provider, model_path),
        object="video",
        status="failed" if error else status,
        created_at=0,
        model=model_path,
        error=(
            {"code": "fal_error", "message": error} if error else None  # mutable-ok: VideoObject requires a dict
        ),
    )


class FalAIVideoConfig(BaseVideoConfig):
    def __init__(
        self,
        sync_client_factory: Callable[[], HTTPHandler] = _get_httpx_client,
        async_client_factory: Callable[[], AsyncHTTPHandler] = _get_fal_ai_async_httpx_client,
    ) -> None:
        super().__init__()
        self._sync_client_factory: Final = sync_client_factory
        self._async_client_factory: Final = async_client_factory

    def get_supported_openai_params(self, model: str) -> _SupportedParams:
        supported_params: Final[_SupportedParams] = [  # mutable-ok: BaseVideoConfig requires a list
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]
        return supported_params

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> _VideoParams:
        supported_params: Final[frozenset[str]] = frozenset(self.get_supported_openai_params(model))
        input_reference: Final[object] = video_create_optional_params.get("input_reference")
        if "input_reference" in video_create_optional_params and not isinstance(input_reference, str):
            raise ValueError("fal.ai needs a public image URL for input_reference")
        profile: Final[_ModelProfile] = _profile_for_model(model)
        input_reference_params: Final[Mapping[str, object]] = (
            MappingProxyType({})
            if not isinstance(input_reference, str)
            else MappingProxyType(
                {
                    profile.reference_key: (
                        [input_reference]  # mutable-ok: fal.ai expects a list for H3 references
                        if profile.reference_as_list
                        else input_reference
                    ),
                }
            )
        )
        duration_params: Final[Mapping[str, object]] = (
            MappingProxyType({})
            if "seconds" not in video_create_optional_params
            else self._duration_params(video_create_optional_params["seconds"], profile)
        )
        size_params: Final[Mapping[str, str]] = (
            _size_params(video_create_optional_params["size"], profile)
            if "size" in video_create_optional_params
            else MappingProxyType({})
        )
        user_params: Final[Mapping[str, str]] = (
            MappingProxyType({"end_user_id": user})
            if isinstance(user := video_create_optional_params.get("user"), str)
            else MappingProxyType({})
        )
        mapped_params: Final[_VideoParams] = {
            **input_reference_params,
            **duration_params,
            **size_params,
            **user_params,
            **{  # mutable-ok: BaseVideoConfig requires a mutable parameter mapping
                key: value for key, value in video_create_optional_params.items() if key not in supported_params
            },
        }
        return mapped_params

    @staticmethod
    def _duration_params(seconds: object, profile: _ModelProfile) -> Mapping[str, object]:
        duration: Final[str | None] = _duration_value(seconds)
        if duration is None:
            raise ValueError("fal.ai seconds must be a numeric value")
        if duration == "auto":
            return MappingProxyType({}) if profile.integer_duration else MappingProxyType({"duration": duration})
        return MappingProxyType({"duration": int(duration) if profile.integer_duration else duration})

    def validate_environment(
        self,
        headers: _VideoHeaders,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> _VideoHeaders:
        final_api_key: Final[str | None] = (
            api_key
            or (litellm_params.api_key if litellm_params is not None else None)
            or get_secret_str("FAL_AI_API_KEY")
        )
        if not final_api_key:
            raise ValueError("FAL_AI_API_KEY is not set")
        validated_headers: Final[_VideoHeaders] = {
            **headers,
            "Authorization": f"Key {final_api_key}",
            "Content-Type": "application/json",
        }
        return validated_headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: _VideoParams,
    ) -> str:
        return (api_base or "https://queue.fal.run").rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: _VideoParams,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[_VideoParams, RequestFiles, str]:
        request_data: Final[_VideoParams] = {
            "prompt": prompt,
            **{  # mutable-ok: HTTP JSON payload requires a mutable mapping
                key: value for key, value in video_create_optional_request_params.items() if key != "model"
            },
        }
        return request_data, [], f"{api_base.rstrip('/')}/{model}"  # mutable-ok: HTTP files payload requires a list

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        response_data: Final[Mapping[str, object]] = _response_data(raw_response)
        profile: Final[_ModelProfile] = _profile_for_model(model)
        request_params: Final[Mapping[str, object]] = request_data or MappingProxyType({})
        request_id: Final[str] = _response_string(response_data, "request_id")
        provider: Final[str] = custom_llm_provider or _FAL_AI_PROVIDER
        duration: Final[float | None] = _numeric_duration(request_params.get("duration"))
        resolution: Final[object] = request_params.get("resolution")
        seconds: Final[str | None] = _duration_value(request_params["duration"]) if duration is not None else None
        size: Final[str | None] = resolution if isinstance(resolution, str) else None
        usage: Final[_VideoParams] = {  # mutable-ok: VideoObject requires a mutable usage mapping
            key: value
            for key, value in (
                ("duration_seconds", duration),
                (
                    "video_resolution",
                    resolution if isinstance(resolution, str) else profile.default_resolution,
                ),
            )
            if value is not None
        }
        video_object: Final[VideoObject] = VideoObject(
            id=encode_video_id_with_provider(request_id, provider, model),
            object="video",
            status="queued",
            created_at=int(time.time()),
            model=model,
            seconds=seconds,
            size=size,
        )
        video_object.usage = usage
        return video_object

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        request_id, model_id = self._decode_video_id(video_id)
        encoded_request_id: Final[str] = encode_url_path_segment(request_id, field_name="video_id")
        return (
            f"{api_base.rstrip('/')}/{_queue_request_base_path(model_id)}/requests/{encoded_request_id}/status",
            {},  # mutable-ok: BaseVideoConfig requires a mutable mapping
        )

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        response_data: Final[Mapping[str, object]] = _response_data(raw_response)
        result_error: Final[str | None] = self._fetch_result_error(raw_response, response_data, client)
        return _status_video_object(
            response_data=response_data,
            raw_response=raw_response,
            custom_llm_provider=custom_llm_provider,
            result_error=result_error,
        )

    def _fetch_result_error(
        self,
        raw_response: httpx.Response,
        response_data: Mapping[str, object],
        client: HTTPHandler | None,
    ) -> str | None:
        result_request: Final[tuple[str, Mapping[str, str]] | None] = _result_request(raw_response, response_data)
        if result_request is None:
            return None
        result_url, result_headers = result_request
        result_client: Final[HTTPHandler] = client if client is not None else self._sync_client_factory()
        try:
            result_response: Final[httpx.Response] = result_client.get(
                url=result_url,
                headers=dict(result_headers),  # mutable-ok: HTTPHandler.get only accepts a dict
            )
        except httpx.TransportError:
            return None
        return _terminal_result_error(result_response)

    async def async_transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> VideoObject:
        response_data: Final[Mapping[str, object]] = _response_data(raw_response)
        result_error: Final[str | None] = await self._fetch_result_error_async(raw_response, response_data, client)
        return _status_video_object(
            response_data=response_data,
            raw_response=raw_response,
            custom_llm_provider=custom_llm_provider,
            result_error=result_error,
        )

    async def _fetch_result_error_async(
        self,
        raw_response: httpx.Response,
        response_data: Mapping[str, object],
        client: AsyncHTTPHandler | None,
    ) -> str | None:
        result_request: Final[tuple[str, Mapping[str, str]] | None] = _result_request(raw_response, response_data)
        if result_request is None:
            return None
        result_url, result_headers = result_request
        result_client: Final[AsyncHTTPHandler] = client if client is not None else self._async_client_factory()
        try:
            result_response: Final[httpx.Response] = await result_client.get(
                url=result_url,
                headers=dict(result_headers),  # mutable-ok: AsyncHTTPHandler.get only accepts a dict
            )
        except httpx.TransportError:
            return None
        return _terminal_result_error(result_response)

    @staticmethod
    def _decode_video_id(video_id: str) -> tuple[str, str]:
        decoded: Final = decode_video_id_with_provider(video_id)
        request_id: Final[str] = decoded.get("video_id", video_id)
        model_id: Final[str | None] = decoded.get("model_id")
        if not model_id:
            raise ValueError("fal.ai video ids must be created through litellm with a model")
        return request_id, model_id

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        variant: str | None = None,
    ) -> tuple[str, _VideoStringParams]:
        request_id, model_id = self._decode_video_id(video_id)
        encoded_request_id: Final[str] = encode_url_path_segment(request_id, field_name="video_id")
        return (
            f"{api_base.rstrip('/')}/{_queue_request_base_path(model_id)}/requests/{encoded_request_id}",
            {},  # mutable-ok: BaseVideoConfig requires a mutable mapping
        )

    @staticmethod
    def _extract_video_url(response_data: Mapping[str, object]) -> str:
        raw_video_data: Final[object] = response_data.get("video")
        video_data: Final[Mapping[str, object] | None] = (
            TypeAdapter(Mapping[str, object]).validate_python(raw_video_data)
            if isinstance(raw_video_data, Mapping)
            else None
        )
        if video_data is not None:
            video_url: Final[object] = video_data.get("url")
            if isinstance(video_url, str) and video_url:
                return video_url
        error_message: Final[str | None] = _error_text(response_data)
        if error_message:
            raise ValueError(f"fal.ai video result did not include a video URL: {error_message}")
        raise ValueError("fal.ai video result did not include a video URL")

    def transform_video_content_response(self, raw_response: httpx.Response, logging_obj: object) -> bytes:
        error: Final[str | None] = _result_error(raw_response)
        if error is not None:
            raise FalAIVideoError(
                status_code=raw_response.status_code,
                message=error,
                headers=dict(raw_response.headers),  # mutable-ok: exception headers require a mutable dictionary
                request=raw_response.request,
                response=raw_response,
            )
        video_url: Final[str] = self._extract_video_url(_response_data(raw_response))
        httpx_client: Final[HTTPHandler] = self._sync_client_factory()
        video_response: Final[httpx.Response] = httpx_client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
            video_url
        )
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(self, raw_response: httpx.Response, logging_obj: object) -> bytes:
        error: Final[str | None] = _result_error(raw_response)
        if error is not None:
            raise FalAIVideoError(
                status_code=raw_response.status_code,
                message=error,
                headers=dict(raw_response.headers),  # mutable-ok: exception headers require a mutable dictionary
                request=raw_response.request,
                response=raw_response,
            )
        video_url: Final[str] = self._extract_video_url(_response_data(raw_response))
        async_httpx_client: Final[AsyncHTTPHandler] = self._async_client_factory()
        video_response: Final[httpx.Response] = await async_httpx_client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTP handler stubs are untyped
            video_url
        )
        video_response.raise_for_status()
        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video remix is not supported for fal.ai")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video remix is not supported for fal.ai")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video listing is not supported for fal.ai")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> _VideoStringParams:
        raise NotImplementedError("video listing is not supported for fal.ai")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video delete is not supported for fal.ai")

    def transform_video_delete_response(self, raw_response: httpx.Response, logging_obj: object) -> VideoObject:
        raise NotImplementedError("video delete is not supported for fal.ai")

    def transform_video_create_character_request(
        self,
        name: str,
        video: object,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoFiles]:
        raise NotImplementedError("video character creation is not supported for fal.ai")

    def transform_video_create_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> CharacterObject:
        raise NotImplementedError("video character creation is not supported for fal.ai")

    def transform_video_get_character_request(
        self,
        character_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video character retrieval is not supported for fal.ai")

    def transform_video_get_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> CharacterObject:
        raise NotImplementedError("video character retrieval is not supported for fal.ai")

    def transform_video_edit_request(
        self,
        prompt: str,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        video_file: FileContent | None = None,
        extra_body: Mapping[str, object] | None = None,
        prefetched_source_data: Mapping[str, object] | None = None,
    ) -> tuple[str, Mapping[str, object], RequestFiles | None]:
        raise NotImplementedError("video edit is not supported for fal.ai")

    def transform_video_edit_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video edit is not supported for fal.ai")

    def transform_video_extension_request(
        self,
        prompt: str,
        video_id: str,
        seconds: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video extension is not supported for fal.ai")

    def transform_video_extension_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video extension is not supported for fal.ai")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: _VideoHeaders | httpx.Headers,
    ) -> BaseLLMException:
        return FalAIVideoError(status_code=status_code, message=error_message, headers=headers)
