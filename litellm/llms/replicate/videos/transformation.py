import math
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Final, TypeAlias

import httpx
from httpx._types import FileContent, RequestFiles

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.async_task_utils import unsupported_video_fields
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # shared HTTP factory lacks typed params
    get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # shared HTTP factory is private
)
from litellm.llms.replicate.common_utils import (
    ReplicateError,
    async_replicate_download,
    replicate_api_base,
    replicate_auth_headers,
    replicate_download,
    replicate_model_owner,
    replicate_output_urls,
    replicate_prediction,
    replicate_prediction_failure,
    replicate_prediction_input,
    replicate_prediction_request,
    replicate_prediction_string,
    replicate_prediction_url,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import LlmProviders
from litellm.types.videos.main import CharacterObject, VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider

_REPLICATE_PROVIDER: Final[str] = LlmProviders.REPLICATE.value
_STATUS_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {
        "starting": "queued",
        "processing": "in_progress",
        "succeeded": "completed",
        "failed": "failed",
        "canceled": "failed",
    }
)
_VideoParams: TypeAlias = dict[str, object]
_VideoHeaders: TypeAlias = dict[str, str]
_VideoStringParams: TypeAlias = dict[str, str]
_VideoFiles: TypeAlias = list[object]


@dataclass(frozen=True, slots=True)
class _ModelProfile:
    resolution_tiers: tuple[tuple[int, str], ...]
    aspect_ratio_key: str
    reference_key: str


_GENERIC_PROFILE: Final[_ModelProfile] = _ModelProfile(
    resolution_tiers=((480, "480p"), (720, "720p"), (1080, "1080p"), (sys.maxsize, "4k")),
    aspect_ratio_key="aspect_ratio",
    reference_key="image",
)
_MINIMAX_PROFILE: Final[_ModelProfile] = _ModelProfile(
    resolution_tiers=((480, "480P"), (768, "768P"), (1440, "2K"), (sys.maxsize, "4K")),
    aspect_ratio_key="ratio",
    reference_key="first_frame_image",
)


def _profile_for_model(model: str) -> _ModelProfile:
    return _MINIMAX_PROFILE if replicate_model_owner(model) == "minimax" else _GENERIC_PROFILE


def _duration_value(seconds: object) -> int | float:
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float, str)):
        raise TypeError("Replicate seconds must be a numeric value")
    try:
        duration: Final = float(seconds)
    except ValueError as error:
        raise ValueError("Replicate seconds must be a numeric value") from error
    return int(duration) if duration.is_integer() else duration


def _size_params(size: object, profile: _ModelProfile) -> Mapping[str, object]:
    if not isinstance(size, str):
        return MappingProxyType({})
    normalized: Final = size.strip().lower()
    named_resolution: Final = next(
        (resolution for _, resolution in profile.resolution_tiers if resolution.lower() == normalized), None
    )
    if named_resolution is not None:
        return MappingProxyType({"resolution": named_resolution})
    if normalized.count("x") != 1:
        return MappingProxyType({})
    width_text, height_text = normalized.split("x")
    if not (width_text.isdigit() and height_text.isdigit()) or int(width_text) <= 0 or int(height_text) <= 0:
        return MappingProxyType({})
    width: Final = int(width_text)
    height: Final = int(height_text)
    divisor: Final = math.gcd(width, height)
    short_side: Final = min(width, height)
    resolution: Final = next(
        (resolution for threshold, resolution in profile.resolution_tiers if short_side <= threshold),
        profile.resolution_tiers[-1][1],
    )
    return MappingProxyType(
        {"resolution": resolution, profile.aspect_ratio_key: f"{width // divisor}:{height // divisor}"}
    )


def _usage_from_input(prediction_input: Mapping[str, object]) -> _VideoParams:
    duration: Final = prediction_input.get("duration")
    resolution: Final = prediction_input.get("resolution")
    return {
        key: value
        for key, value in (
            (
                "duration_seconds",
                float(duration) if isinstance(duration, (int, float)) and not isinstance(duration, bool) else None,
            ),
            ("video_resolution", resolution if isinstance(resolution, str) else None),
        )
        if value is not None
    }


def _video_object(
    prediction: Mapping[str, object],
    model: str | None,
    custom_llm_provider: str | None,
    created_at: int,
) -> VideoObject:
    prediction_id: Final = replicate_prediction_string(prediction, "id") or ""
    failure: Final = replicate_prediction_failure(prediction)
    raw_status: Final = replicate_prediction_string(prediction, "status") or "starting"
    prediction_input: Final = replicate_prediction_input(prediction)
    duration: Final = prediction_input.get("duration")
    resolution: Final = prediction_input.get("resolution")
    return VideoObject(
        id=encode_video_id_with_provider(prediction_id, custom_llm_provider or _REPLICATE_PROVIDER, model),
        object="video",
        status="failed" if failure else _STATUS_MAP.get(raw_status, "queued"),
        created_at=created_at,
        model=model,
        seconds=str(duration) if isinstance(duration, (int, float)) and not isinstance(duration, bool) else None,
        size=resolution if isinstance(resolution, str) else None,
        error={"code": "replicate_error", "message": failure} if failure else None,
        usage=dict(_usage_from_input(prediction_input)),
    )


def _owned_prediction(raw_response: httpx.Response) -> Mapping[str, object]:
    prediction: Final = replicate_prediction(raw_response)
    requested: Final = raw_response.request.url.fragment
    if not requested or requested.startswith("deployments/"):
        return prediction
    requested_path, _, requested_version = requested.partition(":")
    actual_path: Final = replicate_prediction_string(prediction, "model")
    actual_version: Final = replicate_prediction_string(prediction, "version")
    if (actual_path is not None and actual_path != requested_path) or (
        requested_version and actual_version is not None and actual_version != requested_version
    ):
        raise ReplicateError(
            status_code=404,
            message=f"Replicate prediction {replicate_prediction_string(prediction, 'id')} does not belong to {requested}",
            headers=raw_response.headers,
        )
    return prediction


def _get_replicate_async_httpx_client() -> AsyncHTTPHandler:
    return get_async_httpx_client(llm_provider=LlmProviders.REPLICATE)


class ReplicateVideoConfig(BaseVideoConfig):
    def __init__(
        self,
        sync_client_factory: Callable[[], HTTPHandler] = get_httpx_client,
        async_client_factory: Callable[[], AsyncHTTPHandler] = _get_replicate_async_httpx_client,
    ) -> None:
        super().__init__()
        self._sync_client_factory: Final = sync_client_factory
        self._async_client_factory: Final = async_client_factory

    def get_supported_openai_params(
        self, model: str
    ) -> list[str]:  # mutable-ok: base class contract returns a plain list
        return ["model", "prompt", "input_reference", "seconds", "size", "extra_headers"]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> _VideoParams:
        unsupported: Final = unsupported_video_fields(
            video_create_optional_params,
            model,
            _REPLICATE_PROVIDER,
            drop_params,
            frozenset({"generate_audio", "aspect_ratio", "seed", "duration"}),
        )
        supported: Final = frozenset(self.get_supported_openai_params(model))
        profile: Final = _profile_for_model(model)
        input_reference: Final = video_create_optional_params.get("input_reference")
        if "input_reference" in video_create_optional_params and not isinstance(input_reference, str):
            raise ValueError("Replicate needs a public image URL for input_reference")
        reference_params: Final[Mapping[str, object]] = (
            MappingProxyType({profile.reference_key: input_reference})
            if isinstance(input_reference, str)
            else MappingProxyType({})
        )
        duration_params: Final[Mapping[str, object]] = (
            MappingProxyType({"duration": _duration_value(video_create_optional_params["seconds"])})
            if "seconds" in video_create_optional_params
            else MappingProxyType({})
        )
        size_params: Final[Mapping[str, object]] = (
            _size_params(video_create_optional_params["size"], profile)
            if "size" in video_create_optional_params
            else MappingProxyType({})
        )
        return {
            **reference_params,
            **duration_params,
            **size_params,
            **{
                key: value
                for key, value in video_create_optional_params.items()
                if key not in supported and key not in unsupported and key != "user"
            },
        }

    def validate_environment(
        self,
        headers: _VideoHeaders,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> _VideoHeaders:
        return replicate_auth_headers(
            headers, api_key or (litellm_params.api_key if litellm_params is not None else None)
        )

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: _VideoParams) -> str:
        return replicate_api_base(api_base)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: _VideoParams,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[_VideoParams, RequestFiles, str]:
        prediction_input: Final = {
            **{key: value for key, value in video_create_optional_request_params.items() if key != "model"},
            "prompt": prompt,
        }
        request: Final = replicate_prediction_request(api_base, model, prediction_input)
        return dict(request.body), [], request.url

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        return _video_object(replicate_prediction(raw_response), model, custom_llm_provider, int(time.time()))

    @staticmethod
    def _decode_video_id(video_id: str) -> tuple[str, str | None]:
        decoded: Final = decode_video_id_with_provider(video_id)
        return decoded.get("video_id", video_id), decoded.get("model_id")

    @classmethod
    def _prediction_url_for_video(cls, video_id: str, api_base: str) -> str:
        prediction_id, model_id = cls._decode_video_id(video_id)
        prediction_url: Final = httpx.URL(replicate_prediction_url(api_base, prediction_id))
        return str(prediction_url.copy_with(fragment=model_id) if model_id else prediction_url)

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        return self._prediction_url_for_video(video_id, api_base), {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        prediction: Final = _owned_prediction(raw_response)
        model: Final = raw_response.request.url.fragment or replicate_prediction_string(prediction, "model")
        return _video_object(prediction, model, custom_llm_provider, 0)

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        variant: str | None = None,
    ) -> tuple[str, _VideoStringParams]:
        return self._prediction_url_for_video(video_id, api_base), {}

    @staticmethod
    def _video_url(raw_response: httpx.Response) -> str:
        prediction: Final = _owned_prediction(raw_response)
        failure: Final = replicate_prediction_failure(prediction)
        if failure is not None:
            raise ReplicateError(status_code=raw_response.status_code, message=failure, headers=raw_response.headers)
        urls: Final = replicate_output_urls(prediction.get("output"))
        if not urls:
            raise ReplicateError(
                status_code=raw_response.status_code,
                message=f"Replicate prediction has no video output yet (status {replicate_prediction_string(prediction, 'status')})",
                headers=raw_response.headers,
            )
        return urls[0]

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        client: "HTTPHandler | None" = None,
    ) -> bytes:
        download_client: Final = client if client is not None else self._sync_client_factory()
        return replicate_download(download_client, self._video_url(raw_response)).content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        client: "AsyncHTTPHandler | None" = None,
    ) -> bytes:
        download_client: Final = client if client is not None else self._async_client_factory()
        return (await async_replicate_download(download_client, self._video_url(raw_response))).content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video remix is not supported for Replicate")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video remix is not supported for Replicate")

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
        raise NotImplementedError("video listing is not supported for Replicate")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> _VideoStringParams:
        raise NotImplementedError("video listing is not supported for Replicate")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video delete is not supported for Replicate")

    def transform_video_delete_response(self, raw_response: httpx.Response, logging_obj: object) -> VideoObject:
        raise NotImplementedError("video delete is not supported for Replicate")

    def transform_video_create_character_request(
        self,
        name: str,
        video: object,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoFiles]:
        raise NotImplementedError("video character creation is not supported for Replicate")

    def transform_video_create_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> CharacterObject:
        raise NotImplementedError("video character creation is not supported for Replicate")

    def transform_video_get_character_request(
        self,
        character_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: _VideoHeaders,
    ) -> tuple[str, _VideoParams]:
        raise NotImplementedError("video character retrieval is not supported for Replicate")

    def transform_video_get_character_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
    ) -> CharacterObject:
        raise NotImplementedError("video character retrieval is not supported for Replicate")

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
        raise NotImplementedError("video edit is not supported for Replicate")

    def transform_video_edit_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video edit is not supported for Replicate")

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
        raise NotImplementedError("video extension is not supported for Replicate")

    def transform_video_extension_response(
        self,
        raw_response: httpx.Response,
        logging_obj: object,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video extension is not supported for Replicate")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: _VideoHeaders | httpx.Headers,
    ) -> BaseLLMException:
        return ReplicateError(status_code=status_code, message=error_message, headers=headers)
