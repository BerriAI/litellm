"""
WaveSpeed AI video generation configuration.

WaveSpeed video models use the same prediction API as image models:

- ``POST {api_base}/api/v3/{model}`` creates the task
- ``GET {api_base}/api/v3/predictions/{id}/result`` reports status and, once complete, the output URLs

That maps onto the OpenAI video contract as create, status retrieve, and content download,
so the client polls the status endpoint instead of the provider blocking on a poll loop.

API Reference: https://wavespeed.ai/docs
"""

from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Any,  # noqa: TID251  # runtime stand-in for the TYPE_CHECKING-only logging type
    Final,
    Literal,
    Never,
    TypeAlias,
)

import httpx
from httpx._types import FileContent, RequestFiles
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm.litellm_core_utils.url_utils import async_safe_get, safe_get
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

from ..common_utils import (
    PENDING_STATUSES,
    WaveSpeedError,
    WaveSpeedPrediction,
    build_headers,
    build_result_url,
    build_submit_url,
    get_api_base,
    get_outputs,
    map_status_to_openai,
    optional_entry,
    optional_pair,
    to_reference_uri,
    to_request_payload,
    unwrap_envelope,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj: TypeAlias = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj: TypeAlias = Any


class _VideoObjectData(TypedDict, extra_items=object):
    id: ReadOnly[str]
    object: ReadOnly[Literal["video"]]
    status: ReadOnly[str]
    created_at: ReadOnly[int]


def _parse_created_at(created_at: str | None) -> int:
    if not created_at:
        return 0
    try:
        return int(datetime.fromisoformat(created_at.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return 0


def _to_error(prediction: WaveSpeedPrediction, error: str | None) -> Mapping[str, str] | None:
    if not error:
        return None
    return MappingProxyType({"code": prediction.get("status", "failed"), "message": error})


def _to_video_object(prediction: WaveSpeedPrediction, model: str | None) -> VideoObject:
    error: Final = prediction.get("error")
    video_data: Final[_VideoObjectData] = {
        "id": prediction.get("id", ""),
        "object": "video",
        "status": map_status_to_openai(prediction.get("status", "")),
        "created_at": _parse_created_at(prediction.get("created_at")),
        **optional_entry("model", model),
        **optional_entry("error", _to_error(prediction, error)),
    }
    return VideoObject(**video_data)


def _to_int_seconds(seconds: object) -> int | None:
    if seconds is None or isinstance(seconds, bool):
        return None
    if isinstance(seconds, (int, float)):
        return int(seconds)
    if isinstance(seconds, str):
        try:
            return int(float(seconds))
        except ValueError:
            return None
    return None


class WaveSpeedVideoConfig(BaseVideoConfig):
    """
    Configuration for WaveSpeed AI video generation.

    Any WaveSpeed video model id works as-is, e.g.
    ``wavespeed/bytedance/seedance-2.5/text-to-video``. Model-specific fields that have no
    OpenAI equivalent are passed straight through to the prediction body.
    """

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: base contract returns bare `list`
        return ["model", "prompt", "input_reference", "seconds", "size", "user", "extra_headers"]  # mutable-ok: ditto

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: base config contract returns bare `dict`
        supported: Final = frozenset(self.get_supported_openai_params(model))
        size: Final = video_create_optional_params.get("size")
        seconds: Final = _to_int_seconds(video_create_optional_params.get("seconds"))
        input_reference: Final = video_create_optional_params.get("input_reference")

        mapped_size: Final = size.lower().replace("x", "*") if isinstance(size, str) and "x" in size.lower() else None

        return to_request_payload(
            (
                *((k, v) for k, v in video_create_optional_params.items() if k not in supported),
                *optional_pair("image", to_reference_uri(input_reference) if input_reference else None),
                *optional_pair("size", mapped_size),
                *optional_pair("duration", seconds),
            )
        )

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:  # mutable-ok: base config contract returns bare `dict`
        resolved_key: Final = api_key or (litellm_params.api_key if litellm_params else None) or litellm.api_key
        return to_request_payload(MappingProxyType({**build_headers(resolved_key), **headers}))

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        return get_api_base(api_base)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict, RequestFiles, str]:  # mutable-ok: base config contract returns bare `dict`
        body: Final = to_request_payload(MappingProxyType({"prompt": prompt, **video_create_optional_request_params}))
        return body, [], build_submit_url(api_base, model)  # mutable-ok: httpx RequestFiles is a list

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        video_obj: Final = _to_video_object(unwrap_envelope(raw_response), model)
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model)
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict]:  # mutable-ok: base config contract returns bare `dict`
        return build_result_url(api_base, extract_original_video_id(video_id)), to_request_payload(MappingProxyType({}))

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        video_obj: Final = _to_video_object(unwrap_envelope(raw_response), None)
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, None)
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        variant: str | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: base config contract returns bare `dict`
        return build_result_url(api_base, extract_original_video_id(video_id)), to_request_payload(MappingProxyType({}))

    def _extract_output_url(self, raw_response: httpx.Response) -> str:
        prediction: Final = unwrap_envelope(raw_response)
        outputs: Final = get_outputs(prediction)
        if outputs:
            return outputs[0]

        status: Final = prediction.get("status", "")
        if status in PENDING_STATUSES:
            raise WaveSpeedError(
                status_code=409,
                message=f"WaveSpeed prediction {prediction.get('id', '')} is still {status}. Retry once it completes.",
            )
        raise WaveSpeedError(
            status_code=400,
            message=(
                f"WaveSpeed prediction {prediction.get('id', '')} has no video output "
                f"(status {status or 'unknown'}): {prediction.get('error') or 'no error detail returned'}"
            ),
        )

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        output_url: Final = self._extract_output_url(raw_response)
        video_response: Final = safe_get(litellm.module_level_client, output_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        output_url: Final = self._extract_output_url(raw_response)
        video_response: Final = await async_safe_get(litellm.module_level_aclient, output_url)
        video_response.raise_for_status()
        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        extra_body: Mapping[str, object] | None = None,
    ) -> Never:
        raise NotImplementedError("video remix is not supported for WaveSpeed")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> Never:
        raise NotImplementedError("video remix is not supported for WaveSpeed")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> Never:
        raise NotImplementedError("video listing is not supported for WaveSpeed")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> Never:
        raise NotImplementedError("video listing is not supported for WaveSpeed")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> Never:
        raise NotImplementedError("video delete is not supported for WaveSpeed")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Never:
        raise NotImplementedError("video delete is not supported for WaveSpeed")

    def transform_video_create_character_request(
        self,
        name: str,
        video: object,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> Never:
        raise NotImplementedError("video create character is not supported for WaveSpeed")

    def transform_video_create_character_response(
        self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj
    ) -> Never:
        raise NotImplementedError("video create character is not supported for WaveSpeed")

    def transform_video_get_character_request(
        self,
        character_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> Never:
        raise NotImplementedError("video get character is not supported for WaveSpeed")

    def transform_video_get_character_response(
        self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj
    ) -> Never:
        raise NotImplementedError("video get character is not supported for WaveSpeed")

    def transform_video_edit_request(
        self,
        prompt: str,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: matches BaseVideoConfig.transform_video_edit_request signature
        video_file: FileContent | None = None,
        extra_body: dict[str, Any] | None = None,  # mutable-ok: matches base signature
        prefetched_source_data: dict[str, Any] | None = None,  # mutable-ok: matches base signature
    ) -> Never:
        raise NotImplementedError("video edit is not supported for WaveSpeed")

    def transform_video_edit_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> Never:
        raise NotImplementedError("video edit is not supported for WaveSpeed")

    def transform_video_extension_request(
        self,
        prompt: str,
        video_id: str,
        seconds: str | None,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        extra_body: Mapping[str, object] | None = None,
    ) -> Never:
        raise NotImplementedError("video extension is not supported for WaveSpeed")

    def transform_video_extension_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> Never:
        raise NotImplementedError("video extension is not supported for WaveSpeed")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: matches BaseLLMException/base get_error_class contract
    ) -> WaveSpeedError:
        return WaveSpeedError(status_code=status_code, message=error_message, headers=headers)
