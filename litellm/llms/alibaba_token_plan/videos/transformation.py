from __future__ import annotations

from collections.abc import Mapping
from math import gcd
from typing import TYPE_CHECKING, Final, Literal

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, HttpUrl, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.url_utils import async_safe_get, encode_url_path_segment, safe_get
from litellm.llms.alibaba_token_plan.common_utils import VIDEO_PATH, get_api_url, image_reference, validate_headers
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

_STATUSES: Final = {"PENDING": "queued", "RUNNING": "in_progress", "SUCCEEDED": "completed"}
_SUPPORTED_PARAMS: Final = ("seconds", "size", "input_reference", "input", "parameters")


class _Output(BaseModel, frozen=True):
    task_id: str
    task_status: str
    video_url: HttpUrl | None = None
    code: str | None = None
    message: str | None = None


class _Usage(BaseModel, frozen=True):
    duration: int | None = None
    output_video_duration: int | None = None
    SR: int | None = None


class _Result(BaseModel, frozen=True):
    output: _Output
    usage: _Usage | None = None


class _VideoContext(BaseModel, frozen=True):
    video_id: str | None = None


def _size_parameters(size: object) -> Mapping[str, str]:
    if not isinstance(size, str):
        return {}
    width, height = (int(dimension) for dimension in size.lower().split("x"))
    divisor: Final = gcd(width, height)
    return {"resolution": f"{min(width, height)}P", "ratio": f"{width // divisor}:{height // divisor}"}


class AlibabaTokenPlanVideoConfig(BaseVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base provider interface
        return list(_SUPPORTED_PARAMS)

    def map_openai_params(
        self, video_create_optional_params: VideoCreateOptionalRequestParams, model: str, drop_params: bool
    ) -> dict[str, object]:  # mutable-ok: base provider interface
        return {key: value for key, value in video_create_optional_params.items() if key in _SUPPORTED_PARAMS}

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:  # mutable-ok: base provider interface
        resolved_key: Final = api_key or (litellm_params.api_key if litellm_params else None)
        return {**validate_headers(headers, resolved_key), **({"X-DashScope-Async": "enable"} if model else {})}

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return get_api_url(api_base, "").rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, object], RequestFiles, str]:  # mutable-ok: base provider interface
        params: Final = video_create_optional_request_params
        reference: Final = params.get("input_reference")
        media: Final = (
            {
                "media": [
                    {
                        "type": "first_frame" if model.endswith("-i2v") else "reference_image",
                        "url": image_reference(reference),
                    }
                ]
            }
            if reference is not None
            else {}
        )
        seconds: Final = params.get("seconds")
        native_input: Final = TypeAdapter(dict[str, object]).validate_python(params.get("input") or {})
        native_parameters: Final = TypeAdapter(dict[str, object]).validate_python(params.get("parameters") or {})
        return (
            {
                "model": model,
                "input": {**media, **native_input, "prompt": prompt},
                "parameters": {
                    **_size_parameters(params.get("size")),
                    **({"duration": int(str(seconds))} if seconds is not None else {}),
                    **native_parameters,
                },
            },
            {},
            f"{api_base}/{VIDEO_PATH}",
        )

    def _video(self, raw_response: httpx.Response, model: str | None) -> VideoObject:
        if not raw_response.is_success:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        try:
            result: Final = _Result.model_validate_json(raw_response.content)
        except ValidationError:
            raise self.get_error_class(raw_response.text, 502, raw_response.headers) from None
        output: Final = result.output
        status: Final = _STATUSES.get(output.task_status, "failed")
        duration: Final = (result.usage.output_video_duration or result.usage.duration) if result.usage else None
        return VideoObject(
            id=encode_video_id_with_provider(output.task_id, "alibaba_token_plan", model),
            object="video",
            status=status,
            model=model,
            progress=100 if status == "completed" else None,
            seconds=str(duration) if duration is not None else None,
            error={"code": output.code or output.task_status, "message": output.message or output.task_status}
            if status == "failed"
            else None,
            usage={
                **({"duration_seconds": duration} if duration is not None else {}),
                **({"video_resolution": f"{result.usage.SR}p"} if result.usage and result.usage.SR else {}),
            },
        )

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        return self._video(raw_response, model)

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: base provider interface
        task_id: Final = encode_url_path_segment(extract_original_video_id(video_id), field_name="video_id")
        return f"{api_base}/api/v1/tasks/{task_id}", {}

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        client: HTTPHandler | None = None,
    ) -> VideoObject:
        context: Final = _VideoContext.model_validate(logging_obj.model_call_details)
        model: Final = decode_video_id_with_provider(context.video_id).get("model_id") if context.video_id else None
        return self._video(raw_response, model)

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        variant: str | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: base provider interface
        return self.transform_video_status_retrieve_request(video_id, api_base, litellm_params, headers)

    def _video_url(self, raw_response: httpx.Response) -> str:
        video: Final = self._video(raw_response, None)
        url: Final = _Result.model_validate_json(raw_response.content).output.video_url
        if video.status != "completed" or url is None:
            raise self.get_error_class(f"Video content is not available: {video.status}", 409, raw_response.headers)
        return str(url)

    def _downloaded(self, video_response: httpx.Response) -> bytes:
        if not video_response.is_success:
            raise self.get_error_class("Alibaba Token Plan video download failed", video_response.status_code, {})
        return video_response.content

    def transform_video_content_response(self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj) -> bytes:
        video_url: Final = self._video_url(raw_response)
        try:
            return self._downloaded(safe_get(litellm.module_level_client, video_url))
        except httpx.HTTPError:
            raise self.get_error_class("Alibaba Token Plan video download failed", 502, {}) from None

    async def async_transform_video_content_response(
        self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj
    ) -> bytes:
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        video_url: Final = self._video_url(raw_response)
        client: Final = get_async_httpx_client(llm_provider=litellm.LlmProviders.ALIBABA_TOKEN_PLAN)
        try:
            return self._downloaded(await async_safe_get(client, video_url))
        except httpx.HTTPError:
            raise self.get_error_class("Alibaba Token Plan video download failed", 502, {}) from None

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(status_code=status_code, message=error_message, headers=dict(headers))

    def _unsupported(self, operation: Literal["remix", "list", "delete"]) -> NotImplementedError:
        return NotImplementedError(f"Alibaba Token Plan does not support video {operation}")

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: base provider interface
        raise self._unsupported("remix")

    def transform_video_remix_response(
        self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj, custom_llm_provider: str | None = None
    ) -> VideoObject:
        raise self._unsupported("remix")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: base provider interface
        raise self._unsupported("list")

    def transform_video_list_response(
        self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj, custom_llm_provider: str | None = None
    ) -> dict[str, str]:  # mutable-ok: base provider interface
        raise self._unsupported("list")

    def transform_video_delete_request(
        self, video_id: str, api_base: str, litellm_params: GenericLiteLLMParams, headers: Mapping[str, str]
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: base provider interface
        raise self._unsupported("delete")

    def transform_video_delete_response(
        self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj
    ) -> VideoObject:
        raise self._unsupported("delete")
