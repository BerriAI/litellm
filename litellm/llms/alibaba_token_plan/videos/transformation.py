from __future__ import annotations

import base64
from collections.abc import Mapping
from datetime import datetime, timedelta, timezone
from math import gcd
from os import PathLike
from pathlib import Path
from typing import TYPE_CHECKING, Final, Literal, Protocol, runtime_checkable
from urllib.parse import urlsplit, urlunsplit

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, Field, HttpUrl, TypeAdapter, ValidationError

import litellm
from litellm.litellm_core_utils.token_counter import get_image_type
from litellm.litellm_core_utils.url_utils import async_safe_get, encode_url_path_segment, safe_get
from litellm.llms.alibaba_token_plan.common_utils import VIDEO_ENDPOINT, get_native_api_url, validate_headers
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


class _Media(BaseModel, frozen=True, extra="forbid"):
    type: Literal["first_frame", "reference_image"]
    url: str


class _Input(BaseModel, frozen=True, extra="forbid"):
    prompt: str | None = None
    media: tuple[_Media, ...] = ()


class _Parameters(BaseModel, frozen=True, extra="forbid"):
    resolution: Literal["480P", "720P", "1080P"] | None = None
    ratio: Literal["16:9", "9:16", "1:1", "4:3", "3:4", "4:5", "5:4", "9:21", "21:9"] | None = None
    duration: int | None = Field(default=None, ge=3, le=15)
    watermark: bool | None = None
    seed: int | None = Field(default=None, ge=0, le=2147483647)


class _Output(BaseModel, frozen=True):
    task_id: str
    task_status: Literal["PENDING", "RUNNING", "SUCCEEDED", "FAILED", "CANCELED", "UNKNOWN"]
    submit_time: str | None = None
    end_time: str | None = None
    video_url: HttpUrl | None = None
    code: str | None = None
    message: str | None = None


class _Usage(BaseModel, frozen=True):
    duration: int | None = None
    output_video_duration: int | None = None
    video_count: int | None = None
    SR: int | None = None
    ratio: str | None = None


class _Result(BaseModel, frozen=True):
    output: _Output | None = None
    usage: _Usage | None = None
    code: str | None = None
    message: str | None = None


class _VideoContext(BaseModel, frozen=True):
    video_id: str | None = None


@runtime_checkable
class _BinaryReader(Protocol):
    def tell(self) -> int: ...

    def seek(self, offset: int, whence: int = 0, /) -> int: ...

    def read(self, n: int = -1, /) -> bytes: ...


def _timestamp(value: str | None) -> int | None:
    if value is None:
        return None
    parsed: Final = datetime.fromisoformat(value)
    localized: Final = parsed.replace(tzinfo=timezone(timedelta(hours=8))) if parsed.tzinfo is None else parsed
    return int(localized.timestamp())


def _operation_url(api_base: str, endpoint: str) -> str:
    parsed: Final = urlsplit(api_base)
    path: Final = f"{parsed.path.rstrip('/')}/{endpoint.lstrip('/')}"
    return urlunsplit((parsed.scheme, parsed.netloc, path, parsed.query, parsed.fragment))


def _reference_url(reference: object) -> str:
    if isinstance(reference, str):
        if urlsplit(reference).scheme in ("http", "https") or reference.startswith("data:image/"):
            return reference
        raise ValueError("Video reference must be a public HTTP(S) URL or an image data URI")
    if isinstance(reference, tuple):
        parts: Final = TypeAdapter(tuple[object, ...]).validate_python(reference)
        if len(parts) >= 2:
            return _reference_url(parts[1])
        raise ValueError("Video reference file tuples must include file content")
    if isinstance(reference, PathLike):
        path: Final = TypeAdapter(str).validate_python(reference.__fspath__())
        with Path(path).open("rb") as image_file:
            return _reference_url(image_file)
    if isinstance(reference, _BinaryReader):
        position: Final = reference.tell()
        try:
            reference.seek(0)
            return _reference_url(reference.read(20 * 1024 * 1024 + 1))
        finally:
            reference.seek(position)
    if not isinstance(reference, bytes):
        raise TypeError("Video reference must be image bytes, a binary file, a public URL, or an image data URI")
    if len(reference) > 20 * 1024 * 1024:
        raise ValueError("Video reference images must not exceed 20 MB")
    image_type: Final = get_image_type(reference)
    if image_type not in ("jpeg", "png", "webp"):
        raise ValueError("Video reference images must be JPEG, PNG, or WebP")
    return f"data:image/{image_type};base64,{base64.b64encode(reference).decode('ascii')}"


def _size_parameters(size: object, model: str) -> Mapping[str, str]:
    if size is None:
        return {}
    if model.endswith("-i2v"):
        raise ValueError("HappyHorse image-to-video preserves the image aspect ratio; use resolution instead of size")
    if not isinstance(size, str):
        raise TypeError("Video size must be width x height")
    width, height = (int(dimension) for dimension in size.split("x"))
    divisor: Final = gcd(width, height)
    if divisor == 0 or min(width, height) not in (480, 720, 1080):
        raise ValueError("Use native parameters.resolution and parameters.ratio for this video size")
    return {"resolution": f"{min(width, height)}P", "ratio": f"{width // divisor}:{height // divisor}"}


class AlibabaTokenPlanVideoConfig(BaseVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: base provider interface
        return ["input_reference", "seconds", "size", "extra_headers"]

    def map_openai_params(
        self, video_create_optional_params: VideoCreateOptionalRequestParams, model: str, drop_params: bool
    ) -> dict[str, object]:  # mutable-ok: base provider interface
        supported: Final = frozenset(self.get_supported_openai_params(model)) | {
            "model",
            "input",
            "parameters",
            "resolution",
            "extra_body",
        }
        unsupported: Final = tuple(key for key in video_create_optional_params if key not in supported)
        if unsupported and not (drop_params or litellm.drop_params):
            raise litellm.UnsupportedParamsError(
                message=f"Alibaba Token Plan video does not support: {', '.join(unsupported)}",
                model=model,
                llm_provider="alibaba_token_plan",
            )
        return {key: value for key, value in video_create_optional_params.items() if key in supported}

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
        resolved_base: Final = urlsplit(get_native_api_url(api_base, ""))
        return urlunsplit(
            (
                resolved_base.scheme,
                resolved_base.netloc,
                resolved_base.path.rstrip("/").removesuffix(f"/{VIDEO_ENDPOINT}"),
                resolved_base.query,
                resolved_base.fragment,
            )
        )

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, object], RequestFiles, str]:  # mutable-ok: base provider interface
        if model not in ("happyhorse-1.1-t2v", "happyhorse-1.1-i2v", "happyhorse-1.1-r2v"):
            raise ValueError(f"Unsupported Alibaba Token Plan video model: {model}")
        native_input: Final = _Input.model_validate(video_create_optional_request_params.get("input") or {})
        reference: Final = video_create_optional_request_params.get("input_reference")
        if reference is not None and native_input.media:
            raise ValueError("Provide either input_reference or input.media")
        media: Final = (
            (
                _Media(
                    type="first_frame" if model.endswith("-i2v") else "reference_image", url=_reference_url(reference)
                ),
            )
            if reference is not None
            else native_input.media
        )
        if model.endswith("-i2v") and (len(media) != 1 or media[0].type != "first_frame"):
            raise ValueError("HappyHorse image-to-video requires exactly one first_frame in input.media")
        if model.endswith("-r2v") and (
            not 1 <= len(media) <= 9 or any(item.type != "reference_image" for item in media)
        ):
            raise ValueError("HappyHorse reference-to-video requires 1 to 9 reference_image entries in input.media")
        if model.endswith("-t2v") and media:
            raise ValueError("HappyHorse text-to-video does not support reference images")
        resolved_prompt: Final = native_input.prompt if native_input.prompt is not None else prompt
        if not resolved_prompt and not model.endswith("-i2v"):
            raise ValueError("HappyHorse text-to-video and reference-to-video require a prompt")
        native_parameters: Final = _Parameters.model_validate(
            video_create_optional_request_params.get("parameters") or {}
        )
        seconds: Final = video_create_optional_request_params.get("seconds")
        resolution: Final = video_create_optional_request_params.get("resolution")
        parameters: Final = _Parameters.model_validate(
            {
                **_size_parameters(video_create_optional_request_params.get("size"), model),
                **({"duration": seconds} if seconds is not None else {}),
                **({"resolution": resolution} if resolution is not None else {}),
                **native_parameters.model_dump(exclude_none=True),
            }
        )
        if model.endswith("-i2v") and parameters.ratio is not None:
            raise ValueError("HappyHorse image-to-video preserves the image aspect ratio and does not support ratio")
        return (
            {
                "model": model,
                "input": {
                    "prompt": resolved_prompt,
                    **(
                        {"media": [{"type": item.type, "url": _reference_url(item.url)} for item in media]}
                        if media
                        else {}
                    ),
                },
                "parameters": parameters.model_dump(exclude_none=True),
            },
            {},
            _operation_url(api_base, VIDEO_ENDPOINT),
        )

    def _result(self, raw_response: httpx.Response) -> _Result:
        if not raw_response.is_success:
            raise self.get_error_class(raw_response.text, raw_response.status_code, raw_response.headers)
        try:
            result: Final = _Result.model_validate_json(raw_response.content)
        except ValidationError:
            raise self.get_error_class("Invalid Alibaba Token Plan video response", 502, raw_response.headers) from None
        if result.code or result.output is None:
            raise self.get_error_class(
                result.message or result.code or "Missing video task output", 502, raw_response.headers
            )
        return result

    def _video(
        self, raw_response: httpx.Response, model: str | None, request_data: Mapping[str, object] | None = None
    ) -> VideoObject:
        result: Final = self._result(raw_response)
        output: Final = result.output
        if output is None:
            raise self.get_error_class("Missing video task output", 502, raw_response.headers)
        parameters: Final = _Parameters.model_validate((request_data or {}).get("parameters") or {})
        duration: Final = (
            (result.usage.output_video_duration or result.usage.duration) if result.usage else parameters.duration
        )
        resolution: Final = f"{result.usage.SR}P" if result.usage and result.usage.SR else parameters.resolution
        status: Final = {
            "PENDING": "queued",
            "RUNNING": "in_progress",
            "SUCCEEDED": "completed",
            "FAILED": "failed",
            "CANCELED": "failed",
            "UNKNOWN": "failed",
        }[output.task_status]
        return VideoObject(
            id=encode_video_id_with_provider(output.task_id, "alibaba_token_plan", model),
            object="video",
            status=status,
            model=model,
            created_at=_timestamp(output.submit_time),
            completed_at=_timestamp(output.end_time),
            progress=100 if status == "completed" else None,
            seconds=str(duration) if duration is not None else None,
            error={"code": output.code or output.task_status, "message": output.message or output.task_status}
            if status == "failed"
            else None,
            usage={
                **({"duration_seconds": duration} if duration is not None else {}),
                **({"video_resolution": resolution.lower()} if resolution else {}),
                **(
                    {"video_count": result.usage.video_count}
                    if result.usage and result.usage.video_count is not None
                    else {}
                ),
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
        return self._video(raw_response, model, request_data)

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: base provider interface
        task_id: Final = encode_url_path_segment(extract_original_video_id(video_id), field_name="video_id")
        return _operation_url(api_base, f"api/v1/tasks/{task_id}"), {}

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
        if variant not in (None, "video"):
            raise ValueError("Alibaba Token Plan video only supports the video content variant")
        return self.transform_video_status_retrieve_request(video_id, api_base, litellm_params, headers)

    def _video_download_url(self, raw_response: httpx.Response) -> str:
        output: Final = self._result(raw_response).output
        if output is None or output.task_status != "SUCCEEDED":
            status: Final = output.task_status if output else "UNKNOWN"
            raise self.get_error_class(f"Video content is not available: {status}", 409, raw_response.headers)
        if output.video_url is None:
            raise self.get_error_class("Completed video task has no video URL", 502, raw_response.headers)
        return str(output.video_url)

    def transform_video_content_response(self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj) -> bytes:
        try:
            video_response: Final = safe_get(litellm.module_level_client, self._video_download_url(raw_response))
        except httpx.HTTPError:
            raise self.get_error_class("Alibaba Token Plan video download failed", 502, {}) from None
        if not video_response.is_success:
            raise self.get_error_class(
                "Alibaba Token Plan video download failed",
                video_response.status_code,
                {},
            )
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        from litellm.llms.custom_httpx.http_handler import get_async_httpx_client

        try:
            video_response: Final = await async_safe_get(
                get_async_httpx_client(llm_provider=litellm.LlmProviders.ALIBABA_TOKEN_PLAN),
                self._video_download_url(raw_response),
            )
        except httpx.HTTPError:
            raise self.get_error_class("Alibaba Token Plan video download failed", 502, {}) from None
        if not video_response.is_success:
            raise self.get_error_class(
                "Alibaba Token Plan video download failed",
                video_response.status_code,
                {},
            )
        return video_response.content

    def get_error_class(
        self, error_message: str, status_code: int, headers: Mapping[str, str] | httpx.Headers
    ) -> BaseLLMException:
        return BaseLLMException(status_code=status_code, message=error_message, headers=dict(headers))

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: base provider interface
        raise NotImplementedError("Alibaba Token Plan does not support video remix")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Alibaba Token Plan does not support video remix")

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
        raise NotImplementedError("Alibaba Token Plan does not support listing video tasks")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: base provider interface
        raise NotImplementedError("Alibaba Token Plan does not support listing video tasks")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: base provider interface
        raise NotImplementedError("Alibaba Token Plan does not support deleting video tasks")

    def transform_video_delete_response(
        self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj
    ) -> VideoObject:
        raise NotImplementedError("Alibaba Token Plan does not support deleting video tasks")
