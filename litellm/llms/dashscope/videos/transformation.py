"""
DashScope (Alibaba Model Studio) async video task API: create, poll and download. DashScope has no list,
delete or remix endpoint, so those raise.
"""

import base64
from collections.abc import Mapping
from datetime import datetime
from io import BufferedReader, BytesIO
from math import gcd
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # house cached-client factory has no public alias and its stub leaves params untyped
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # factory stub leaves params untyped
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)
from litellm.utils import get_model_info

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.custom_httpx.http_handler import HTTPHandler


class DashScopeVideoError(BaseLLMException):
    pass


class _TaskOutput(BaseModel, frozen=True):
    task_id: str = ""
    task_status: str = ""
    submit_time: str | None = None
    scheduled_time: str | None = None
    end_time: str | None = None
    orig_prompt: str | None = None
    video_url: str | None = None
    code: str | None = None
    message: str | None = None


class _TaskUsage(BaseModel, frozen=True):
    video_count: int | None = None
    duration: float | None = None
    input_video_duration: float | None = None
    output_video_duration: float | None = None
    fps: int | None = None
    SR: int | None = None
    ratio: str | None = None


class _TaskResponse(BaseModel, frozen=True):
    output: _TaskOutput | None = None
    usage: _TaskUsage | None = None
    request_id: str | None = None
    code: str | None = None
    message: str | None = None


DASHSCOPE_VIDEO_DEFAULT_API_BASE: Final = "https://dashscope.aliyuncs.com"
DASHSCOPE_VIDEO_SYNTHESIS_PATH: Final = "/api/v1/services/aigc/video-generation/video-synthesis"
DASHSCOPE_TASKS_PATH: Final = "/api/v1/tasks"
DASHSCOPE_COMPATIBLE_MODE_PATH: Final = "/compatible-mode/v1"

_EMPTY_PARAMS: Final[dict[str, object]] = {}  # mutable-ok: BaseVideoConfig contract empty params

_TASK_RESPONSE_ADAPTER: Final = TypeAdapter(_TaskResponse)
_PARAMETERS_ADAPTER: Final = TypeAdapter(dict[str, object])

_STATUS_MAP: Final = MappingProxyType(
    {
        "PENDING": "queued",
        "RUNNING": "in_progress",
        "SUCCEEDED": "completed",
        "FAILED": "failed",
        "CANCELED": "cancelled",
        "UNKNOWN": "failed",
    }
)

_PENDING_STATUSES: Final = frozenset({"PENDING", "RUNNING"})

_LEGACY_INPUT_KEYS: Final = (
    "img_url",
    "first_frame_url",
    "last_frame_url",
    "audio_url",
)

_INPUT_KEYS: Final = ("media", "negative_prompt", *_LEGACY_INPUT_KEYS)

_REFERENCE_INPUT_FIELD: Final = "dashscope_video_reference_input"
_MEDIA_REFERENCE_TYPES: Final = frozenset({"first_frame", "reference_image"})

DASHSCOPE_DEFAULT_DURATION_SECONDS: Final = 5
DASHSCOPE_DEFAULT_RESOLUTION: Final = "1080P"
DASHSCOPE_SMART_DURATION: Final = -1

_PARAMETER_KEYS: Final = (
    "resolution",
    "ratio",
    "duration",
    "audio",
    "seed",
    "prompt_extend",
    "watermark",
)

_DROP_FROM_REQUEST: Final = frozenset(
    {
        "model",
        "prompt",
        "user",
        "characters",
        "image",
        "extra_headers",
        "extra_query",
        "extra_body",
    }
)

_RESOLUTION_TIERS: Final = MappingProxyType({480: "480p", 720: "720p", 1080: "1080p"})

_RESOLUTION_HEIGHT_TIERS: Final[tuple[tuple[int, str], ...]] = (
    (600, "480p"),
    (900, "720p"),
)


def _parse_task_response(raw_response: httpx.Response) -> _TaskResponse:
    return _TASK_RESPONSE_ADAPTER.validate_python(raw_response.json())


def _normalized_api_base(api_base: str) -> str:
    trimmed: Final = api_base.rstrip("/")
    if trimmed.endswith(DASHSCOPE_COMPATIBLE_MODE_PATH):
        return trimmed[: -len(DASHSCOPE_COMPATIBLE_MODE_PATH)]
    return trimmed


def _ratio_from_size(size: str) -> str | None:
    if ":" in size:
        return size
    width, height = _size_dimensions(size) or (0, 0)
    if not width or not height:
        return None
    divisor: Final = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


def _size_dimensions(size: str) -> tuple[int, int] | None:
    width_str, separator, height_str = size.partition("x")
    if not separator or not (width_str.isdigit() and height_str.isdigit()):
        return None
    return int(width_str), int(height_str)


def _resolution_from_size(size: str) -> str | None:
    """
    DashScope bills per second by named tier, so a size must map to its tier or it bills at the 1080P default.
    """
    dimensions: Final = _size_dimensions(size)
    if dimensions is None:
        return None
    shortest_side: Final = min(dimensions)
    return next(
        (label.upper() for threshold, label in _RESOLUTION_HEIGHT_TIERS if shortest_side < threshold),
        "1080P",
    )


def _duration_param(seconds: object) -> int | None:
    if isinstance(seconds, bool):
        return None
    if isinstance(seconds, int):
        return seconds
    if not isinstance(seconds, str):
        return None
    try:
        return int(float(seconds))
    except ValueError:
        return None


def _read_all_bytes(file_obj: object) -> bytes:
    if isinstance(file_obj, (BytesIO, BufferedReader)):
        current_position: Final = file_obj.tell()
        file_obj.seek(0)
        content: Final = file_obj.read()
        file_obj.seek(current_position)
        return content
    if isinstance(file_obj, bytes):
        return file_obj
    if isinstance(file_obj, bytearray):
        return bytes(file_obj)
    read: Final = getattr(file_obj, "read", None)
    if callable(read):
        data: Final = read()
        if isinstance(data, bytes):
            return data
    raise ValueError("input_reference must be a URL string, bytes, or a file object")


def _image_url(image: object) -> str:
    if isinstance(image, str):
        return image
    content_type: Final = ImageEditRequestUtils.get_image_content_type(image)
    encoded: Final = base64.b64encode(_read_all_bytes(image)).decode("utf-8")
    return f"data:{content_type};base64,{encoded}"


def _media_reference_type(model: str) -> str | None:
    """Models without the field predate the media array and take the image on the flat ``img_url`` field."""
    try:
        info: Final = get_model_info(model=model, custom_llm_provider="dashscope")
    except Exception:
        return None
    provider_specific: Final = info.get("provider_specific_entry")
    value: Final = provider_specific.get(_REFERENCE_INPUT_FIELD) if isinstance(provider_specific, Mapping) else None
    return value if isinstance(value, str) and value in _MEDIA_REFERENCE_TYPES else None


def _resolution_label(usage: _TaskUsage | None, requested_resolution: object) -> str | None:
    if usage is not None and isinstance(usage.SR, int) and not isinstance(usage.SR, bool):
        tier: Final = _RESOLUTION_TIERS.get(usage.SR)
        if tier is not None:
            return tier
    if isinstance(requested_resolution, str) and requested_resolution.strip():
        return requested_resolution.strip().lower()
    return None


def _video_usage(usage: _TaskUsage | None, requested: Mapping[str, object]) -> Mapping[str, object]:
    """
    ``usage.duration`` is DashScope's billed duration, including input video seconds, so it beats the request.
    """
    billed_duration: Final = usage.duration if usage is not None else None
    requested_duration: Final = requested.get("duration")
    duration_seconds: Final = (
        float(billed_duration)
        if isinstance(billed_duration, (int, float)) and not isinstance(billed_duration, bool)
        else float(requested_duration)
        if isinstance(requested_duration, (int, float))
        and not isinstance(requested_duration, bool)
        and requested_duration > 0
        else None
    )
    resolution: Final = _resolution_label(usage, requested.get("resolution"))
    return MappingProxyType(
        {
            key: value
            for key, value in (("duration_seconds", duration_seconds), ("video_resolution", resolution))
            if value is not None
        }
    )


def _timestamp(value: str | None) -> int | None:
    """
    DashScope stamps times in UTC+8 with no offset.
    """
    if not value:
        return None
    try:
        parsed: Final = datetime.strptime(f"{value}+0800", "%Y-%m-%d %H:%M:%S.%f%z")
    except ValueError:
        return None
    return int(parsed.timestamp())


def _error_block(output: _TaskOutput) -> Mapping[str, object] | None:
    if not (output.code or output.message):
        return None
    return MappingProxyType(
        {key: value for key, value in (("code", output.code), ("message", output.message)) if value is not None}
    )


def _size_from_usage(usage: _TaskUsage | None) -> str | None:
    if usage is None or not isinstance(usage.SR, int) or isinstance(usage.SR, bool) or usage.SR <= 0:
        return None
    if not usage.ratio or ":" not in usage.ratio:
        return None
    width_str, _, height_str = usage.ratio.partition(":")
    if not (width_str.isdigit() and height_str.isdigit()):
        return None
    ratio_width: Final = int(width_str)
    ratio_height: Final = int(height_str)
    if not ratio_width or not ratio_height:
        return None
    shortest_ratio_side: Final = min(ratio_width, ratio_height)
    return f"{usage.SR * ratio_width // shortest_ratio_side}x{usage.SR * ratio_height // shortest_ratio_side}"


def _video_object_from_task(
    task: _TaskResponse,
    model: str | None = None,
    requested: Mapping[str, object] | None = None,
) -> VideoObject:
    output: Final = task.output or _TaskOutput()
    status: Final = _STATUS_MAP.get(output.task_status, "queued")
    usage: Final = _video_usage(task.usage, requested or _EMPTY_PARAMS)
    seconds: Final = usage.get("duration_seconds")
    error_block: Final = _error_block(output)
    return VideoObject(
        id=output.task_id,
        object="video",
        status=status,
        created_at=_timestamp(output.submit_time),
        completed_at=_timestamp(output.end_time) if status == "completed" else None,
        error=dict(error_block) if error_block is not None else None,  # mutable-ok: VideoObject.error is a dict field
        seconds=str(seconds) if seconds is not None else None,
        size=_size_from_usage(task.usage),
        model=model,
        usage=dict(usage) if usage else None,  # mutable-ok: VideoObject.usage is a dict field
    )


def _video_url_from_task(task: _TaskResponse) -> str:
    output: Final = task.output or _TaskOutput()
    if output.video_url:
        return output.video_url

    if output.task_status in _PENDING_STATUSES:
        raise ValueError(f"Video is still processing (status: {output.task_status}). Please wait and try again.")
    if output.code or output.message:
        raise ValueError(f"Video generation failed: {output.message or output.code}")
    if output.task_status == "UNKNOWN":
        raise ValueError("Task not found. DashScope task ids expire 24 hours after creation.")
    raise ValueError("Video URL not found in task response. The task may not have succeeded yet.")


def _polled_model_id(logging_obj: object) -> str | None:
    """
    The deployment the proxy routes by is the model encoded in the polled id, so it must survive into the returned id.
    """
    litellm_params: Final = getattr(logging_obj, "litellm_params", None)
    video_id: Final = litellm_params.get("video_id") if isinstance(litellm_params, Mapping) else None
    if not isinstance(video_id, str):
        return None
    return decode_video_id_with_provider(video_id).get("model_id") or None


class DashScopeVideoConfig(BaseVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig contract returns list
        return [  # mutable-ok: BaseVideoConfig contract returns list
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "media",
            "resolution",
            "ratio",
            "duration",
            "audio",
            "seed",
            "prompt_extend",
            "watermark",
            "negative_prompt",
            "parameters",
            *_LEGACY_INPUT_KEYS,
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseVideoConfig contract
        mapped_params: Final[dict[str, object]] = {}  # mutable-ok: BaseVideoConfig contract; extra_body merges into it
        for key, value in video_create_optional_params.items():
            if value is None or key in _DROP_FROM_REQUEST:
                continue
            if key == "seconds":
                duration = _duration_param(value)
                if duration is not None:
                    mapped_params["duration"] = duration
            elif key == "size":
                if not isinstance(value, str):
                    continue
                ratio = _ratio_from_size(value)
                if ratio is not None:
                    mapped_params.setdefault("ratio", ratio)
                resolution = _resolution_from_size(value)
                if resolution is not None:
                    mapped_params.setdefault("resolution", resolution)
            elif key == "parameters":
                try:
                    mapped_params.update(_PARAMETERS_ADAPTER.validate_python(value))
                except ValidationError as e:
                    raise ValueError("parameters must be an object of DashScope request fields") from e
            else:
                mapped_params[key] = value
        return mapped_params

    def _resolve_api_key(self, api_key: str | None) -> str:
        resolved_api_key: Final = api_key or get_secret_str("DASHSCOPE_API_KEY")
        if resolved_api_key is None:
            raise ValueError(
                "DashScope API key is required. Set DASHSCOPE_API_KEY environment variable or pass api_key parameter."
            )
        return resolved_api_key

    def _resolve_video_api_base(self, video_api_base: str | None) -> str:
        return video_api_base or get_secret_str("DASHSCOPE_API_BASE_VIDEO") or DASHSCOPE_VIDEO_DEFAULT_API_BASE

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract; handler expects a mutable headers dict
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract
        resolved_api_key: Final = self._resolve_api_key(
            api_key
            or (litellm_params.api_key if litellm_params is not None and litellm_params.api_key else None)
            or litellm.api_key
        )

        auth_headers: Final[dict[str, str]] = {  # mutable-ok: httpx request headers are a mutable dict
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
            "X-DashScope-Async": "enable",
        }
        headers.update(auth_headers)
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: BaseVideoConfig contract
    ) -> str:
        return _normalized_api_base(self._resolve_video_api_base(api_base))

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict[str, object],  # mutable-ok: BaseVideoConfig contract
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[dict[str, object], RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract
        reference_field, reference_value = self._reference_input(model, video_create_optional_request_params)
        passthrough_input: Final = tuple(
            (key, video_create_optional_request_params[key])
            for key in _INPUT_KEYS
            if video_create_optional_request_params.get(key) is not None
        )
        reference_entry: Final = (
            ((reference_field, reference_value),)
            if reference_field is not None and reference_field not in frozenset(key for key, _ in passthrough_input)
            else ()
        )
        request_input: Final[dict[str, object]] = dict(  # mutable-ok: request body dict, JSON-serialized by the handler
            (("prompt", prompt), *passthrough_input, *reference_entry)
        )

        parameters: Final[dict[str, object]] = dict(  # mutable-ok: request body dict, JSON-serialized by the handler
            (key, video_create_optional_request_params[key])
            for key in _PARAMETER_KEYS
            if video_create_optional_request_params.get(key) is not None
        )
        if parameters.get("duration") == DASHSCOPE_SMART_DURATION:
            raise UnsupportedParamsError(
                message=(
                    "Smart duration (duration=-1) is not supported through litellm: the video is billed when the task "
                    "is created, before DashScope has picked its length. Pass an explicit duration in seconds."
                ),
                model=model,
                llm_provider="dashscope",
            )

        request_data: Final[dict[str, object]] = {  # mutable-ok: request body dict, JSON-serialized by the handler
            "model": model,
            "input": request_input,
        }
        if parameters:
            request_data["parameters"] = parameters

        return request_data, (), f"{api_base}{DASHSCOPE_VIDEO_SYNTHESIS_PATH}"

    @staticmethod
    def _reference_input(
        model: str,
        video_create_optional_request_params: Mapping[str, object],
    ) -> tuple[str | None, object]:
        input_reference: Final = video_create_optional_request_params.get("input_reference")
        if input_reference is None:
            return None, None
        image_url: Final = _image_url(input_reference)
        media_type: Final = _media_reference_type(model)
        if media_type is None:
            return "img_url", image_url
        # a plain dict because json.dumps cannot serialize a MappingProxyType
        return "media", (dict[str, object](type=media_type, url=image_url),)

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> VideoObject:
        task: Final = _parse_task_response(raw_response)
        self._raise_for_task_error(task, raw_response)
        requested: Final = self._requested_parameters(request_data)
        video_obj: Final = _video_object_from_task(task, model=model, requested=requested)
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model)
        return video_obj

    @staticmethod
    def _requested_parameters(request_data: Mapping[str, object] | None) -> Mapping[str, object]:
        """
        The create call is the only billed one, so an omitted duration or tier bills DashScope's defaults, not zero.
        """
        parameters: Final = (request_data or _EMPTY_PARAMS).get("parameters")
        requested: Final = (
            _PARAMETERS_ADAPTER.validate_python(parameters) if isinstance(parameters, Mapping) else _EMPTY_PARAMS
        )
        return MappingProxyType(
            {
                "duration": DASHSCOPE_DEFAULT_DURATION_SECONDS,
                "resolution": DASHSCOPE_DEFAULT_RESOLUTION,
                **requested,
            }
        )

    def _raise_for_task_error(self, task: _TaskResponse, raw_response: httpx.Response) -> None:
        """
        Create and lookup both report failures as a 200 with top-level code and message and no output.
        """
        if task.output is not None or not (task.code or task.message):
            return
        raise DashScopeVideoError(
            status_code=raw_response.status_code,
            message=task.message or task.code or "DashScope video request failed",
            headers=raw_response.headers,
            response=raw_response,
        )

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        return self._task_url(video_id, api_base), _EMPTY_PARAMS

    @staticmethod
    def _task_url(video_id: str, api_base: str) -> str:
        original_task_id: Final = extract_original_video_id(video_id)
        encoded_task_id: Final = encode_url_path_segment(original_task_id, field_name="video_id")
        return f"{api_base}{DASHSCOPE_TASKS_PATH}/{encoded_task_id}"

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        client: "HTTPHandler | None" = None,
    ) -> VideoObject:
        task: Final = _parse_task_response(raw_response)
        self._raise_for_task_error(task, raw_response)
        model: Final = _polled_model_id(logging_obj)
        video_obj: Final = _video_object_from_task(task, model=model)
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model)
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
        variant: str | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        return self._task_url(video_id, api_base), _EMPTY_PARAMS

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> bytes:
        video_url: Final = _video_url_from_task(_parse_task_response(raw_response))

        httpx_client: Final = _get_httpx_client()
        video_response: Final = httpx_client.get(video_url)  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get stub leaves params/headers untyped
        video_response.raise_for_status()

        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> bytes:
        video_url: Final = _video_url_from_task(_parse_task_response(raw_response))

        async_httpx_client: Final = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.DASHSCOPE,
        )
        video_response: Final = await async_httpx_client.get(video_url)  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get stub leaves params/headers untyped
        video_response.raise_for_status()

        return video_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
        extra_body: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError(
            "Video remix is not supported by DashScope. Wan 3.0 edits and extends a video by passing it back as a "
            "reference_video media entry on video_generation() with an editing or extension prompt."
        )

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by DashScope.")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError(
            "Video list is not supported by DashScope. Retrieve tasks individually by task id within their 24 hour "
            "retention window."
        )

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("Video list is not supported by DashScope.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError(
            "Video delete is not supported by DashScope. Tasks and their artifacts expire 24 hours after creation."
        )

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by DashScope.")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: BaseVideoConfig contract
    ) -> BaseLLMException:
        return DashScopeVideoError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
