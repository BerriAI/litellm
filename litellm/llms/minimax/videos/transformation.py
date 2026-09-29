"""
MiniMax V2 video task API: create, poll, list, download and delete. Regeneration (/v2/video_regeneration)
only upscales a finished 768P task to 2K and ignores the prompt, so it is not exposed as an OpenAI remix.
"""

import base64
from collections.abc import Mapping, Sequence
from io import BufferedReader, BytesIO
from math import gcd
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal

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
from litellm.llms.openai.cost_calculation import video_generation_cost
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)
from litellm.utils import get_model_info

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging
    from litellm.llms.custom_httpx.http_handler import HTTPHandler


class MinimaxVideoError(BaseLLMException):
    pass


class _TaskError(BaseModel, frozen=True):
    code: str | None = None
    message: str | None = None


class _TaskUsage(BaseModel, frozen=True):
    total_seconds: int | None = None
    input_seconds: int | None = None
    output_seconds: int | None = None
    input_image_count: int | None = None
    input_audio_seconds: int | None = None
    total_tokens: int | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class _TaskContent(BaseModel, frozen=True):
    url: str | None = None
    prompt: str | None = None


class _ContentItem(BaseModel, frozen=True):
    type: str = ""
    text: str | None = None
    role: str | None = None


class _MediaUrl(BaseModel, frozen=True, extra="forbid"):
    url: str


class _CallerMediaItem(BaseModel, frozen=True, extra="forbid"):
    type: Literal["image_url", "video_url", "audio_url"]
    image_url: _MediaUrl | None = None
    video_url: _MediaUrl | None = None
    audio_url: _MediaUrl | None = None
    role: Literal["first_frame", "last_frame", "reference_image", "reference_video", "reference_audio"] | None = None


class _MiniMaxTask(BaseModel, frozen=True):
    id: str = ""
    model: str | None = None
    status: str = ""
    error: _TaskError | None = None
    created_at: int | None = None
    updated_at: int | None = None
    content: _TaskContent | None = None
    resolution: str | None = None
    duration: int | None = None
    usage: _TaskUsage | None = None
    ratio: str | None = None
    task_type: str | None = None
    modality: str | None = None


class _TaskIdResponse(BaseModel, frozen=True):
    task_id: str = ""


class _TaskResponse(BaseModel, frozen=True):
    task: _MiniMaxTask | None = None


class _ListResponse(BaseModel, frozen=True):
    items: Sequence[_MiniMaxTask] | None = None
    total: int | None = None


class _DeleteResponse(BaseModel, frozen=True):
    task_id: str = ""
    status: str = ""


MINIMAX_VIDEO_DEFAULT_API_BASE: Final = "https://api.minimax.io"
MINIMAX_VIDEO_DEFAULT_RESOLUTION: Final = "768P"
MINIMAX_VIDEO_DEFAULT_DURATION_SECONDS: Final = 5
MINIMAX_TEXT_TO_VIDEO_DEFAULT_RATIO: Final = "16:9"

_EMPTY_PARAMS: Final[dict[str, object]] = {}  # mutable-ok: BaseVideoConfig contract empty params

_RESPONSE_ADAPTER: Final = TypeAdapter(dict[str, object])
_TASK_ID_RESPONSE_ADAPTER: Final = TypeAdapter(_TaskIdResponse)
_TASK_RESPONSE_ADAPTER: Final = TypeAdapter(_TaskResponse)
_LIST_RESPONSE_ADAPTER: Final = TypeAdapter(_ListResponse)
_DELETE_RESPONSE_ADAPTER: Final = TypeAdapter(_DeleteResponse)
_CONTENT_ITEMS_ADAPTER: Final = TypeAdapter(list[_ContentItem])
_CALLER_MEDIA_ADAPTER: Final = TypeAdapter(list[_CallerMediaItem])

_STATUS_MAP: Final = MappingProxyType(
    {
        "queued": "queued",
        "running": "in_progress",
        "succeeded": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
    }
)

_DROP_FROM_CREATE_BODY: Final = frozenset(
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

_CREATE_BODY_KEYS: Final = (
    "resolution",
    "duration",
    "ratio",
    "callback_url",
    "aigc_watermark",
    "extra",
)


def _parse_task_response(raw_response: httpx.Response) -> _MiniMaxTask:
    return _TASK_RESPONSE_ADAPTER.validate_python(raw_response.json()).task or _MiniMaxTask()


def _parse_task_id_response(raw_response: httpx.Response) -> str:
    return _TASK_ID_RESPONSE_ADAPTER.validate_python(raw_response.json()).task_id


def _parse_delete_response(raw_response: httpx.Response) -> tuple[str, str]:
    deleted: Final = _DELETE_RESPONSE_ADAPTER.validate_python(raw_response.json())
    return deleted.task_id, deleted.status


MINIMAX_SURFACE_SUFFIXES: Final = ("/v1", "/v2", "/anthropic")


def _normalized_api_base(api_base: str) -> str:
    """
    A MiniMax key covers the chat surfaces (``/v1``, ``/anthropic``) and video (``/v2``), so strip whichever
    surface suffix a shared api_base carries back to the host.
    """
    trimmed: Final = api_base.rstrip("/")
    matched: Final = next((suffix for suffix in MINIMAX_SURFACE_SUFFIXES if trimmed.endswith(suffix)), None)
    return trimmed[: -len(matched)] if matched else trimmed


def _ratio_from_size(size: str) -> str | None:
    if ":" in size:
        return size
    width_str, separator, height_str = size.partition("x")
    if not separator or not (width_str.isdigit() and height_str.isdigit()):
        return None
    width: Final = int(width_str)
    height: Final = int(height_str)
    divisor: Final = gcd(width, height)
    return f"{width // divisor}:{height // divisor}"


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


def _first_frame_content_item(
    image: object,
) -> dict[str, object]:  # mutable-ok: content items are JSON request-body fragments
    return {  # mutable-ok: request-body content item serialized to JSON by the handler
        "type": "image_url",
        "image_url": {"url": _image_url(image)},  # mutable-ok: request-body content item field
        "role": "first_frame",
    }


def _content_items(content: object) -> tuple[_ContentItem, ...]:
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return ()
    try:
        return tuple(_CONTENT_ITEMS_ADAPTER.validate_python(content))
    except ValidationError:
        return ()


def _is_text_only_content(content: object) -> bool:
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)) or not content:
        return False
    try:
        items: Final = _CONTENT_ITEMS_ADAPTER.validate_python(content)
    except ValidationError:
        return False
    return all(item.type == "text" for item in items)


def _video_object_from_task(task: _MiniMaxTask) -> VideoObject:
    status: Final = _STATUS_MAP.get(task.status, "queued")
    usage_dump: Final = task.usage.model_dump(exclude_none=True) if task.usage is not None else None
    return VideoObject(
        id=task.id,
        object="video",
        status=status,
        created_at=task.created_at,
        completed_at=task.updated_at if status == "completed" else None,
        error=task.error.model_dump(exclude_none=True) if task.error is not None else None,
        seconds=str(task.duration) if task.duration is not None else None,
        model=task.model,
        usage=usage_dump or None,
    )


def _create_cost_usd(model: str, duration: float, resolution: str | None, input_image_count: int) -> float | None:
    """
    MiniMax bills input images beyond a per-model free allowance on top of output seconds, and the create
    call is the only billed one, so the whole charge is reported for the cost calculator to use as is.
    """
    try:
        info: Final = get_model_info(model=model, custom_llm_provider="minimax")
    except Exception:
        return None
    provider_specific: Final = info.get("provider_specific_entry")
    free_images: Final = (
        provider_specific.get("minimax_free_input_images") if isinstance(provider_specific, Mapping) else None
    )
    image_rate: Final = info.get("input_cost_per_image")
    if not isinstance(free_images, (int, float)) or not isinstance(image_rate, (int, float)):
        return None
    output_cost: Final = video_generation_cost(
        model=model, duration_seconds=duration, custom_llm_provider="minimax", video_resolution=resolution
    )
    return output_cost + max(0, input_image_count - int(free_images)) * image_rate


def _video_url_from_task(task: _MiniMaxTask) -> str:
    if task.content is not None and task.content.url:
        return task.content.url

    if task.status in ("queued", "running"):
        raise ValueError(f"Video is still processing (status: {task.status}). Please wait and try again.")
    if task.error is not None:
        raise ValueError(f"Video generation failed: {task.error.message or 'unknown error'}")
    raise ValueError("Video URL not found in task response. The task may not have succeeded yet.")


class MinimaxVideoConfig(BaseVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig contract returns list
        return [  # mutable-ok: BaseVideoConfig contract returns list
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "resolution",
            "user",
            "extra_headers",
            "duration",
            "ratio",
            "content",
            "callback_url",
            "aigc_watermark",
            "extra",
            "parameters",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseVideoConfig contract
        mapped_params: Final[dict[str, object]] = {}  # mutable-ok: BaseVideoConfig contract; extra_body merges into it
        for key, value in video_create_optional_params.items():
            if value is None or key in _DROP_FROM_CREATE_BODY:
                continue
            if key == "seconds":
                duration = _duration_param(value)
                if duration is not None:
                    mapped_params["duration"] = duration
            elif key == "size":
                ratio = _ratio_from_size(value) if isinstance(value, str) else None
                if ratio is not None:
                    mapped_params.setdefault("ratio", ratio)
            elif key == "parameters":
                try:
                    mapped_params.update(_RESPONSE_ADAPTER.validate_python(value))
                except ValidationError as e:
                    raise ValueError("parameters must be an object of MiniMax request fields") from e
            else:
                mapped_params[key] = value
        return mapped_params

    def validate_environment(
        self,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract; handler expects a mutable headers dict
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract
        resolved_api_key: Final = (
            api_key
            or (litellm_params.api_key if litellm_params is not None and litellm_params.api_key else None)
            or litellm.api_key
            or get_secret_str("MINIMAX_API_KEY")
        )

        if resolved_api_key is None:
            raise ValueError(
                "MiniMax API key is required. Set MINIMAX_API_KEY environment variable or pass api_key parameter."
            )

        auth_headers: Final[dict[str, str]] = {  # mutable-ok: httpx request headers are a mutable dict
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }
        headers.update(auth_headers)
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: BaseVideoConfig contract
    ) -> str:
        resolved_api_base: Final = api_base or get_secret_str("MINIMAX_API_BASE") or MINIMAX_VIDEO_DEFAULT_API_BASE
        return _normalized_api_base(resolved_api_base)

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict[str, object],  # mutable-ok: BaseVideoConfig contract
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[dict[str, object], RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract
        content: Final = self._content_param(video_create_optional_request_params, prompt)
        if any(item.type == "video_url" for item in _content_items(content)):
            raise UnsupportedParamsError(
                message=(
                    "Reference video input is not supported through litellm: MiniMax bills its duration, which is only "
                    "known after the task is created and billed. Use image or audio references instead."
                ),
                model=model,
                llm_provider="minimax",
            )
        request_data: Final[dict[str, object]] = {  # mutable-ok: request body dict, JSON-serialized by the handler
            "model": model,
            "content": content,
        }
        for key in _CREATE_BODY_KEYS:
            if video_create_optional_request_params.get(key) is not None:
                request_data[key] = video_create_optional_request_params[key]

        request_data.setdefault("resolution", MINIMAX_VIDEO_DEFAULT_RESOLUTION)
        request_data.setdefault("duration", MINIMAX_VIDEO_DEFAULT_DURATION_SECONDS)
        if "ratio" not in request_data and _is_text_only_content(content):
            request_data["ratio"] = MINIMAX_TEXT_TO_VIDEO_DEFAULT_RATIO

        return request_data, (), f"{api_base}/v2/video_generation"

    @staticmethod
    def _content_param(video_create_optional_request_params: Mapping[str, object], prompt: str) -> object:
        """
        ``prompt`` is what guardrails scanned, so it is always the one text item MiniMax takes; a caller's
        ``content`` array only contributes media, never text that would bypass that check.
        """
        explicit_content: Final = video_create_optional_request_params.get("content")
        if explicit_content is not None:
            try:
                media: Final = _CALLER_MEDIA_ADAPTER.validate_python(explicit_content)
            except ValidationError as e:
                raise UnsupportedParamsError(
                    message=(
                        "content must be a list of MiniMax image_url, video_url or audio_url items; pass the text of "
                        f"the request as prompt. {e.error_count()} invalid item field(s)."
                    ),
                    llm_provider="minimax",
                ) from e
            return [  # mutable-ok: JSON request-body content
                {"type": "text", "text": prompt},
                *(item.model_dump(exclude_none=True) for item in media),
            ]

        content_items: Final[list[dict[str, object]]] = [  # mutable-ok: JSON request-body content items
            {"type": "text", "text": prompt}  # mutable-ok: JSON request-body content item
        ]
        input_reference: Final = video_create_optional_request_params.get("input_reference")
        if input_reference is not None:
            content_items.append(_first_frame_content_item(input_reference))
        return content_items

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> VideoObject:
        task_id: Final = _parse_task_id_response(raw_response)
        request_mapping: Final[Mapping[str, object]] = request_data if request_data is not None else _EMPTY_PARAMS
        duration: Final = request_mapping.get("duration")
        resolution: Final = request_mapping.get("resolution")
        input_image_count: Final = sum(
            1 for item in _content_items(request_mapping.get("content")) if item.type == "image_url"
        )

        video_obj: Final = VideoObject(
            id=task_id,
            object="video",
            status="queued",
            model=model,
            seconds=str(duration) if duration is not None else None,
        )
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model)

        usage: Final[dict[str, object]] = {}  # mutable-ok: VideoObject.usage is a mutable dict field
        if isinstance(duration, (int, float)):
            usage["duration_seconds"] = float(duration)
        if isinstance(resolution, str):
            usage["video_resolution"] = resolution.strip().lower()
        if input_image_count:
            usage["input_image_count"] = input_image_count
        create_cost: Final = (
            _create_cost_usd(model, float(duration), resolution.strip().lower(), input_image_count)
            if isinstance(duration, (int, float)) and isinstance(resolution, str)
            else None
        )
        if create_cost is not None:
            usage["provider_reported_cost_usd"] = create_cost
        video_obj.usage = usage

        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        original_task_id: Final = extract_original_video_id(video_id)
        encoded_task_id: Final = encode_url_path_segment(original_task_id, field_name="video_id")
        return f"{api_base}/v2/query/video_generation/{encoded_task_id}", _EMPTY_PARAMS

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        client: "HTTPHandler | None" = None,
    ) -> VideoObject:
        task: Final = _parse_task_response(raw_response)
        video_obj: Final = _video_object_from_task(task)
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, task.model)
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
        variant: str | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        original_task_id: Final = extract_original_video_id(video_id)
        encoded_task_id: Final = encode_url_path_segment(original_task_id, field_name="video_id")
        return f"{api_base}/v2/query/video_generation/{encoded_task_id}", _EMPTY_PARAMS

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> bytes:
        task: Final = _parse_task_response(raw_response)
        video_url: Final = _video_url_from_task(task)

        httpx_client: Final = _get_httpx_client()
        video_response: Final = httpx_client.get(video_url)  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get stub leaves params/headers untyped
        video_response.raise_for_status()

        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> bytes:
        task: Final = _parse_task_response(raw_response)
        video_url: Final = _video_url_from_task(task)

        async_httpx_client: Final = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.MINIMAX,
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
            "Video remix is not supported by MiniMax. Its regeneration endpoint only upscales a finished 768P "
            "MiniMax-H3 task to 2K and ignores the prompt; send a new video_generation() request with the edited "
            "prompt instead."
        )

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by MiniMax.")

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
        if after is not None:
            raise UnsupportedParamsError(
                message=(
                    "MiniMax video list does not support cursor pagination via 'after'. "
                    "Pass extra_query={'page_num': N} to request a later page."
                ),
                llm_provider="minimax",
            )
        if order is not None and order != "desc":
            raise UnsupportedParamsError(
                message="MiniMax video list only returns newest first; order must be 'desc' or omitted.",
                llm_provider="minimax",
            )
        params: Final[dict[str, object]] = {}  # mutable-ok: query params dict consumed by the http handler
        if limit is not None:
            params["page_size"] = str(limit)
        if extra_query:
            params.update(extra_query)
        return f"{api_base}/v2/query/video_generation", params

    def transform_video_list_response(  # pyright: ignore[reportIncompatibleMethodOverride]  # base declares dict[str, str] but the payload is a heterogeneous list body
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: OpenAI list body served as JSON by the proxy
        list_payload: Final = _LIST_RESPONSE_ADAPTER.validate_python(raw_response.json())
        total: Final = list_payload.total

        data: Final[list[dict[str, object]]] = []  # mutable-ok: OpenAI list body served as JSON by the proxy
        for task in list_payload.items or ():
            video_obj = _video_object_from_task(task)
            if custom_llm_provider and video_obj.id:
                video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, video_obj.model)
            data.append(video_obj.model_dump())

        list_response: Final[dict[str, object]] = {  # mutable-ok: OpenAI list body served as JSON by the proxy
            "object": "list",
            "data": data,
            "total": total if isinstance(total, int) and not isinstance(total, bool) else len(data),
        }
        if data:
            list_response["first_id"] = data[0]["id"]
            list_response["last_id"] = data[-1]["id"]
        return list_response

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        original_task_id: Final = extract_original_video_id(video_id)
        encoded_task_id: Final = encode_url_path_segment(original_task_id, field_name="video_id")
        return f"{api_base}/v2/video_generation/{encoded_task_id}", _EMPTY_PARAMS

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> VideoObject:
        task_id, status = _parse_delete_response(raw_response)
        return VideoObject(
            id=task_id,
            object="video",
            status=status,
        )

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: BaseVideoConfig contract
    ) -> BaseLLMException:
        return MinimaxVideoError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
