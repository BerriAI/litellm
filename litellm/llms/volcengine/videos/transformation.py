import json
from collections.abc import Mapping, Sequence
from math import log
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, TypeVar

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, StrictBool, StrictInt, StrictStr, TypeAdapter, ValidationError

import litellm
from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.async_task_utils import (
    input_reference_url,
    polled_video_models,
    routed_video_models,
)
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # house cached-client factory has no public alias and its stub leaves params untyped
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # factory stub leaves params untyped
)
from litellm.llms.volcengine.common_utils import get_volcengine_base_url
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ModelInfo
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider, extract_original_video_id
from litellm.utils import get_model_info

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging


_ParsedT: Final = TypeVar("_ParsedT")


class VolcEngineVideoError(BaseLLMException):
    pass


class _TaskError(BaseModel, frozen=True):
    code: str | None = None
    message: str | None = None


class _TaskContent(BaseModel, frozen=True):
    video_url: str | None = None


class _TaskUsage(BaseModel, frozen=True):
    total_tokens: int | None = None


class _Task(BaseModel, frozen=True):
    id: str
    status: str
    model: str | None = None
    error: _TaskError | None = None
    content: _TaskContent | None = None
    usage: _TaskUsage | None = None
    created_at: int | None = None
    updated_at: int | None = None
    resolution: str | None = None
    ratio: str | None = None
    duration: int | float | None = None


class _TaskCreateResponse(BaseModel, frozen=True):
    id: str | None = None
    error: _TaskError | None = None


class _TaskList(BaseModel, frozen=True):
    items: Sequence[_Task] = ()
    total: int | None = None


class _ContentItem(BaseModel, frozen=True):
    type: str | None = None
    role: str | None = None


class _KnownFields(BaseModel, frozen=True):
    resolution: StrictStr | None = None
    ratio: StrictStr | None = None
    seed: StrictInt | None = None
    camera_fixed: StrictBool | None = None
    watermark: StrictBool | None = None
    generate_audio: StrictBool | None = None


VOLCENGINE_VIDEO_TASKS_PATH: Final = "/api/v3/contents/generations/tasks"
VOLCENGINE_VIDEO_DEFAULT_DURATION_SECONDS: Final = 4
VOLCENGINE_VIDEO_DEFAULT_RESOLUTION: Final = "720p"
_VOLCENGINE_API_MARKER: Final = "/api/"
_VOLCENGINE_COMPAT_SUFFIXES: Final = ("/api", "/v1")
_FRAMES_FIELD: Final = "volcengine_video_frames"

_EMPTY_PARAMS: Final[dict[str, object]] = {}  # mutable-ok: BaseVideoConfig contract empty params

_TASK_ADAPTER: Final = TypeAdapter(_Task)
_TASK_CREATE_ADAPTER: Final = TypeAdapter(_TaskCreateResponse)
_TASK_LIST_ADAPTER: Final = TypeAdapter(_TaskList)
_FIELDS_ADAPTER: Final = TypeAdapter(dict[str, object])
_CONTENT_ITEMS_ADAPTER: Final = TypeAdapter(tuple[dict[str, object], ...])
_FRAMES_ADAPTER: Final = TypeAdapter(dict[str, dict[str, str]])
_EMPTY_FRAMES: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType({})

_STATUS_MAP: Final = MappingProxyType(
    {
        "queued": "queued",
        "running": "in_progress",
        "succeeded": "completed",
        "failed": "failed",
        "cancelled": "cancelled",
        "expired": "failed",
    }
)
_PENDING_STATUSES: Final = frozenset({"queued", "running"})
_TERMINAL_STATUSES: Final = frozenset({"completed", "failed", "cancelled"})

_CREATE_PASSTHROUGH_KEYS: Final = (
    "resolution",
    "ratio",
    "duration",
    "seed",
    "camera_fixed",
    "watermark",
    "generate_audio",
    "service_tier",
    "priority",
    "output_format",
    "execution_expires_after",
    "callback_url",
    "safety_identifier",
    "tools",
    "omni_reference_task_type",
)

_REQUEST_METADATA: Final = frozenset({"model", "prompt", "user", "extra_headers", "extra_query", "extra_body"})
_UNSUPPORTED_BY_VOLCENGINE: Final = frozenset({"characters", "image"})
_MAPPED_ELSEWHERE: Final = frozenset({*_REQUEST_METADATA, *_UNSUPPORTED_BY_VOLCENGINE, "seconds", "size", "parameters"})
_READ_FROM_THE_REQUEST: Final = _MAPPED_ELSEWHERE - _UNSUPPORTED_BY_VOLCENGINE
_LITELLM_CONTROLS: Final = frozenset({"drop_params", "additional_drop_params", "organization"})
_DROP_KEYS_ADAPTER: Final = TypeAdapter(frozenset[str])

# litellm/videos/utils.py merges the caller's raw ``extra_body`` back over the mapped params, which returns the keys
# this adapter maps itself a second time, as fields of their own. They are already accounted for, so they count as
# handled rather than unknown.
_MAPPED_EXTRA_BODY_COPIES: Final = frozenset({"size", "seconds", "user"})
_CREATE_HANDLED_KEYS: Final = frozenset(
    {
        *_CREATE_PASSTHROUGH_KEYS,
        "content",
        "input_reference",
        "frames",
        *_MAPPED_EXTRA_BODY_COPIES,
        *_LITELLM_CONTROLS,
    }
)

_VIDEO_INPUT_UNSUPPORTED: Final = (
    "{operation} is not supported through litellm: Ark bills a task with a video input at a different rate per "
    "token, and litellm bills the video when the task is created, before that cost is known"
)
_INPUT_VIDEO_ROLE: Final = "reference_video"
_INPUT_VIDEO_TYPE: Final = "video_url"

_RATIO_SET: Final = ("16:9", "9:16", "4:3", "3:4", "1:1", "21:9")
_RESOLUTION_SHORTEST_SIDE: Final = MappingProxyType({"480p": 480, "720p": 720, "1080p": 1080, "4k": 2160})
_TIER_FRAME_ASPECT: Final = 16 / 9


def _video_error(raw_response: httpx.Response, status_code: int, message: str) -> VolcEngineVideoError:
    """The provider's own error status wins, and a body that failed on a 2xx answers the caller's status instead,
    since a 2xx error has the proxy retry a call the provider already rejected."""
    return VolcEngineVideoError(
        status_code=raw_response.status_code if raw_response.status_code >= 400 else status_code,
        message=message,
        headers=raw_response.headers,
    )


def _parse(adapter: TypeAdapter[_ParsedT], raw_response: httpx.Response) -> _ParsedT:
    try:
        return adapter.validate_python(raw_response.json())
    except (json.JSONDecodeError, ValidationError) as e:
        raise _video_error(raw_response, 502, f"Unexpected Volcengine video response: {e}") from e


def _unsupported(model: str, message: str) -> UnsupportedParamsError:
    return UnsupportedParamsError(message=message, model=model, llm_provider="volcengine")


def _droppable(model: str, message: str) -> UnsupportedParamsError:
    return _unsupported(model, f"{message}, set drop_params=True to ignore it")


def _normalized_api_base(api_base: str) -> str:
    """The Ark credential is shared with the chat, responses and Anthropic surfaces, so its api_base may carry any of
    their paths, and the video paths carry their own version segment."""
    trimmed: Final = api_base.rstrip("/")
    marker_index: Final = trimmed.rfind(_VOLCENGINE_API_MARKER)
    if marker_index != -1:
        return trimmed[:marker_index]
    suffix: Final = next((suffix for suffix in _VOLCENGINE_COMPAT_SUFFIXES if trimmed.endswith(suffix)), None)
    return trimmed[: -len(suffix)] if suffix is not None else trimmed


def _size_dimensions(size: str) -> tuple[int, int] | None:
    width_str, separator, height_str = size.partition("x")
    if not separator or not (width_str.isdigit() and height_str.isdigit()):
        return None
    if not int(width_str) or not int(height_str):
        return None
    return int(width_str), int(height_str)


def _ratio_value(ratio: str) -> float:
    width_str, _, height_str = ratio.partition(":")
    return int(width_str) / int(height_str)


def _ratio_from_size(size: str) -> str | None:
    """Ark only takes its named ratios, and its own tiers do not reduce onto them (854x480 is 427:240), so the
    closest ratio wins."""
    dimensions: Final = _size_dimensions(size)
    if dimensions is None:
        return None
    width, height = dimensions
    return min(_RATIO_SET, key=lambda candidate: abs(log(width / height / _ratio_value(candidate))))


def _resolution_from_size(size: str) -> str | None:
    """Ark keeps a tier's pixel count across its ratios (960x960 and 1280x720 are both 720p), so the tier whose 16:9
    frame has the closest pixel count wins."""
    dimensions: Final = _size_dimensions(size)
    if dimensions is None:
        return None
    width, height = dimensions
    return min(
        _RESOLUTION_SHORTEST_SIDE,
        key=lambda tier: abs(log(width * height / (_RESOLUTION_SHORTEST_SIDE[tier] ** 2 * _TIER_FRAME_ASPECT))),
    )


def _duration_param(seconds: object) -> int | None:
    if isinstance(seconds, bool):
        return None
    if isinstance(seconds, int):
        return seconds
    if not isinstance(seconds, (str, float)):
        return None
    try:
        parsed: Final = float(seconds)
    except ValueError:
        return None
    return int(parsed) if parsed.is_integer() else None


def _duration_fields(seconds: object, model: str, drop_params: bool) -> Mapping[str, object]:
    if seconds is None:
        return _EMPTY_PARAMS
    duration: Final = _duration_param(seconds)
    if duration is not None:
        return MappingProxyType({"duration": duration})
    if drop_params:
        return _EMPTY_PARAMS
    raise _droppable(model, f"seconds must be a whole number of seconds, got {seconds!r}")


def _size_fields(size: object, model: str, drop_params: bool) -> Mapping[str, object]:
    if size is None:
        return _EMPTY_PARAMS
    ratio: Final = _ratio_from_size(size) if isinstance(size, str) else None
    resolution: Final = _resolution_from_size(size) if isinstance(size, str) else None
    if ratio is not None and resolution is not None:
        return MappingProxyType({"ratio": ratio, "resolution": resolution})
    if drop_params:
        return _EMPTY_PARAMS
    raise _droppable(model, f"size must be WIDTHxHEIGHT, got {size!r}")


def _parameters_fields(parameters: object, model: str, drop_params: bool) -> Mapping[str, object]:
    if parameters is None:
        return _EMPTY_PARAMS
    try:
        fields: Final = _FIELDS_ADAPTER.validate_python(parameters)
    except ValidationError:
        if drop_params:
            return _EMPTY_PARAMS
        raise _droppable(
            model, f"parameters must be an object of Ark request fields, got {type(parameters).__name__}"
        ) from None
    misplaced: Final = tuple(sorted(key for key in fields if key in _READ_FROM_THE_REQUEST))
    if misplaced and not drop_params:
        raise _droppable(model, f"send {', '.join(misplaced)} as top-level request fields, not inside parameters")
    return MappingProxyType({key: value for key, value in fields.items() if key not in _READ_FROM_THE_REQUEST})


def _user_fields(user: object, model: str, drop_params: bool) -> Mapping[str, object]:
    if isinstance(user, str) and user:
        return MappingProxyType({"safety_identifier": user})
    if user is None or isinstance(user, str) or drop_params:
        return _EMPTY_PARAMS
    raise _droppable(model, f"user must be a string, got {type(user).__name__}")


def _additional_drop_keys(params: Mapping[str, object]) -> frozenset[str]:
    try:
        return _DROP_KEYS_ADAPTER.validate_python(params.get("additional_drop_params") or ())
    except ValidationError:
        return frozenset()


def _without(params: Mapping[str, object], keys: frozenset[str]) -> Mapping[str, object]:
    return MappingProxyType({key: value for key, value in params.items() if key not in keys})


def _malformed_fields(fields: Mapping[str, object]) -> tuple[tuple[str, str], ...]:
    try:
        _ = _KnownFields.model_validate(dict(fields))
    except ValidationError as e:
        return tuple((str(error["loc"][0]), error["msg"]) for error in e.errors() if error["loc"])
    return ()


def _drops_params(litellm_params: GenericLiteLLMParams) -> bool:
    return litellm_params.drop_params is True or litellm.drop_params is True


def _model_info(model: str) -> ModelInfo | None:
    try:
        return get_model_info(model=model, custom_llm_provider="volcengine")
    except Exception:  # noqa: BLE001  # get_model_info raises bare Exception for unmapped models
        return None


def _pricing_model(model: str, litellm_params: GenericLiteLLMParams) -> str:
    """A deployment priced through ``model_info.base_model``, like an ``ep-`` endpoint id, bills at that model's
    tiers."""
    model_info: Final = _FIELDS_ADAPTER.validate_python(litellm_params["model_info"] or _EMPTY_PARAMS)
    base_model: Final = model_info.get("base_model")
    return base_model if isinstance(base_model, str) and base_model else model


def _priced_tiers(model: str) -> tuple[str, ...] | None:
    """A tier the model has no price for is one Ark does not offer it at, and litellm would bill it at the base rate.
    A model with no tier price listed, missing from the cost map or registered unpriced by the router, has nothing
    to check against."""
    info: Final = _model_info(model)
    if info is None:
        return None
    return (
        tuple(tier for tier in _RESOLUTION_SHORTEST_SIDE if info.get(f"output_cost_per_second_{tier}") is not None)
        or None
    )


def _delivered_frames(model: str | None) -> Mapping[str, Mapping[str, str]]:
    info: Final = _model_info(model) if model else None
    provider_specific: Final[object] = info.get("provider_specific_entry") if info is not None else None
    frames: Final[object] = provider_specific.get(_FRAMES_FIELD) if isinstance(provider_specific, Mapping) else None
    try:
        return _FRAMES_ADAPTER.validate_python(frames) if frames is not None else _EMPTY_FRAMES
    except ValidationError:
        return _EMPTY_FRAMES


def _size_from_geometry(resolution: str | None, ratio: str | None, model: str | None) -> str | None:
    """Ark echoes a tier and a ratio rather than pixels, and its rendered frame does not always follow the tier's
    nominal size, so the frame listed for the model wins over scaling the tier's shortest side up the ratio."""
    if not resolution or not ratio or ":" not in ratio:
        return None
    tier: Final = resolution.strip().lower()
    normalized_ratio: Final = ratio.strip()
    listed: Final = _delivered_frames(model).get(tier, MappingProxyType({})).get(normalized_ratio)
    if listed is not None:
        return listed
    shortest_side: Final = _RESOLUTION_SHORTEST_SIDE.get(tier)
    width_str, _, height_str = normalized_ratio.partition(":")
    if shortest_side is None or not (width_str.isdigit() and height_str.isdigit()):
        return None
    ratio_width: Final = int(width_str)
    ratio_height: Final = int(height_str)
    if not ratio_width or not ratio_height:
        return None
    shortest_ratio_side: Final = min(ratio_width, ratio_height)
    return f"{-(-shortest_side * ratio_width // shortest_ratio_side)}x{-(-shortest_side * ratio_height // shortest_ratio_side)}"


def _is_input_video(item: _ContentItem) -> bool:
    return item.type == _INPUT_VIDEO_TYPE or item.role == _INPUT_VIDEO_ROLE


def _supplied_items(supplied: object, model: str, drop_params: bool) -> tuple[Mapping[str, object], ...]:
    try:
        items: Final = _CONTENT_ITEMS_ADAPTER.validate_python(supplied)
        entries: Final = tuple(_ContentItem.model_validate(item) for item in items)
    except ValidationError:
        if drop_params:
            return ()
        raise _droppable(model, "content must be an array of Ark content items") from None
    if any(entry.type == "text" for entry in entries) and not drop_params:
        raise _droppable(
            model,
            "A text item in content is not supported, put the text in prompt since it is the only text input "
            "guardrails scan before the request reaches Ark",
        )
    if any(_is_input_video(entry) for entry in entries) and not drop_params:
        raise _droppable(model, _VIDEO_INPUT_UNSUPPORTED.format(operation="Video input (a reference_video item)"))
    kept: Final = (item for item, entry in zip(items, entries) if entry.type != "text" and not _is_input_video(entry))
    return tuple(MappingProxyType(item) for item in kept)


def _reference_items(params: Mapping[str, object], model: str, drop_params: bool) -> tuple[Mapping[str, object], ...]:
    """A caller-supplied ``content`` array wins over ``input_reference``, since its roles are the only way to express
    reference-to-video."""
    supplied: Final = params.get("content")
    if supplied is not None:
        return _supplied_items(supplied, model, drop_params)
    reference: Final = params.get("input_reference")
    if reference is None:
        return ()
    try:
        url: Final = input_reference_url(reference)
    except ValueError as e:
        if drop_params:
            return ()
        raise _droppable(model, str(e)) from None
    return (MappingProxyType({"type": "image_url", "image_url": {"url": url}, "role": "first_frame"}),)


def _sent_content(
    prompt: str, params: Mapping[str, object], model: str, drop_params: bool
) -> tuple[Mapping[str, object], ...]:
    """The prompt always leads, as it is the only text the input guardrails scan."""
    return (MappingProxyType({"type": "text", "text": prompt}), *_reference_items(params, model, drop_params))


def _seconds_label(seconds: object) -> str | None:
    if isinstance(seconds, float) and seconds.is_integer():
        return str(int(seconds))
    return None if seconds is None else str(seconds)


def _requested_usage(request_data: Mapping[str, object]) -> Mapping[str, object]:
    """The create call is the only billed one and Ark's create response carries no usage, so the requested duration
    and tier are what the video bills at."""
    duration: Final = _duration_param(request_data.get("duration"))
    resolution: Final = request_data.get("resolution")
    return MappingProxyType(
        {
            key: value
            for key, value in (
                ("duration_seconds", float(duration) if duration is not None and duration > 0 else None),
                ("video_resolution", resolution.strip().lower() if isinstance(resolution, str) else None),
            )
            if value is not None
        }
    )


def _task_usage(task: _Task) -> Mapping[str, object]:
    resolution: Final = task.resolution.strip().lower() if task.resolution else None
    return MappingProxyType(
        {
            key: value
            for key, value in (
                ("duration_seconds", float(task.duration) if task.duration is not None else None),
                ("video_resolution", resolution),
                ("total_tokens", task.usage.total_tokens if task.usage is not None else None),
            )
            if value is not None
        }
    )


def _task_status(task: _Task, raw_response: httpx.Response) -> str:
    status: Final = _STATUS_MAP.get(task.status.strip().lower())
    if status is None:  # defaulting it would hand back a job the caller polls forever
        raise _video_error(raw_response, 502, f"Unknown Volcengine task status: {task.status}")
    return status


def _video_object_from_task(task: _Task, raw_response: httpx.Response, model: str | None) -> VideoObject:
    status: Final = _task_status(task, raw_response)
    usage: Final = _task_usage(task)
    seconds: Final = usage.get("duration_seconds")
    error: Final[dict[str, object] | None] = (  # mutable-ok: VideoObject.error is a dict field
        {key: value for key, value in (("code", task.error.code), ("message", task.error.message)) if value is not None}
        if task.error is not None
        else None
    )
    return VideoObject(
        id=task.id,
        object="video",
        status=status,
        created_at=task.created_at,
        completed_at=task.updated_at if status in _TERMINAL_STATUSES else None,
        error=error,
        seconds=_seconds_label(seconds),
        size=_size_from_geometry(task.resolution, task.ratio, task.model),
        model=model or task.model,
        usage=dict(usage) if usage else None,
    )


def _video_url_from_task(task: _Task, raw_response: httpx.Response) -> str:
    """A download before the video is ready is the caller polling too early, not a fault of the deployment, so it
    answers 400, which litellm neither retries nor counts toward a deployment cooldown. A 404 would cool the
    deployment down, and a 409 or a 5xx would be retried against a job that is simply not finished."""
    if task.content is not None and task.content.video_url:
        return task.content.video_url
    status: Final = task.status.strip().lower()
    if status in _PENDING_STATUSES:
        raise _video_error(
            raw_response, 400, f"Video is still processing (status: {status}). Please wait and try again."
        )
    if task.error is not None and (task.error.message or task.error.code):
        raise _video_error(raw_response, 400, f"Video generation failed: {task.error.message or task.error.code}")
    if status == "expired":
        raise _video_error(raw_response, 400, "Task expired. Ark keeps task records for 7 days after creation.")
    if status in ("failed", "cancelled"):
        raise _video_error(raw_response, 400, f"Video generation ended as {status} with no video URL.")
    raise _video_error(raw_response, 502, "Video URL not found in task response.")


class VolcEngineVideoConfig(BaseVideoConfig):
    """Volcengine Ark (Doubao-Seedance) on Ark's async task API: create, poll, download from the signed
    ``content.video_url``, and list."""

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig contract returns list
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
            "content",
            "parameters",
            *_CREATE_PASSTHROUGH_KEYS,
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseVideoConfig contract
        requested: Final = _without(video_create_optional_params, _additional_drop_keys(video_create_optional_params))
        unsupported: Final = next(
            (key for key, value in requested.items() if key in _UNSUPPORTED_BY_VOLCENGINE and value is not None),
            None,
        )
        if unsupported is not None and not drop_params:
            raise _droppable(model, f"'{unsupported}' is not supported by Volcengine video generation")
        passthrough: Final = MappingProxyType(
            {key: value for key, value in requested.items() if value is not None and key not in _MAPPED_ELSEWHERE}
        )
        return {
            **_user_fields(requested.get("user"), model, drop_params),
            **_size_fields(requested.get("size"), model, drop_params),
            **_duration_fields(requested.get("seconds"), model, drop_params),
            **passthrough,
            **_parameters_fields(requested.get("parameters"), model, drop_params),
        }

    def _resolve_api_key(self, api_key: str | None) -> str:
        resolved_api_key: Final = api_key or get_secret_str("ARK_API_KEY") or get_secret_str("VOLCENGINE_API_KEY")
        if resolved_api_key is None:
            raise ValueError(
                "Volcengine API key is required. Set ARK_API_KEY or VOLCENGINE_API_KEY environment variable or pass "
                "api_key parameter."
            )
        return resolved_api_key

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
        headers.update({"Authorization": f"Bearer {resolved_api_key}", "Content-Type": "application/json"})
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict[str, object],  # mutable-ok: BaseVideoConfig contract
    ) -> str:
        return _normalized_api_base(
            get_volcengine_base_url(api_base or get_secret_str("VOLCENGINE_API_BASE") or get_secret_str("ARK_API_BASE"))
        )

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict[str, object],  # mutable-ok: BaseVideoConfig contract
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[dict[str, object], RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract
        """
        Ark bills a video when the task finishes, while litellm bills it here at create, so every request has to
        carry a length and a tier whose cost is known now. Ark-picked length (``duration: -1``) and a video input
        (edits, extensions and reference videos, billed at a different rate per token) are rejected for that reason,
        and an omitted length is sent as OpenAI's documented default of 4 seconds, since the 2.5 family's own default
        is ``-1``.
        """
        drop_params: Final = _drops_params(litellm_params)
        dropped_by_config: Final = _additional_drop_keys(video_create_optional_request_params)
        kept: Final = _without(video_create_optional_request_params, dropped_by_config)
        params: Final = _without(
            {
                **{key: value for key, value in kept.items() if key != "parameters"},
                **_parameters_fields(kept.get("parameters"), model, drop_params),
            },
            dropped_by_config,
        )
        content: Final = _sent_content(prompt, params, model, drop_params)
        if params.get("frames") is not None and not drop_params:
            raise _droppable(model, "frames is not supported through litellm, pass seconds instead")
        requested_duration: Final = params.get("duration")
        parsed_duration: Final = None if requested_duration is None else _duration_param(requested_duration)
        if requested_duration is not None and (parsed_duration is None or parsed_duration <= 0) and not drop_params:
            raise _droppable(
                model,
                f"duration must be a positive whole number of seconds, got {requested_duration!r}. Ark-picked length "
                "(-1) is not supported through litellm, since the video is billed when the task is created, before "
                "Ark has picked its length",
            )
        duration: Final = (
            parsed_duration
            if parsed_duration is not None and parsed_duration > 0
            else VOLCENGINE_VIDEO_DEFAULT_DURATION_SECONDS
        )

        unknown_fields: Final = tuple(
            sorted(key for key, value in params.items() if value is not None and key not in _CREATE_HANDLED_KEYS)
        )
        if unknown_fields and not drop_params:
            raise _droppable(model, f"litellm does not send {', '.join(unknown_fields)} to Ark video generation")
        requested_fields: Final = MappingProxyType(
            {key: params[key] for key in _CREATE_PASSTHROUGH_KEYS if params.get(key) is not None}
        )
        malformed: Final = _malformed_fields(requested_fields)
        if malformed and not drop_params:
            raise _droppable(model, "; ".join(f"invalid {field}: {message}" for field, message in malformed))
        resolution: Final = requested_fields.get("resolution")
        priced_tiers: Final = _priced_tiers(_pricing_model(model, litellm_params))
        drops_resolution: Final = (
            priced_tiers is not None and isinstance(resolution, str) and resolution.strip().lower() not in priced_tiers
        )
        if drops_resolution and not drop_params:
            raise _droppable(
                model, f"{model} is not offered at resolution {resolution!r}, only at {', '.join(priced_tiers or ())}"
            )
        dropped: Final = frozenset({*(field for field, _ in malformed), *(("resolution",) if drops_resolution else ())})
        kept_fields: Final = MappingProxyType(
            {key: value for key, value in requested_fields.items() if key not in dropped}
        )
        default_tier: Final = (
            MappingProxyType({"resolution": VOLCENGINE_VIDEO_DEFAULT_RESOLUTION})
            if "resolution" not in kept_fields
            and priced_tiers is not None
            and VOLCENGINE_VIDEO_DEFAULT_RESOLUTION in priced_tiers
            else _EMPTY_PARAMS
        )

        request_data: Final[dict[str, object]] = {  # mutable-ok: request body dict, JSON-serialized by the handler
            "model": model,
            "content": [dict(item) for item in content],
            **default_tier,
            **kept_fields,
            "duration": duration,
        }
        return request_data, (), f"{api_base}{VOLCENGINE_VIDEO_TASKS_PATH}"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> VideoObject:
        task: Final = _parse(_TASK_CREATE_ADAPTER, raw_response)
        if not task.id:
            message: Final = task.error.message if task.error is not None and task.error.message else "no task id"
            raise _video_error(raw_response, 400, f"Volcengine did not return a video task id: {message}")
        requested: Final = request_data or _EMPTY_PARAMS
        usage: Final = _requested_usage(requested)
        seconds: Final = usage.get("duration_seconds")
        requested_model, served_model = routed_video_models(logging_obj) or (model, None)
        resolution: Final = requested.get("resolution")
        ratio: Final = requested.get("ratio")
        return VideoObject(
            id=encode_video_id_with_provider(task.id, custom_llm_provider, requested_model, served_model)
            if custom_llm_provider
            else task.id,
            object="video",
            status="queued",
            model=model,
            seconds=_seconds_label(seconds),
            size=_size_from_geometry(
                resolution if isinstance(resolution, str) else None, ratio if isinstance(ratio, str) else None, model
            ),
            usage=dict(usage) if usage else None,
        )

    @staticmethod
    def _task_url(video_id: str, api_base: str) -> str:
        original_task_id: Final = extract_original_video_id(video_id)
        encoded_task_id: Final = encode_url_path_segment(original_task_id, field_name="video_id")
        return f"{api_base}{VOLCENGINE_VIDEO_TASKS_PATH}/{encoded_task_id}"

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        return self._task_url(video_id, api_base), _EMPTY_PARAMS

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        client: "HTTPHandler | None" = None,
    ) -> VideoObject:
        task: Final = _parse(_TASK_ADAPTER, raw_response)
        model, served_model = polled_video_models(logging_obj)
        video_obj: Final = _video_object_from_task(task, raw_response, model)
        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(task.id, custom_llm_provider, model, served_model)
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
        client: "HTTPHandler | None" = None,
    ) -> bytes:
        video_url: Final = _video_url_from_task(_parse(_TASK_ADAPTER, raw_response), raw_response)
        httpx_client: Final[HTTPHandler] = client if client is not None else _get_httpx_client()
        video_response: Final = httpx_client.get(video_url)  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get stub leaves params/headers untyped
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        client: "AsyncHTTPHandler | None" = None,
    ) -> bytes:
        video_url: Final = _video_url_from_task(_parse(_TASK_ADAPTER, raw_response), raw_response)
        async_httpx_client: Final[AsyncHTTPHandler] = (
            client if client is not None else get_async_httpx_client(llm_provider=litellm.LlmProviders.VOLCENGINE)
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
        raise _unsupported("", _VIDEO_INPUT_UNSUPPORTED.format(operation="Video remix"))

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by Volcengine.")

    def transform_video_edit_request(
        self,
        prompt: str,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
        video_file: object | None = None,
        extra_body: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
        prefetched_source_data: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, Mapping[str, object], RequestFiles | None]:
        raise _unsupported("", _VIDEO_INPUT_UNSUPPORTED.format(operation="Video edit"))

    def transform_video_extension_request(
        self,
        prompt: str,
        video_id: str,
        seconds: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
        extra_body: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise _unsupported("", _VIDEO_INPUT_UNSUPPORTED.format(operation="Video extension"))

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
        """Ark pages by page_num/page_size with filters and has no cursor or order."""
        unsupported: Final = tuple(name for name, value in (("after", after), ("order", order)) if value is not None)
        if unsupported and not _drops_params(litellm_params):
            raise _droppable(
                "", f"Ark's task list pages by page_num in extra_query and has no {' or '.join(unsupported)}"
            )
        return f"{api_base}{VOLCENGINE_VIDEO_TASKS_PATH}", {
            **({"page_size": str(limit)} if limit is not None else {}),
            **(extra_query or {}),
        }

    def transform_video_list_response(  # pyright: ignore[reportIncompatibleMethodOverride]  # base declares dict[str, str] but the payload is a heterogeneous list body
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: OpenAI list body served as JSON by the proxy
        list_payload: Final = _parse(_TASK_LIST_ADAPTER, raw_response)
        videos: Final = tuple(_video_object_from_task(task, raw_response, None) for task in list_payload.items)
        data: Final = [
            video.model_dump()
            | (
                {"id": encode_video_id_with_provider(video.id, custom_llm_provider, video.model)}
                if custom_llm_provider
                else {}
            )
            for video in videos
        ]
        return {
            "object": "list",
            "data": data,
            "total": list_payload.total if list_payload.total is not None else len(data),
            "has_more": False,
            **({"first_id": data[0]["id"], "last_id": data[-1]["id"]} if data else {}),
        }

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict[str, str],  # mutable-ok: BaseVideoConfig contract
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("Video delete is not supported by Volcengine.")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> VideoObject:
        raise NotImplementedError("Video delete is not supported by Volcengine.")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: BaseVideoConfig contract
    ) -> BaseLLMException:
        return VolcEngineVideoError(status_code=status_code, message=error_message, headers=headers)
