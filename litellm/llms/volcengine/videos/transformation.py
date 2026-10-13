"""Volcengine Ark (方舟) video generation, async task API.

One endpoint family serves every Doubao-Seedance model:

- ``POST /api/v3/contents/generations/tasks`` creates a task and returns
  ``{"id": "cgt-..."}``
- ``GET /api/v3/contents/generations/tasks/{id}`` polls it; once ``succeeded``
  the artifact sits at ``content.video_url``, a 24 hour signed TOS link
- ``GET /api/v3/contents/generations/tasks`` lists tasks
- ``DELETE /api/v3/contents/generations/tasks/{id}`` cancels a queued task or
  removes a finished one

The body is flat apart from a typed ``content`` array: the prompt is a ``text``
item and every image/video/audio input is a url item tagged with a ``role``
(``first_frame`` / ``last_frame`` / ``reference_image`` / ``reference_video`` /
``reference_audio``). Ark derives the task type from the roles present and
refuses to mix the frame roles with the reference roles, so ``content`` is the
only way to express reference-to-video, video editing and video extension.

Ark bills per output token, not per second: ``tokens = width * height * fps *
duration / 1024`` at 24fps, which makes a per-second rate per resolution tier
exact for the 16:9 geometry Ark publishes. 1280x720 is 21600 tokens/s, so the
70 CNY per million tokens list rate for Seedance 2.5 is 1.512 CNY/s.
"""

import base64
from collections.abc import Mapping, Sequence
from io import BufferedReader, BytesIO
from math import log
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, TypeAdapter, ValidationError

import litellm
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage, reportUnknownVariableType]  # house cached-client factory has no public alias and its stub leaves params untyped
    get_async_httpx_client,  # pyright: ignore[reportUnknownVariableType]  # factory stub leaves params untyped
)
from litellm.llms.volcengine.common_utils import get_volcengine_base_url
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging


class VolcEngineVideoError(BaseLLMException):
    pass


class _TaskError(BaseModel, frozen=True):
    code: str | None = None
    message: str | None = None


class _TaskContent(BaseModel, frozen=True):
    video_url: str | None = None
    last_frame_url: str | None = None


class _TaskUsage(BaseModel, frozen=True):
    completion_tokens: int | None = None
    total_tokens: int | None = None


class _Task(BaseModel, frozen=True):
    id: str = ""
    model: str | None = None
    status: str = ""
    error: _TaskError | None = None
    content: _TaskContent | None = None
    usage: _TaskUsage | None = None
    created_at: int | str | None = None
    updated_at: int | str | None = None
    resolution: str | None = None
    ratio: str | None = None
    duration: int | str | None = None
    frames: int | None = None
    framespersecond: int | None = None


class _TaskCreateResponse(BaseModel, frozen=True):
    id: str = ""
    error: _TaskError | None = None


class _TaskList(BaseModel, frozen=True):
    items: Sequence[_Task] | None = None
    total: int | None = None


VOLCENGINE_VIDEO_TASKS_PATH: Final = "/api/v3/contents/generations/tasks"
_VOLCENGINE_API_MARKER: Final = "/api/"
_VOLCENGINE_COMPAT_SUFFIXES: Final = ("/api", "/v1")

_EMPTY_PARAMS: Final[dict[str, object]] = {}  # mutable-ok: BaseVideoConfig contract empty params

_TASK_ADAPTER: Final = TypeAdapter(_Task)
_TASK_CREATE_ADAPTER: Final = TypeAdapter(_TaskCreateResponse)
_TASK_LIST_ADAPTER: Final = TypeAdapter(_TaskList)
_PARAMETERS_ADAPTER: Final = TypeAdapter(dict[str, object])
_CONTENT_ITEMS_ADAPTER: Final = TypeAdapter(tuple[dict[str, object], ...])

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

_TERMINAL_STATUSES: Final = frozenset({"completed", "failed", "cancelled"})

_CREATE_PASSTHROUGH_KEYS: Final = (
    "resolution",
    "ratio",
    "duration",
    "frames",
    "seed",
    "camera_fixed",
    "watermark",
    "generate_audio",
    "return_last_frame",
    "service_tier",
    "draft",
    "priority",
    "output_format",
    "execution_expires_after",
    "callback_url",
    "safety_identifier",
    "tools",
    "omni_reference_task_type",
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
        "content",
    }
)

_EDIT_DURATION_PARAM: Final = "duration"

_RATIO_SET: Final = ("16:9", "9:16", "4:3", "3:4", "1:1", "21:9")

_RESOLUTION_SHORTEST_SIDE: Final = MappingProxyType(
    {
        "480p": 480,
        "720p": 720,
        "1080p": 1080,
        "2160p": 2160,
        "4k": 2160,
    }
)

_RESOLUTION_HEIGHT_TIERS: Final[tuple[tuple[int, str], ...]] = (
    (600, "480p"),
    (900, "720p"),
    (1500, "1080p"),
)

_DEFAULT_DURATION_SECONDS: Final = MappingProxyType(
    {
        "doubao-seedance-2-0": 5.0,
    }
)

# Ark echoes the delivered geometry only as a resolution tier plus a ratio, and
# the frame it renders does not follow the tier's nominal pixel size: the 2.0
# family's 16:9 "480p" canvas is 864x496, not the 854x480 the tier name implies.
# Values are the published frames from the Seedance 2.0 Mini model card
# (replicate.com/bytedance/seedance-2.0-mini/readme, 2026-10-13) and were all
# re-verified against the dev gateway with ffprobe at 480p across every ratio
# and at 720p for 16:9 and 4:3; the remaining 720p ratios are the card's.
# A (tier, ratio) that is absent falls back to the tier's nominal shortest side
# scaled up the ratio.
_DELIVERED_FRAME: Final = MappingProxyType(
    {
        "doubao-seedance-2-0": MappingProxyType(
            {
                ("480p", "16:9"): (864, 496),
                ("480p", "4:3"): (752, 560),
                ("480p", "1:1"): (640, 640),
                ("480p", "3:4"): (560, 752),
                ("480p", "9:16"): (496, 864),
                ("480p", "21:9"): (992, 432),
                ("720p", "16:9"): (1280, 720),
                ("720p", "4:3"): (1112, 834),
                ("720p", "1:1"): (960, 960),
                ("720p", "3:4"): (834, 1112),
                ("720p", "9:16"): (720, 1280),
                ("720p", "21:9"): (1470, 630),
            }
        ),
        "doubao-seedance-2-5": MappingProxyType(
            {
                ("480p", "16:9"): (854, 480),
                ("480p", "4:3"): (752, 560),
                ("480p", "1:1"): (640, 640),
                ("480p", "3:4"): (560, 752),
                ("480p", "9:16"): (480, 854),
                ("480p", "21:9"): (992, 432),
                ("720p", "16:9"): (1280, 720),
                ("720p", "4:3"): (1112, 834),
                ("720p", "1:1"): (960, 960),
                ("720p", "3:4"): (834, 1112),
                ("720p", "9:16"): (720, 1280),
                ("720p", "21:9"): (1470, 630),
            }
        ),
    }
)


def _normalized_api_base(api_base: str) -> str:
    """
    Normalize a Volcengine api_base to the Ark host root.

    ``get_volcengine_base_url`` is shared with the chat, responses and
    Anthropic-compatible configs, so an api_base carries whichever surface the
    caller uses: ``/api/v3`` for the native one, ``/api/compatible`` for the
    Anthropic one, a bare host otherwise. The video paths carry their own
    version segment, so the surface is cut at the first ``/api/`` marker before
    the transform methods append theirs.
    """
    trimmed: Final = api_base.rstrip("/")
    marker_index: Final = trimmed.find(_VOLCENGINE_API_MARKER)
    if marker_index != -1:
        return trimmed[:marker_index]
    for suffix in _VOLCENGINE_COMPAT_SUFFIXES:
        if trimmed.endswith(suffix):
            return trimmed[: -len(suffix)]
    return trimmed


def _size_dimensions(size: str) -> tuple[int, int] | None:
    width_str, separator, height_str = size.partition("x")
    if not separator or not (width_str.isdigit() and height_str.isdigit()):
        return None
    width: Final = int(width_str)
    height: Final = int(height_str)
    if not width or not height:
        return None
    return width, height


def _ratio_from_size(size: str) -> str | None:
    """
    Snap an OpenAI ``size`` (``WIDTHxHEIGHT``) onto one of Ark's named ratios.

    Ark only accepts a fixed ratio set, and its own tiers do not reduce onto it
    (854x480 is 427:240), so the closest ratio by aspect wins rather than the
    gcd-reduced form, which Ark would reject.
    """
    if ":" in size:
        return size
    dimensions: Final = _size_dimensions(size)
    if dimensions is None:
        return None
    width, height = dimensions
    requested: Final = width / height
    return min(_RATIO_SET, key=lambda candidate: abs(log(requested / _ratio_value(candidate))))


def _ratio_value(ratio: str) -> float:
    width_str, _, height_str = ratio.partition(":")
    if not (width_str.isdigit() and height_str.isdigit()) or not int(height_str):
        return 0.0
    return int(width_str) / int(height_str)


def _resolution_from_size(size: str) -> str | None:
    """
    Pick the Ark resolution tier an OpenAI ``size`` falls in.

    Ark takes a tier name (``720p``) rather than a pixel size and bills per
    second by that tier, so ``1280x720`` has to become ``720p`` instead of
    being dropped and silently billed at whatever the model defaults to.
    """
    dimensions: Final = _size_dimensions(size)
    if dimensions is None:
        return None
    shortest_side: Final = min(dimensions)
    return next(
        (label for threshold, label in _RESOLUTION_HEIGHT_TIERS if shortest_side < threshold),
        "4k",
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


def _content_items(prompt: str, params: Mapping[str, object]) -> tuple[dict[str, object], ...]:
    """
    Build Ark's typed ``content`` array.

    A caller-supplied ``content`` wins outright, since the roles are the only
    way to express reference-to-video, edit and extension jobs; the prompt is
    still prepended unless they already passed a text item. Otherwise the
    prompt is the only text item and ``input_reference`` becomes the first
    frame, which is the image-to-video shape.
    """
    supplied: Final = params.get("content")
    if supplied is not None:
        try:
            items: Final = _CONTENT_ITEMS_ADAPTER.validate_python(supplied)
        except ValidationError as e:
            raise ValueError("content must be an array of Ark content items") from e
        if any(item.get("type") == "text" for item in items):
            return items
        return ({"type": "text", "text": prompt}, *items)

    reference: Final = params.get("input_reference")
    if reference is None:
        return ({"type": "text", "text": prompt},)
    return (
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": _image_url(reference)}, "role": "first_frame"},
    )


def _parse_task(raw_response: httpx.Response) -> _Task:
    return _TASK_ADAPTER.validate_python(raw_response.json())


def _coerce_timestamp(value: int | str | None) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    try:
        return int(float(value))
    except ValueError:
        return None


def _coerce_duration(value: int | str | None) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return float(value)
    try:
        return float(value)
    except ValueError:
        return None


def _scaled_side(shortest_side: int, ratio_side: int, shortest_ratio_side: int) -> int:
    """
    Scale the tier's shortest side up the ratio, rounding up.

    Ark publishes 480p as 854x480 while 480 * 16 / 9 is 853.33, so truncating
    reports a width the model never produces.
    """
    return -(-shortest_side * ratio_side // shortest_ratio_side)


def _delivered_frame(model: str | None, tier: str, ratio: str) -> tuple[int, int] | None:
    if not model:
        return None
    for prefix, frames in _DELIVERED_FRAME.items():
        if model.startswith(prefix):
            return frames.get((tier, ratio))
    return None


def _size_from_geometry(resolution: str | None, ratio: str | None, model: str | None) -> str | None:
    """
    Reconstruct an OpenAI ``WIDTHxHEIGHT`` size from Ark's echoed geometry.

    Ark reports the delivered geometry as a resolution tier (``720p``, the
    shortest side in pixels) plus a ratio, so the two are recombined rather
    than reporting the bare tier as a pixel size. Models whose rendered frame
    does not follow the tier's nominal pixel size (see ``_DELIVERED_FRAME``)
    are served from that table instead.
    """
    if not resolution or not ratio or ":" not in ratio:
        return None
    tier: Final = resolution.strip().lower()
    normalized_ratio: Final = ratio.strip()
    measured: Final = _delivered_frame(model, tier, normalized_ratio)
    if measured is not None:
        return f"{measured[0]}x{measured[1]}"
    shortest_side: Final = _RESOLUTION_SHORTEST_SIDE.get(tier)
    if shortest_side is None:
        return None
    width_str, _, height_str = normalized_ratio.partition(":")
    if not (width_str.isdigit() and height_str.isdigit()):
        return None
    ratio_width: Final = int(width_str)
    ratio_height: Final = int(height_str)
    if not ratio_width or not ratio_height:
        return None
    shortest_ratio_side: Final = min(ratio_width, ratio_height)
    width: Final = _scaled_side(shortest_side, ratio_width, shortest_ratio_side)
    height: Final = _scaled_side(shortest_side, ratio_height, shortest_ratio_side)
    return f"{width}x{height}"


def _size_from_task(task: _Task) -> str | None:
    return _size_from_geometry(task.resolution, task.ratio, task.model)


def _video_usage(task: _Task) -> Mapping[str, object]:
    """
    Build the usage block the video cost calculator reads.

    ``duration`` and ``resolution`` are Ark's own delivered values, which is
    what the per-second tiers have to be applied to: ``duration: -1`` and
    ``ratio: adaptive`` both let the service deliver something other than what
    was asked for. ``total_tokens`` is carried through because Ark bills by
    token, so it is the figure to reconcile an invoice against.
    """
    return MappingProxyType(
        {
            key: value
            for key, value in (
                ("duration_seconds", _coerce_duration(task.duration)),
                ("video_resolution", task.resolution.strip().lower() if task.resolution else None),
                ("total_tokens", task.usage.total_tokens if task.usage is not None else None),
            )
            if value is not None
        }
    )


def _video_object_from_task(task: _Task) -> VideoObject:
    status: Final = _STATUS_MAP.get(task.status.strip().lower(), "failed")
    usage: Final = _video_usage(task)
    seconds: Final = usage.get("duration_seconds")
    error: Final[dict[str, object] | None] = (
        {key: value for key, value in (("code", task.error.code), ("message", task.error.message)) if value is not None}
        if task.error is not None
        else None
    )
    return VideoObject(
        id=task.id,
        object="video",
        status=status,
        created_at=_coerce_timestamp(task.created_at),
        completed_at=_coerce_timestamp(task.updated_at) if status in _TERMINAL_STATUSES else None,
        error=error,  # mutable-ok: VideoObject.error is a dict field
        seconds=str(seconds) if seconds is not None else None,
        size=_size_from_task(task),
        model=task.model,
        usage=dict(usage) if usage else None,  # mutable-ok: VideoObject.usage is a dict field
    )


def _positive_seconds(value: object) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) if value > 0 else None
    if isinstance(value, str):
        try:
            parsed: Final = float(value)
        except ValueError:
            return None
        return parsed if parsed > 0 else None
    return None


def _default_create_duration(model: str | None) -> float | None:
    if not model:
        return None
    for prefix, seconds in _DEFAULT_DURATION_SECONDS.items():
        if model.startswith(prefix):
            return seconds
    return None


def _requested_usage(request_data: Mapping[str, object] | None, model: str | None = None) -> Mapping[str, object]:
    """
    Build the usage block for the create response out of the requested values.

    Only the create, edit and remix calls are costed (see ``_VIDEO_CALL_TYPES``),
    and Ark's create response carries no usage at all, so without this every
    video would be logged at zero cost. The requested duration and resolution
    are the best estimate available at create time; the delivered values on the
    status response are what an invoice should be reconciled against.

    A caller that leaves ``seconds`` off gets Ark's own default length, which is
    still billable, so a model whose default is a fixed number falls back to it
    here. The 2.5 family defaults to ``-1`` (Ark picks the length), which has no
    honest number to report, so it stays unbilled until the status response.
    """
    params: Final = request_data or _EMPTY_PARAMS
    resolution: Final = params.get("resolution")
    duration: Final = params.get("duration")
    requested_seconds: Final = _positive_seconds(duration) if duration is not None else _default_create_duration(model)
    return MappingProxyType(
        {
            key: value
            for key, value in (
                ("duration_seconds", requested_seconds),
                (
                    "video_resolution",
                    resolution.strip().lower() if isinstance(resolution, str) and resolution.strip() else None,
                ),
            )
            if value is not None
        }
    )


def _encoded_task_dump(task: _Task, custom_llm_provider: str | None) -> dict[str, object]:
    video_obj: Final = _video_object_from_task(task)
    if custom_llm_provider and video_obj.id:
        video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, video_obj.model)
    return video_obj.model_dump()


def _video_error(status_code: int, message: str) -> VolcEngineVideoError:
    return VolcEngineVideoError(status_code=status_code, message=message)


def _video_url_from_task(task: _Task) -> str:
    """
    A task whose artifact cannot be served yet is the caller's own state to fix, not a
    gateway fault, so every branch here answers 4xx. A 5xx would have the router retry a
    job that is simply unfinished and count the caller's early download against the
    model's error rate.
    """
    content: Final = task.content
    if content is not None and content.video_url:
        return content.video_url

    status: Final = task.status.strip().lower()
    if status in ("queued", "running"):
        raise _video_error(404, f"Video is still processing (status: {task.status}). Please wait and try again.")
    if task.error is not None and (task.error.message or task.error.code):
        raise _video_error(400, f"Video generation failed: {task.error.message or task.error.code}")
    if status == "expired":
        raise _video_error(404, "Task expired before it finished. Ark task records expire 7 days after creation.")
    raise _video_error(502, "Video URL not found in task response. The task may not have succeeded yet.")


def _string_param(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return None


def _routed_model(litellm_params: GenericLiteLLMParams | Mapping[str, object]) -> str | None:
    """
    The model the router already resolved for this request, if any.

    ``litellm_params`` is a dict-like pydantic model whose ``get`` is untyped,
    so the value is read out and validated rather than trusted.
    """
    dumped: Final = dict(litellm_params) if isinstance(litellm_params, Mapping) else litellm_params.model_dump()
    return _string_param(_PARAMETERS_ADAPTER.validate_python(dumped).get("model"))


def _edit_model(
    video_id: str,
    extra_body: Mapping[str, object] | None,
    litellm_params: GenericLiteLLMParams | Mapping[str, object],
) -> str:
    """
    Resolve the model an edit or extension task should run on.

    litellm's edit and extension transforms are handed no model argument, so it
    is recovered from, in order: an explicit ``extra_body.model``, the model
    encoded in the video id litellm minted, or the model the router already
    resolved for this request. The router's copy keeps litellm's provider
    prefix, which Ark does not take.
    """
    supplied: Final = _string_param((extra_body or _EMPTY_PARAMS).get("model"))
    if supplied is not None:
        return supplied
    model_id: Final = decode_video_id_with_provider(video_id).get("model_id")
    if model_id:
        return model_id
    routed: Final = _routed_model(litellm_params)
    if routed is not None:
        return routed.rsplit("/", 1)[-1]
    raise ValueError(
        "Editing and extending need the model that produced the source video. Pass the video id returned by "
        "video_generation() so it can be decoded, or set model in extra_body."
    )


def _source_video_url(video_id: str, extra_body: Mapping[str, object] | None) -> str:
    """
    Resolve the source video the edit or extension task should reference.

    Ark takes a video URL, not a task id. litellm's edit prefetch hook only
    issues POSTs and Ark's task query is a GET, so an id cannot be resolved
    here; the caller passes the URL itself, either as ``video_id`` or as
    ``video_url`` in ``extra_body``.
    """
    supplied: Final = (extra_body or _EMPTY_PARAMS).get("video_url")
    if isinstance(supplied, str) and supplied.strip():
        return supplied.strip()
    original: Final = extract_original_video_id(video_id)
    if original.startswith(("http://", "https://")):
        return original
    raise ValueError(
        "Volcengine video edit and extension need the source video URL, not a task id. Pass the URL as video_id or "
        "as video_url in extra_body; Ark's task query is a GET, so litellm's edit prefetch hook cannot resolve an id."
    )


def _source_video_content_item(video_url: str) -> dict[str, object]:
    return {"type": "video_url", "video_url": {"url": video_url}, "role": "reference_video"}


class VolcEngineVideoConfig(BaseVideoConfig):
    """
    Configuration class for Volcengine Ark video generation (Doubao-Seedance).

    Ark uses an async task API where:
    1. POST /api/v3/contents/generations/tasks creates a task and returns its id
    2. The task is polled via GET /api/v3/contents/generations/tasks/{id}
    3. On success the artifact is downloaded from the 24h signed content.video_url
    """

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig contract returns list
        return [  # mutable-ok: BaseVideoConfig contract returns list
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
        """
        Map OpenAI-style video params onto Ark's flat request fields.

        Mappings:
        - seconds -> duration (integer; -1 asks Ark to pick)
        - size -> ratio ("1280x720" snaps to "16:9") and resolution ("720p")
        - parameters -> merged in as a provider-specific block
        - content, ratio, resolution, duration, frames, seed, watermark,
          generate_audio and the rest pass through

        OpenAI-only fields Ark does not define (user, characters, image) are
        dropped. ``input_reference`` is consumed by
        transform_video_create_request, which turns it into a content item.
        """
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
                    raise ValueError("parameters must be an object of Ark request fields") from e
            else:
                mapped_params[key] = value
        return mapped_params

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
        return _normalized_api_base(get_volcengine_base_url(api_base))

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
        Build Ark's flat body around the typed ``content`` array.

        ``input_reference`` becomes a ``first_frame`` content item, base64
        encoded into a data URI when a file object is given rather than a URL.
        A caller-supplied ``content`` array always wins, as the roles on it are
        the only way to express the multi-modal reference, edit and extension
        jobs Ark accepts.
        """
        request_data: Final[dict[str, object]] = {  # mutable-ok: request body dict, JSON-serialized by the handler
            "model": model,
            "content": list(_content_items(prompt, video_create_optional_request_params)),
            **{
                key: video_create_optional_request_params[key]
                for key in _CREATE_PASSTHROUGH_KEYS
                if video_create_optional_request_params.get(key) is not None
            },
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
        task: Final = _TASK_CREATE_ADAPTER.validate_python(raw_response.json())
        if not task.id:
            message: Final = task.error.message if task.error is not None and task.error.message else "no task id"
            raise self.get_error_class(
                error_message=f"Volcengine did not return a video task id: {message}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
        video_obj: Final = VideoObject(id=task.id, object="video", status="queued", model=model)
        requested_usage: Final = _requested_usage(request_data, model)
        if requested_usage:
            video_obj.usage = dict(requested_usage)  # mutable-ok: VideoObject.usage is a dict field
            requested_seconds: Final = requested_usage.get("duration_seconds")
            video_obj.seconds = str(requested_seconds) if requested_seconds is not None else None
        size: Final = _size_from_geometry(
            _string_param((request_data or _EMPTY_PARAMS).get("resolution")),
            _string_param((request_data or _EMPTY_PARAMS).get("ratio")),
            model,
        )
        if size is not None:
            video_obj.size = size
        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, model)
        return video_obj

    def _task_url(self, video_id: str, api_base: str) -> str:
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
    ) -> VideoObject:
        task: Final = _parse_task(raw_response)
        video_obj: Final = _video_object_from_task(task)
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, None)
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
        video_url: Final = _video_url_from_task(_parse_task(raw_response))

        httpx_client: Final = _get_httpx_client()
        video_response: Final = httpx_client.get(video_url)  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler.get stub leaves params/headers untyped
        video_response.raise_for_status()

        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> bytes:
        video_url: Final = _video_url_from_task(_parse_task(raw_response))

        async_httpx_client: Final = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.VOLCENGINE,
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
            "Video remix is not supported by Volcengine. Edit or extend the video instead: /v1/videos/edits and "
            "/v1/videos/extensions map onto Ark's editing and extension task types."
        )

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
    ) -> tuple[str, Mapping[str, object], RequestFiles | None]:  # mutable-ok: BaseVideoConfig contract
        """
        Build Ark's video editing task.

        Editing is a create-task call that references the source video and asks
        Ark to resolve the geometry itself, so ``ratio`` is pinned to
        ``adaptive`` and ``duration`` to ``-1`` unless the caller overrides
        them, both of which Ark rejects the request without.
        """
        request_data: Final[dict[str, object]] = {  # mutable-ok: request body dict, JSON-serialized by the handler
            "model": _edit_model(video_id, extra_body, litellm_params),
            "content": [
                {"type": "text", "text": prompt},
                _source_video_content_item(_source_video_url(video_id, extra_body)),
            ],
            "ratio": "adaptive",
            _EDIT_DURATION_PARAM: -1,
        }
        if extra_body:
            request_data.update(extra_body)
        return f"{api_base}{VOLCENGINE_VIDEO_TASKS_PATH}", request_data, None

    def transform_video_edit_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
        request_data: dict[str, object] | None = None,  # mutable-ok: BaseVideoConfig contract
    ) -> VideoObject:
        task: Final = _TASK_CREATE_ADAPTER.validate_python(raw_response.json())
        edit_model: Final = request_data.get("model") if request_data is not None else None
        video_obj: Final = VideoObject(
            id=task.id,
            object="video",
            status="queued",
            model=edit_model if isinstance(edit_model, str) else None,
        )
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, video_obj.model)
        return video_obj

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
        """
        Build Ark's video extension task.

        Extension is the same create-task shape as editing minus the fixed
        duration: Ark requires ``ratio: adaptive`` and takes the number of
        seconds to add as ``duration``.
        """
        request_data: Final[dict[str, object]] = {  # mutable-ok: request body dict, JSON-serialized by the handler
            "model": _edit_model(video_id, extra_body, litellm_params),
            "content": [
                {"type": "text", "text": prompt},
                _source_video_content_item(_source_video_url(video_id, extra_body)),
            ],
            "ratio": "adaptive",
        }
        duration: Final = _duration_param(seconds)
        if duration is not None:
            request_data["duration"] = duration
        if extra_body:
            request_data.update(extra_body)
        return f"{api_base}{VOLCENGINE_VIDEO_TASKS_PATH}", request_data

    def transform_video_extension_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        task: Final = _TASK_CREATE_ADAPTER.validate_python(raw_response.json())
        video_obj: Final = VideoObject(id=task.id, object="video", status="queued")
        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(video_obj.id, custom_llm_provider, None)
        return video_obj

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
        """
        Map the video list request to GET /api/v3/contents/generations/tasks.

        Ark paginates with page_num/page_size and filters (``filter.status``
        etc.) instead of a cursor, so ``limit`` maps to page_size and the
        Ark-specific filters pass through ``extra_query``; ``after`` and
        ``order`` have no Ark equivalent.
        """
        params: Final[dict[str, object]] = {}  # mutable-ok: query params dict consumed by the http handler
        if limit is not None:
            params["page_size"] = str(limit)
        if extra_query:
            params.update(extra_query)
        return f"{api_base}{VOLCENGINE_VIDEO_TASKS_PATH}", params

    def transform_video_list_response(  # pyright: ignore[reportIncompatibleMethodOverride]  # base declares dict[str, str] but the payload is a heterogeneous list body
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
        custom_llm_provider: str | None = None,
    ) -> dict[str, object]:  # mutable-ok: OpenAI list body served as JSON by the proxy
        list_payload: Final = _TASK_LIST_ADAPTER.validate_python(raw_response.json())

        data: Final = [_encoded_task_dump(task, custom_llm_provider) for task in list_payload.items or ()]

        list_response: Final[dict[str, object]] = {  # mutable-ok: OpenAI list body served as JSON by the proxy
            "object": "list",
            "data": data,
            "total": list_payload.total if isinstance(list_payload.total, int) else len(data),
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
        return self._task_url(video_id, api_base), _EMPTY_PARAMS

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "Logging",
    ) -> VideoObject:
        try:
            task: Final = _parse_task(raw_response)
        except ValueError:
            return VideoObject(id="", object="video", status="deleted")
        return VideoObject(id=task.id, object="video", status="deleted", model=task.model)

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict[str, str] | httpx.Headers,  # mutable-ok: BaseVideoConfig contract
    ) -> BaseLLMException:
        return VolcEngineVideoError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
