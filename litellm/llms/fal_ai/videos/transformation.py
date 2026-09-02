"""Video generation for fal.ai through its queue API (https://queue.fal.run)."""

import base64
import mimetypes
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, TypeAlias, runtime_checkable

import httpx
from httpx._types import RequestFiles
from typing_extensions import ReadOnly, TypedDict

import litellm
from litellm._logging import verbose_logger
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    decode_video_id_with_provider,
    encode_video_id_with_provider,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj

FAL_QUEUE_API_BASE: Final = "https://queue.fal.run"

_PROVIDER_PREFIX: Final = f"{litellm.LlmProviders.FAL_AI.value}/"

_SUPPORTED_OPENAI_PARAMS: Final = ("model", "prompt", "input_reference", "seconds", "size", "user", "extra_headers")

_STATUS_MAP: Final[Mapping[str, str]] = MappingProxyType(
    {"IN_QUEUE": "queued", "IN_PROGRESS": "in_progress", "COMPLETED": "completed"}
)

_PENDING_STATUSES: Final = frozenset({"IN_QUEUE", "IN_PROGRESS"})

_EMPTY: Final[Mapping[str, object]] = MappingProxyType({})

_VIDEO_REFERENCE_ENDPOINTS: Final = frozenset({"video-to-video", "video-edit", "edit-video", "extend-video"})

_NO_DURATION_ENDPOINTS: Final = frozenset({"video-edit", "edit-video", "extend-video"})


@dataclass(frozen=True, slots=True)
class _FamilyRules:
    default_seconds: int = 5
    takes_duration: bool = True
    duration_as_string: bool = False
    resolution_upper: bool = False


_DEFAULT_FAMILY_RULES: Final = _FamilyRules()

_FAMILY_RULES: Final[Mapping[str, _FamilyRules]] = MappingProxyType(
    {
        "minimax/h3": _FamilyRules(resolution_upper=True),
        "bytedance/seedance-2.0": _FamilyRules(duration_as_string=True),
        "xai/grok-imagine-video": _FamilyRules(default_seconds=6),
        "fal-ai/ltx-video": _FamilyRules(takes_duration=False),
    }
)


class _FalQueueResponse(TypedDict, total=False):
    request_id: ReadOnly[str]
    status: ReadOnly[str]
    response_url: ReadOnly[str]
    error: ReadOnly[object]
    detail: ReadOnly[object]


class _FalResult(TypedDict, total=False):
    video: ReadOnly[Mapping[str, object] | str]
    videos: ReadOnly[Sequence[Mapping[str, object] | str]]
    status: ReadOnly[str]
    error: ReadOnly[object]
    detail: ReadOnly[object]


@runtime_checkable
class _Readable(Protocol):
    def read(self) -> object: ...


@runtime_checkable
class _Seekable(Protocol):
    def seek(self, offset: int, /) -> int: ...


@dataclass(frozen=True, slots=True)
class _PollTarget:
    model_path: str
    request_id: str
    headers: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class _VideoReady:
    url: str


@dataclass(frozen=True, slots=True)
class _VideoPending:
    status: str


@dataclass(frozen=True, slots=True)
class _VideoFailed:
    detail: str


_ResultState: TypeAlias = _VideoReady | _VideoPending | _VideoFailed


class FalAIVideoError(BaseLLMException):
    pass


def _model_path(model: str) -> str:
    return model.removeprefix(_PROVIDER_PREFIX).strip("/")


def _root_app_id(model_path: str) -> str:
    return "/".join(model_path.split("/")[:2])


def _endpoint(model_path: str) -> str:
    return model_path.rsplit("/", 1)[-1]


def _rules(model_path: str) -> _FamilyRules:
    return _FAMILY_RULES.get(_root_app_id(model_path), _DEFAULT_FAMILY_RULES)


def _takes_duration(model_path: str) -> bool:
    return _rules(model_path).takes_duration and _endpoint(model_path) not in _NO_DURATION_ENDPOINTS


def _seconds_or_default(model_path: str, seconds: object) -> int:
    try:
        return int(float(str(seconds)))
    except ValueError:
        return _rules(model_path).default_seconds


def _duration_value(model_path: str, seconds: int) -> str | int:
    return str(seconds) if _rules(model_path).duration_as_string else seconds


def _size_to_resolution(model_path: str, size: object) -> str | None:
    if not isinstance(size, str) or "x" not in size.lower():
        return None
    width_str, _, height_str = size.lower().partition("x")
    if not (width_str.isdigit() and height_str.isdigit()):
        return None
    short_side: Final = min(int(width_str), int(height_str))
    if not _rules(model_path).resolution_upper:
        return f"{short_side}p"
    if short_side >= 2160:
        return "4K"
    if short_side >= 1440:
        return "2K"
    return f"{short_side}P"


def _split_reference(reference: object) -> tuple[str | None, object, str | None]:
    if not isinstance(reference, (tuple, list)):
        name: Final = getattr(reference, "name", None)
        return (name if isinstance(name, str) else None), reference, None
    parts: Final[tuple[object, ...]] = (*reference, None, None, None)
    filename: Final = parts[0] if isinstance(parts[0], str) else None
    content_type: Final = parts[2] if isinstance(parts[2], str) else None
    return filename, parts[1], content_type


def _read_bytes(payload: object) -> bytes | None:
    if isinstance(payload, (bytes, bytearray)):
        return bytes(payload)
    if not isinstance(payload, _Readable):
        return None
    if isinstance(payload, _Seekable):
        payload.seek(0)
    data: Final = payload.read()
    return data if isinstance(data, bytes) else None


def _content_type(filename: str | None, declared: str | None, data: bytes, default_mime: str) -> str:
    if declared:
        return declared
    guessed: Final = mimetypes.guess_type(filename)[0] if filename else None
    if guessed:
        return guessed
    if default_mime.startswith("video/"):
        return default_mime
    return ImageEditRequestUtils.get_image_content_type(data)


def _reference_url(reference: object, default_mime: str) -> object:
    if reference is None or isinstance(reference, str):
        return reference
    filename, payload, declared = _split_reference(reference)
    data: Final = _read_bytes(payload)
    if data is None:
        return reference
    content_type: Final = _content_type(filename, declared, data, default_mime)
    return f"data:{content_type};base64,{base64.b64encode(data).decode('ascii')}"


def _url_of(item: Mapping[str, object] | str | None) -> str | None:
    if isinstance(item, str):
        return item or None
    if item is None:
        return None
    url: Final = item.get("url")
    return url if isinstance(url, str) and url else None


def _video_url(result: _FalResult) -> str | None:
    videos: Final = result.get("videos")
    return _url_of(result.get("video")) or _url_of(videos[0] if videos else None)


def _result_state(result: _FalResult) -> _ResultState:
    url: Final = _video_url(result)
    if url:
        return _VideoReady(url)
    status: Final = str(result.get("status", "")).upper()
    if status in _PENDING_STATUSES:
        return _VideoPending(status)
    detail: Final = result.get("error") or result.get("detail")
    return _VideoFailed(str(detail) if detail else f"video url not found in result keys {sorted(result)}")


def _error_detail(response: httpx.Response) -> str:
    try:
        body: Final[_FalQueueResponse] = response.json()
    except ValueError:
        return response.text[:500]
    detail: Final = body.get("error") or body.get("detail")
    return str(detail) if detail else response.text[:500]


class FalAIVideoConfig(BaseVideoConfig):
    """
    fal.ai runs every model behind one async queue:
    1. POST /<model path> submits the job and returns a request id
    2. GET /<root app id>/requests/<id>/status reports IN_QUEUE, IN_PROGRESS or COMPLETED
    3. GET /<root app id>/requests/<id> returns the result payload with the video url

    The root app id is the first two segments of the model path. The full model
    path is encoded into the video id so status and content calls can rebuild
    the queue URLs and the proxy can route them back to the same deployment.
    """

    def __init__(self, result_client: HTTPHandler | None = None) -> None:
        super().__init__()
        self._result_client = result_client
        self._poll_target: _PollTarget | None = None

    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig contract
        return list(_SUPPORTED_OPENAI_PARAMS)  # mutable-ok: BaseVideoConfig returns list

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: BaseVideoConfig contract; the caller merges extra_body into this dict
        model_path: Final = _model_path(model)
        return {  # mutable-ok: VideoGenerationRequestUtils.update/pop extra_body onto this mapping
            **self._reference_param(model_path, video_create_optional_params),
            **self._duration_param(model_path, video_create_optional_params),
            **self._resolution_param(model_path, video_create_optional_params),
            **self._passthrough_params(video_create_optional_params),
        }

    @staticmethod
    def _passthrough_params(params: VideoCreateOptionalRequestParams) -> Mapping[str, object]:
        return MappingProxyType({key: value for key, value in params.items() if key not in _SUPPORTED_OPENAI_PARAMS})

    @staticmethod
    def _reference_param(model_path: str, params: VideoCreateOptionalRequestParams) -> Mapping[str, object]:
        if "input_reference" not in params:
            return _EMPTY
        reference: Final = params["input_reference"]
        if _endpoint(model_path) in _VIDEO_REFERENCE_ENDPOINTS:
            return MappingProxyType({"video_url": _reference_url(reference, "video/mp4")})
        return MappingProxyType({"image_url": _reference_url(reference, "image/png")})

    @staticmethod
    def _duration_param(model_path: str, params: VideoCreateOptionalRequestParams) -> Mapping[str, object]:
        seconds: Final = params.get("seconds")
        if seconds is None or not _takes_duration(model_path):
            return _EMPTY
        return MappingProxyType({"duration": _duration_value(model_path, _seconds_or_default(model_path, seconds))})

    @staticmethod
    def _resolution_param(model_path: str, params: VideoCreateOptionalRequestParams) -> Mapping[str, object]:
        resolution: Final = _size_to_resolution(model_path, params.get("size"))
        return _EMPTY if resolution is None else MappingProxyType({"resolution": resolution})

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract
        resolved_key: Final = (
            api_key
            or (litellm_params.api_key if litellm_params is not None else None)
            or litellm.api_key
            or get_secret_str("FAL_AI_API_KEY")
            or get_secret_str("FAL_KEY")
        )
        if resolved_key is None:
            raise ValueError("fal.ai API key is required. Set FAL_AI_API_KEY or pass api_key")
        return {  # mutable-ok: httpx headers are a dict
            **headers,
            "Authorization": f"Key {resolved_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: Mapping[str, object],
    ) -> str:
        return (api_base or FAL_QUEUE_API_BASE).rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, object], RequestFiles, str]:  # mutable-ok: BaseVideoConfig contract
        model_path: Final = _model_path(model)
        request_data: Final[dict[str, object]] = {  # mutable-ok: httpx json body must be a dict
            "prompt": prompt,
            **video_create_optional_request_params,
            **self._pinned_duration(model_path, video_create_optional_request_params),
        }
        return request_data, (), f"{api_base}/{model_path}"

    @staticmethod
    def _pinned_duration(model_path: str, request_params: Mapping[str, object]) -> Mapping[str, object]:
        if "duration" in request_params or not _takes_duration(model_path):
            return _EMPTY
        return MappingProxyType({"duration": _duration_value(model_path, _rules(model_path).default_seconds)})

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        response_data: Final[_FalQueueResponse] = raw_response.json()
        model_path: Final = _model_path(model)
        request_id: Final = response_data.get("request_id")
        if not request_id:
            raise ValueError(f"fal.ai submit response contains no request_id: {response_data}")
        request_params: Final = request_data or _EMPTY
        billed_seconds: Final = _seconds_or_default(model_path, request_params.get("duration"))
        resolution: Final = request_params.get("resolution")
        video_resolution: Final = resolution.lower() if isinstance(resolution, str) else None
        return VideoObject(
            id=self._encoded_id(request_id, custom_llm_provider, model_path),
            object="video",
            status=_STATUS_MAP.get(str(response_data.get("status", "IN_QUEUE")).upper(), "queued"),
            created_at=int(time.time()),
            model=model_path,
            seconds=str(billed_seconds),
            size=video_resolution,
            usage={  # mutable-ok: VideoObject.usage is a dict field
                key: value
                for key, value in (
                    ("duration_seconds", float(billed_seconds)),
                    ("video_resolution", video_resolution),
                )
                if value is not None
            },
        )

    @staticmethod
    def _encoded_id(request_id: str, custom_llm_provider: str | None, model_path: str | None) -> str:
        if not custom_llm_provider:
            return request_id
        return encode_video_id_with_provider(request_id, custom_llm_provider, model_path)

    def _remember_poll_target(self, video_id: str, headers: Mapping[str, str]) -> _PollTarget:
        decoded: Final = decode_video_id_with_provider(video_id)
        model_path: Final = decoded.get("model_id")
        if not model_path:
            raise ValueError(f"fal.ai video id '{video_id}' does not carry a model path, cannot build the queue URL")
        target: Final = _PollTarget(
            model_path=_model_path(model_path),
            request_id=decoded.get("video_id") or video_id,
            headers=MappingProxyType(dict(headers)),  # mutable-ok: snapshot of the handler's header dict
        )
        self._poll_target = target
        return target

    @staticmethod
    def _request_url(api_base: str, target: _PollTarget) -> str:
        encoded_request_id: Final = encode_url_path_segment(target.request_id, field_name="video_id")
        return f"{api_base}/{_root_app_id(target.model_path)}/requests/{encoded_request_id}"

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, str]]:  # mutable-ok: BaseVideoConfig contract
        target: Final = self._remember_poll_target(video_id, headers)
        return f"{self._request_url(api_base, target)}/status", {}  # mutable-ok: BaseVideoConfig contract

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response_data: Final[_FalQueueResponse] = raw_response.json()
        raw_status: Final = str(response_data.get("status", "")).upper()
        mapped_status: Final = _STATUS_MAP.get(raw_status, "failed")
        failure: Final = self._completed_job_failure(response_data) if mapped_status == "completed" else None
        status: Final = "failed" if failure is not None else mapped_status
        target: Final = self._poll_target
        request_id: Final = response_data.get("request_id") or (target.request_id if target else "")
        now: Final = int(time.time())
        return VideoObject(
            id=self._encoded_id(request_id, custom_llm_provider, target.model_path if target else None),
            object="video",
            status=status,
            created_at=now,
            completed_at=now if status == "completed" else None,
            error=self._error(raw_status, failure, response_data) if status == "failed" else None,
        )

    @staticmethod
    def _error(
        raw_status: str, failure: str | None, response_data: _FalQueueResponse
    ) -> dict[str, str]:  # mutable-ok: VideoObject.error is a dict field
        detail: Final = response_data.get("error") or response_data.get("detail")
        return {  # mutable-ok: VideoObject.error is a dict field
            "code": raw_status or "unknown",
            "message": failure or (str(detail) if detail else "fal.ai video job failed"),
        }

    def _completed_job_failure(self, response_data: _FalQueueResponse) -> str | None:
        """
        fal reports COMPLETED for failed jobs too. Only the result payload tells a
        finished video from a failure, so peek at it and downgrade the status
        instead of handing the caller an opaque error on the content fetch.
        """
        result_url: Final = response_data.get("response_url")
        target: Final = self._poll_target
        if not result_url or target is None:
            return None
        try:
            result: Final = (self._result_client or litellm.module_level_client).get(
                result_url,
                headers=dict(target.headers),  # mutable-ok: httpx headers are a dict
            )
        except Exception as exc:
            verbose_logger.warning("fal.ai result peek failed, keeping status completed: %s", exc)
            return None
        if result.status_code >= 400:
            return f"fal.ai video generation failed: {_error_detail(result)}"
        match _result_state(result.json()):
            case _VideoFailed(detail):
                return f"fal.ai video generation failed: {detail}"
            case _VideoReady() | _VideoPending():
                return None

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        variant: str | None = None,
    ) -> tuple[str, dict[str, str]]:  # mutable-ok: BaseVideoConfig contract
        target: Final = self._remember_poll_target(video_id, headers)
        return self._request_url(api_base, target), {}  # mutable-ok: BaseVideoConfig contract

    @staticmethod
    def _ready_video_url(raw_response: httpx.Response) -> str:
        match _result_state(raw_response.json()):
            case _VideoReady(url):
                return url
            case _VideoPending(status):
                raise ValueError(f"Video is still processing (status: {status}). Please wait and try again.")
            case _VideoFailed(detail):
                raise ValueError(f"fal.ai video generation failed: {detail}")

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> bytes:
        video_url: Final = self._ready_video_url(raw_response)
        video_response: Final = litellm.module_level_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> bytes:
        video_url: Final = self._ready_video_url(raw_response)
        async_client: Final[AsyncHTTPHandler] = get_async_httpx_client(llm_provider=litellm.LlmProviders.FAL_AI)
        video_response: Final = await async_client.get(video_url)
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
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("video remix is not supported for fal.ai")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video remix is not supported for fal.ai")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("video list is not supported for fal.ai")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("video list is not supported for fal.ai")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, dict[str, object]]:  # mutable-ok: BaseVideoConfig contract
        raise NotImplementedError("video delete is not supported for fal.ai")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: "LiteLLMLoggingObj",
    ) -> VideoObject:
        raise NotImplementedError("video delete is not supported for fal.ai")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: Mapping[str, str] | httpx.Headers,
    ) -> BaseLLMException:
        return FalAIVideoError(
            status_code=status_code,
            message=error_message,
            headers=headers if isinstance(headers, httpx.Headers) else dict(headers),  # mutable-ok: exception contract
        )
