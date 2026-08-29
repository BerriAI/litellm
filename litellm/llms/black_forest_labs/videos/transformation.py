"""
Black Forest Labs FLUX 3 Video Configuration

Handles transformation between OpenAI-compatible video params and the Black
Forest Labs FLUX 3 video API.

API Reference: https://docs.bfl.ai/api-reference/utility/generate-a-video-with-flux-3
"""

import time
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final  # noqa: TID251  # BaseVideoConfig types its payloads dict[str, Any]
from urllib.parse import urlparse

import httpx

import litellm
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.custom_httpx.http_handler import (
    AsyncHTTPHandler,
    HTTPHandler,
    _get_httpx_client,  # pyright: ignore[reportPrivateUsage]  # the shared client factory every provider uses to fetch generated media
    get_async_httpx_client,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

from ..common_utils import (
    DEFAULT_API_BASE,
    BlackForestLabsError,
    assert_bfl_polling_url,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj  # rebind-ok: the TYPE_CHECKING alias for the runtime Any below
else:
    LiteLLMLoggingObj = Any  # rebind-ok: runtime stand-in for the type-only Logging alias

VIDEO_MODELS: Final[Mapping[str, str]] = MappingProxyType({"flux-3-video": "/v1/flux-3-video"})

RESOLUTIONS: Final = ("hd", "fhd")
ASPECT_RATIOS: Final = ("21:9", "2:1", "16:9", "4:3", "1:1", "3:4", "9:16", "auto")
MIN_DURATION: Final = 5
MAX_DURATION: Final = 20

# BFL reports one of these on GET /v1/get_result. Anything outside the terminal
# set is still running.
_TERMINAL_STATUSES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Ready": "completed",
        "Error": "failed",
        "Content Moderated": "failed",
        "Request Moderated": "failed",
        "Task not found": "failed",
    }
)
_IN_PROGRESS_STATUSES: Final[Mapping[str, str]] = MappingProxyType(
    {
        "Pending": "queued",
        "Queued": "queued",
        "Reasoning": "in_progress",
        "Generating": "in_progress",
        "Uploading": "in_progress",
    }
)

# BFL dispatches each job to a regional host and hands that host back in
# ``polling_url``. The global host answers 404 for a regional job, so the region
# has to survive from submission through to status and content retrieval. The
# only value carried across those calls is the video id, so the region travels
# inside it, behind a separator BFL's own UUIDs never contain.
_REGION_SEPARATOR: Final = "@"


def _pack_region(job_id: str, polling_url: str | None) -> str:
    """Attach the polling host to a job id, when it differs from the default."""
    if not polling_url:
        return job_id
    host: Final = (urlparse(polling_url).hostname or "").lower()
    if not host or host == urlparse(DEFAULT_API_BASE).hostname:
        return job_id
    return f"{job_id}{_REGION_SEPARATOR}{host}"


def _unpack_region(job_id: str) -> tuple[str, str | None]:
    """Split a packed id back into the bare job id and its polling host."""
    if _REGION_SEPARATOR not in job_id:
        return job_id, None
    bare, _, host = job_id.partition(_REGION_SEPARATOR)
    return bare, host or None


class BlackForestLabsVideoConfig(BaseVideoConfig):
    """
    Configuration for Black Forest Labs FLUX 3 video generation.

    FLUX 3 is a single endpoint with a ``mode`` discriminator: ``t2v`` from a
    prompt alone, ``i2v`` from keyframe images, ``v2v`` to continue an existing
    clip, and ``draft_enhance`` to re-render a draft at full quality.

    Submission returns a job id plus a regional ``polling_url``. That URL has to
    be reused verbatim, because the global host answers 404 for a job dispatched
    to a region.
    """

    def get_supported_openai_params(self, model: str) -> list:  # mutable-ok: BaseVideoConfig signature
        return [  # mutable-ok: BaseVideoConfig signature
            "model",
            "seconds",
            "size",
            "input_reference",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict:  # mutable-ok: BaseVideoConfig signature
        """
        Map OpenAI video params onto FLUX 3 params.

        - ``seconds`` -> ``duration`` (whole seconds, 5 to 20)
        - ``size`` -> ``resolution`` tier, by the shorter side
        - ``input_reference`` -> a single ``keyframes`` entry, which selects i2v
        """
        supported: Final = self.get_supported_openai_params(model)
        remapped: Final = ("seconds", "size", "input_reference")

        duration: Final = self._map_duration(video_create_optional_params.get("seconds"))
        resolution: Final = self._map_size_to_resolution(video_create_optional_params.get("size"))
        input_reference: Final = video_create_optional_params.get("input_reference")

        translated: Final = {  # mutable-ok: JSON request payload
            key: value
            for key, value in (
                ("duration", duration),
                ("resolution", resolution),
                (
                    "keyframes",
                    [input_reference]  # mutable-ok: JSON request payload
                    if input_reference is not None
                    else None,  # mutable-ok: JSON request payload
                ),
            )
            if value is not None
        }
        passthrough: Final = {  # mutable-ok: JSON request payload
            key: value
            for key, value in video_create_optional_params.items()
            if key not in supported and key not in remapped
        }

        return {  # mutable-ok: JSON request payload
            **translated,
            **passthrough,
        }  # mutable-ok: JSON request payload

    def _map_duration(self, seconds: object) -> int | None:
        if not isinstance(seconds, (int, float, str)):
            return None
        try:
            duration: Final = int(float(seconds))
        except (TypeError, ValueError):
            return None
        return max(MIN_DURATION, min(MAX_DURATION, duration))

    def _map_size_to_resolution(self, size: object) -> str | None:
        """
        FLUX 3 takes a named tier, not pixel dimensions, so map by the shorter
        side: at most 720 is ``hd`` and anything larger is ``fhd``.
        """
        if not isinstance(size, str):
            return None
        if size in RESOLUTIONS:
            return size
        if "x" not in size.lower():
            return None
        try:
            width, height = (int(part) for part in size.lower().split("x", 1))
        except ValueError:
            return None
        return "hd" if min(width, height) <= 720 else "fhd"

    def validate_environment(
        self,
        headers: dict,  # mutable-ok: BaseVideoConfig signature
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:  # mutable-ok: BaseVideoConfig signature
        # Bind a new name rather than rebinding the caller's parameter.
        request_api_key: Final = api_key or (litellm_params.api_key if litellm_params else None)

        # Resolve BFL credentials only. Falling back to ``litellm.api_key`` would
        # send another provider's generic key to api.bfl.ai.
        final_api_key: Final = (
            request_api_key or get_secret_str("BFL_API_KEY") or get_secret_str("BLACK_FOREST_LABS_API_KEY")
        )

        if not final_api_key:
            raise BlackForestLabsError(
                status_code=401,
                message="BFL_API_KEY is not set. Please set it via environment variable or pass api_key parameter.",
            )

        headers.update(
            {  # mutable-ok: JSON request payload
                "x-key": final_api_key,
                "Content-Type": "application/json",
                "Accept": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,  # mutable-ok: BaseVideoConfig signature
    ) -> str:
        base_url: Final[str] = api_base or get_secret_str("BFL_API_BASE") or DEFAULT_API_BASE
        return base_url.rstrip("/")

    def _get_model_endpoint(self, model: str) -> str:
        model_name: Final = model.lower().split("/")[-1]
        if model_name in VIDEO_MODELS:
            return VIDEO_MODELS[model_name]
        raise ValueError(
            f"Unknown BFL video model: {model_name}. Supported models: {list(VIDEO_MODELS.keys())}"  # mutable-ok: one-shot, for an error message
        )

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,  # mutable-ok: BaseVideoConfig signature
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig signature
    ) -> tuple[dict, list, str]:  # mutable-ok: BaseVideoConfig signature
        base_request: Final = {  # mutable-ok: JSON request payload
            "prompt": prompt,
            **video_create_optional_request_params,
        }  # mutable-ok: JSON request payload
        mode: Final = self._infer_mode(base_request)

        # draft_enhance re-renders a cached draft, so it carries no prompt.
        request_data: Final = {  # mutable-ok: JSON request payload
            **{  # mutable-ok: JSON request payload
                key: value for key, value in base_request.items() if not (mode == "draft_enhance" and key == "prompt")
            },  # mutable-ok: JSON request payload
            "mode": mode,
        }

        url: Final = f"{api_base}{self._get_model_endpoint(model)}"
        return request_data, [], url  # mutable-ok: JSON request payload

    def _infer_mode(self, request_data: Mapping[str, Any]) -> str:
        """FLUX 3 discriminates on ``mode``; derive it from the inputs given."""
        if request_data.get("mode"):
            return str(request_data["mode"])
        if request_data.get("draft_cache"):
            return "draft_enhance"
        if request_data.get("start_video"):
            return "v2v"
        if request_data.get("keyframes"):
            return "i2v"
        return "t2v"

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,  # mutable-ok: BaseVideoConfig signature
    ) -> VideoObject:
        """
        Submission answers with the job handle, not the finished video:
        ``{"id": ..., "polling_url": ..., "cost": null}``.

        The regional polling URL is kept on the video id so status and content
        calls hit the same region.
        """
        response_data: Final = self._parse_json(raw_response)

        job_id: Final = response_data.get("id")
        if not job_id:
            raise BlackForestLabsError(
                status_code=raw_response.status_code,
                message=f"No job id in BFL response: {response_data}",
            )

        polling_url: Final = response_data.get("polling_url")
        if polling_url:
            assert_bfl_polling_url(polling_url)

        # The region has to outlive this response, and the id is the only value
        # the status and content calls receive.
        regional_job_id: Final = _pack_region(job_id, polling_url)

        video_obj: Final = VideoObject(
            id=regional_job_id,
            object="video",
            status="queued",
            created_at=int(time.time()),
            model=model,
        )

        if request_data:
            if request_data.get("duration") not in (None, "auto"):
                video_obj.seconds = str(request_data["duration"])
            if request_data.get("resolution"):
                video_obj.size = str(request_data["resolution"])

        if custom_llm_provider:
            video_obj.id = encode_video_id_with_provider(regional_job_id, custom_llm_provider, model)

        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig signature
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig signature
        return self._get_result_url(video_id, api_base), {}  # mutable-ok: JSON request payload

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        response_data: Final = self._parse_json(raw_response)
        bfl_status: Final = response_data.get("status", "Pending")

        video_obj: Final = VideoObject(
            id=response_data.get("id", ""),
            object="video",
            status=self._map_status(bfl_status),
            progress=self._map_progress(response_data.get("progress")),
        )

        if bfl_status in _TERMINAL_STATUSES and _TERMINAL_STATUSES[bfl_status] == "failed":
            video_obj.error = {  # mutable-ok: JSON request payload
                "code": bfl_status,
                "message": str(response_data.get("details") or bfl_status),
            }

        cost: Final = response_data.get("cost")
        if cost is not None:
            video_obj.usage = {  # mutable-ok: VideoObject field
                "credits": cost
            }  # mutable-ok: JSON request payload

        if custom_llm_provider and video_obj.id:
            # Re-attach the region: BFL echoes a bare job id, but the caller may
            # reuse this id for content retrieval and must land on the same host.
            polled_url: Final = self._polled_url(raw_response)
            video_obj.id = encode_video_id_with_provider(
                _pack_region(video_obj.id, polled_url), custom_llm_provider, None
            )

        return video_obj

    def _polled_url(self, raw_response: httpx.Response) -> str | None:
        """The URL this response came from, when httpx recorded one.

        ``Response.request`` raises rather than returning None when the response
        was built without a request, which is the case in unit tests.
        """
        try:
            return str(raw_response.request.url)
        except RuntimeError:
            return None

    def _map_status(self, bfl_status: str) -> str:
        if bfl_status in _TERMINAL_STATUSES:
            return _TERMINAL_STATUSES[bfl_status]
        return _IN_PROGRESS_STATUSES.get(bfl_status, "in_progress")

    def _map_progress(self, progress: object) -> int | None:
        if not isinstance(progress, (int, float, str)):
            return None
        try:
            value: Final = float(progress)
        except (TypeError, ValueError):
            return None
        # BFL reports a 0..1 fraction; VideoObject.progress is a percentage.
        return round(value * 100) if value <= 1 else round(value)

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig signature
        variant: str | None = None,
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig signature
        return self._get_result_url(video_id, api_base), {}  # mutable-ok: JSON request payload

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        video_url: Final = self._extract_video_url(self._parse_json(raw_response))
        httpx_client: Final[HTTPHandler] = _get_httpx_client()
        video_response: Final = httpx_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    async def async_transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        video_url: Final = self._extract_video_url(self._parse_json(raw_response))
        async_client: Final[AsyncHTTPHandler] = get_async_httpx_client(
            llm_provider=litellm.LlmProviders.BLACK_FOREST_LABS,
        )
        video_response: Final = await async_client.get(video_url)
        video_response.raise_for_status()
        return video_response.content

    def _extract_video_url(self, response_data: Mapping[str, Any]) -> str:
        status: Final = response_data.get("status", "Pending")
        result: Final[Mapping[str, Any]] = response_data.get("result") or MappingProxyType({})
        video_url: Final = result.get("sample")

        if video_url:
            return video_url

        if status in _TERMINAL_STATUSES:
            raise BlackForestLabsError(
                status_code=500,
                message=f"Video generation did not produce a video (status: {status}).",
            )
        raise BlackForestLabsError(
            status_code=409,
            message=f"Video is still processing (status: {status}). Please wait and try again.",
        )

    def _get_result_url(self, video_id: str, api_base: str) -> str:
        """Build the polling URL, honouring the region the job was dispatched to.

        The global host answers 404 for a regional job, so a packed id sends the
        request back to the host BFL named at submission.
        """
        original_video_id: Final = extract_original_video_id(video_id)
        job_id, region_host = _unpack_region(original_video_id)
        host: Final = f"https://{region_host}" if region_host else api_base.rstrip("/")
        result_url: Final = f"{host}/v1/get_result?id={job_id}"
        assert_bfl_polling_url(result_url)
        return result_url

    def _parse_json(self, raw_response: httpx.Response) -> dict:  # mutable-ok: decoded JSON body
        try:
            return raw_response.json()
        except Exception as e:
            raise BlackForestLabsError(
                status_code=raw_response.status_code,
                message=f"Error parsing BFL response: {e}",
            )

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig signature
        extra_body: dict[str, Any] | None = None,  # mutable-ok: BaseVideoConfig signature
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig signature
        raise NotImplementedError("video remix is not supported by the FLUX 3 video API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        raise NotImplementedError("video remix is not supported by the FLUX 3 video API")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig signature
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: dict[str, Any] | None = None,  # mutable-ok: BaseVideoConfig signature
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig signature
        raise NotImplementedError("video listing is not supported by the FLUX 3 video API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseVideoConfig signature
        raise NotImplementedError("video listing is not supported by the FLUX 3 video API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,  # mutable-ok: BaseVideoConfig signature
    ) -> tuple[str, dict]:  # mutable-ok: BaseVideoConfig signature
        raise NotImplementedError("video delete is not supported by the FLUX 3 video API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("video delete is not supported by the FLUX 3 video API")

    def get_error_class(
        self,
        error_message: str,
        status_code: int,
        headers: dict | httpx.Headers,  # mutable-ok: BaseVideoConfig signature
    ) -> BlackForestLabsError:
        return BlackForestLabsError(status_code=status_code, message=error_message)
