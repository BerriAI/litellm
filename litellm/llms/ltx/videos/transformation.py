import tempfile
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import httpx
from httpx._types import RequestFiles

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


LTX_VIDEO_STORAGE_DIR = Path(tempfile.gettempdir()) / "litellm_ltx_videos"
LTX_VIDEO_STORAGE_MAX_AGE_SECONDS = 24 * 60 * 60


def _get_ltx_video_storage_path(video_id: str) -> Path:
    if not video_id or any(char in video_id for char in ("/", "\\", "\0")):
        raise BaseLLMException(
            status_code=400,
            message="Invalid LTX video_id. video_id must not contain path separators.",
        )

    return LTX_VIDEO_STORAGE_DIR / f"{video_id}.mp4"


def _cleanup_old_ltx_videos() -> None:
    if not LTX_VIDEO_STORAGE_DIR.exists():
        return

    cutoff = time.time() - LTX_VIDEO_STORAGE_MAX_AGE_SECONDS
    for video_path in LTX_VIDEO_STORAGE_DIR.glob("*.mp4"):
        try:
            if video_path.stat().st_mtime < cutoff:
                video_path.unlink(missing_ok=True)
        except OSError:
            pass


def _persist_ltx_video_bytes(video_id: str, video_bytes: bytes) -> Path:
    _cleanup_old_ltx_videos()
    LTX_VIDEO_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    video_path = _get_ltx_video_storage_path(video_id)
    video_path.write_bytes(video_bytes)
    return video_path


class LTXVideoConfig(BaseVideoConfig):
    """
    Configuration class for LTX Video generation.

    LTX Video API is synchronous — it returns binary video data directly
    in the response rather than a task ID to poll.

    Supports two endpoints:
    - POST /v1/text-to-video (text prompt only)
    - POST /v1/image-to-video (text prompt + source image)
    """

    def __init__(self):
        super().__init__()

    def requires_authentication_for_video_content(self) -> bool:
        return False

    def get_supported_openai_params(self, model: str) -> list:
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        mapped_params: Dict[str, Any] = {}

        if "input_reference" in video_create_optional_params:
            mapped_params["image_uri"] = video_create_optional_params["input_reference"]

        if "size" in video_create_optional_params:
            size = video_create_optional_params["size"]
            if isinstance(size, str):
                mapped_params["resolution"] = size

        if "seconds" in video_create_optional_params:
            seconds = video_create_optional_params["seconds"]
            if seconds is not None:
                try:
                    mapped_params["duration"] = (
                        int(float(seconds))
                        if isinstance(seconds, str)
                        else int(seconds)
                    )
                except (ValueError, TypeError):
                    # Ignore invalid seconds values and let the provider fall back to defaults.
                    pass

        if "user" in video_create_optional_params:
            if drop_params:
                pass
            else:
                raise ValueError(
                    f"Parameter user is not supported for model {model}. "
                    "Set drop_params=True to drop unsupported parameters."
                )

        # Pass through LTX-specific parameters
        supported_openai_params = self.get_supported_openai_params(model)
        for key, value in video_create_optional_params.items():
            if key != "user" and key not in supported_openai_params:
                mapped_params[key] = value

        return mapped_params

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        api_key = (
            api_key
            or (litellm_params.api_key if litellm_params else None)
            or litellm.api_key
            or get_secret_str("LTX_API_KEY")
        )

        if api_key is None:
            raise ValueError(
                "LTX API key is required. Set LTX_API_KEY environment variable "
                "or pass api_key parameter."
            )

        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Return the shared LTX API base URL.

        The final create endpoint depends on whether the request includes
        an input image, so `transform_video_create_request()` appends the
        `/text-to-video` or `/image-to-video` suffix.
        """
        if api_base is None:
            api_base = "https://api.ltx.video/v1"

        return api_base.rstrip("/")

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        request_data: Dict[str, Any] = {
            "prompt": prompt,
            "model": model,
        }

        # LTX's image-to-video endpoint accepts image_uri in the JSON body.
        # See docs.ltx.video/input-formats and the /v1/image-to-video reference.
        request_data.update(video_create_optional_request_params)

        files_list: List[Tuple[str, Any]] = []

        # Choose endpoint based on whether image_uri is present
        if "image_uri" in request_data:
            full_api_base = f"{api_base}/image-to-video"
        else:
            full_api_base = f"{api_base}/text-to-video"

        return request_data, files_list, full_api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
    ) -> VideoObject:
        """
        Transform the LTX video creation response.

        LTX returns binary video data directly (application/octet-stream).
        We generate a UUID for the video ID and set status to "completed".
        """
        if not raw_response.content:
            raise BaseLLMException(
                status_code=raw_response.status_code or 502,
                message="LTX returned an empty video response body.",
                request=raw_response.request,
                response=raw_response,
            )

        video_id = str(uuid.uuid4())
        created_at = int(time.time())
        stored_video_path = _persist_ltx_video_bytes(
            video_id=video_id, video_bytes=raw_response.content
        )

        video_data: Dict[str, Any] = {
            "id": video_id,
            "object": "video",
            "status": "completed",
            "created_at": created_at,
            "completed_at": created_at,
        }

        if request_data:
            if "model" in request_data:
                video_data["model"] = request_data["model"]
            if "resolution" in request_data:
                video_data["size"] = request_data["resolution"]
            if "duration" in request_data:
                video_data["seconds"] = str(request_data["duration"])

        video_obj = VideoObject(**video_data)  # type: ignore[arg-type]

        if custom_llm_provider and video_obj.id:
            video_obj.id = encode_video_id_with_provider(
                video_obj.id, custom_llm_provider, model
            )

        usage_data: Dict[str, Any] = {}
        if request_data and "duration" in request_data:
            try:
                # completion_cost() reads usage.duration_seconds for video billing.
                usage_data["duration_seconds"] = float(request_data["duration"])
            except (ValueError, TypeError):
                # Duration is used only for cost accounting, so skip invalid values.
                pass
        video_obj.usage = usage_data
        video_obj._hidden_params = {
            "video_content_path": str(stored_video_path),
            "video_content_url": stored_video_path.resolve().as_uri(),
        }

        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: Optional[str] = None,
    ) -> Tuple[str, Dict]:
        if variant is not None:
            raise NotImplementedError(
                "LTX video content variants are not supported. "
                "Only the generated MP4 can be retrieved."
            )

        original_video_id = extract_original_video_id(video_id)
        stored_video_path = _get_ltx_video_storage_path(original_video_id)
        if not stored_video_path.exists():
            raise BaseLLMException(
                status_code=404,
                message=(
                    "No locally stored LTX video content was found for this video_id. "
                    "LTX stores generated video content in the local temp directory of "
                    "the process that created it, so retrieval will fail from a different "
                    "instance or after a process restart. Recreate the video before "
                    "calling video_content()."
                ),
            )

        return stored_video_path.resolve().as_uri(), {}

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        return raw_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video remix is not supported by LTX API")

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError("Video remix is not supported by LTX API")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video listing is not supported by LTX API")

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        raise NotImplementedError("Video listing is not supported by LTX API")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError("Video deletion is not supported by LTX API")

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        raise NotImplementedError("Video deletion is not supported by LTX API")

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        raise NotImplementedError(
            "Video status retrieval is not supported by LTX API. "
            "LTX video generation is synchronous."
        )

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        raise NotImplementedError(
            "Video status retrieval is not supported by LTX API. "
            "LTX video generation is synchronous."
        )

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
