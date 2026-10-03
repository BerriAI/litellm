import base64
import mimetypes
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Final, Protocol, runtime_checkable

import httpx
from httpx._types import RequestFiles
from pydantic import ValidationError

import litellm
from litellm.constants import DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.vertex_ai.videos.transformation import veo_video_count_from_parameters
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.gemini import (
    GeminiLongRunningOperationResponse,
    GeminiVideoGenerationInstance,
    GeminiVideoGenerationParameters,
    GeminiVideoGenerationRequest,
)
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any

_VEO_MAX_REFERENCE_IMAGES: Final = 3
_VEO_EIGHT_SECOND_RESOLUTIONS: Final = frozenset({"1080p", "4k"})


def _convert_image_to_gemini_format(image_file) -> dict[str, str]:
    """
    Convert image file to Gemini format with base64 encoding and MIME type.

    Args:
        image_file: File-like object opened in binary mode (e.g., open("path", "rb"))

    Returns:
        Dict with bytesBase64Encoded and mimeType
    """
    mime_type: Final = ImageEditRequestUtils.get_image_content_type(image_file)

    if hasattr(image_file, "seek"):
        image_file.seek(0)
    image_bytes: Final = image_file.read()
    base64_encoded: Final = base64.b64encode(image_bytes).decode("utf-8")

    return {"bytesBase64Encoded": base64_encoded, "mimeType": mime_type}


@runtime_checkable
class _BinaryFile(Protocol):
    def read(self) -> bytes: ...


@runtime_checkable
class _Seekable(Protocol):
    def seek(self, offset: int, /) -> int: ...


def _convert_video_to_gemini_format(video_file: _BinaryFile) -> Mapping[str, str]:
    """The MIME type comes from the file name, falling back to video/mp4 (the format Veo generates)"""
    guessed_type: Final = mimetypes.guess_type(str(getattr(video_file, "name", "")))[0]
    mime_type: Final = guessed_type if guessed_type and guessed_type.startswith("video/") else "video/mp4"
    if isinstance(video_file, _Seekable):
        video_file.seek(0)
    return MappingProxyType(
        {"bytesBase64Encoded": base64.b64encode(video_file.read()).decode("utf-8"), "mimeType": mime_type}
    )


def _to_gemini_media(media: object, is_video: bool = False) -> object:
    if not isinstance(media, _BinaryFile):
        return media
    return _convert_video_to_gemini_format(media) if is_video else _convert_image_to_gemini_format(media)


def _to_gemini_reference_image(reference: object) -> object:
    if isinstance(reference, Mapping) and ("image" in reference or "referenceType" in reference):
        return MappingProxyType({**reference, "image": _to_gemini_media(reference.get("image"))})
    return MappingProxyType({"image": _to_gemini_media(reference)})


def _bad_request(message: str, model: str) -> litellm.BadRequestError:
    return litellm.BadRequestError(message=f"Gemini Veo: {message}", model=model, llm_provider="gemini")


def _is_duration_eight_seconds(duration: object) -> bool:
    if not isinstance(duration, (int, float, str)):
        return False
    try:
        return float(duration) == 8
    except ValueError:
        return False


def _validate_veo_request(
    model: str, instance: GeminiVideoGenerationInstance, parameters: Mapping[str, object]
) -> None:
    """
    Reject combinations the Gemini API documents as invalid before they reach Google.
    See https://ai.google.dev/gemini-api/docs/veo#veo-model-parameters
    """
    has_image: Final = instance.image is not None
    has_video: Final = instance.video is not None
    reference_images: Final = instance.referenceImages or ()
    has_reference_images: Final = len(reference_images) > 0
    resolution: Final = str(parameters.get("resolution") or "").strip().lower()

    if instance.lastFrame is not None and not has_image:
        raise _bad_request("lastFrame requires image (the first frame) to be set.", model)
    if has_reference_images and has_image:
        raise _bad_request("referenceImages cannot be combined with image.", model)
    if has_video and has_image:
        raise _bad_request("video (extension) cannot be combined with image.", model)
    if len(reference_images) > _VEO_MAX_REFERENCE_IMAGES:
        raise _bad_request(
            f"at most {_VEO_MAX_REFERENCE_IMAGES} referenceImages are allowed, got {len(reference_images)}.", model
        )
    for reference in reference_images:
        if reference.referenceType.lower() != "asset":
            raise _bad_request(
                f"referenceType '{reference.referenceType}' is not supported. The Gemini API only accepts 'asset' "
                "reference images ('style' references are Vertex AI only).",
                model,
            )

    if has_video and resolution not in ("", "720p"):
        raise _bad_request(f"video extension only supports 720p resolution, got '{resolution}'.", model)

    duration: Final = parameters.get("durationSeconds")
    if duration is not None and not _is_duration_eight_seconds(duration):
        eight_second_reasons: Final = tuple(
            reason
            for reason, applies in (
                ("referenceImages", has_reference_images),
                ("video extension", has_video),
                (f"{resolution} resolution", resolution in _VEO_EIGHT_SECOND_RESOLUTIONS),
            )
            if applies
        )
        if eight_second_reasons:
            raise _bad_request(
                f"durationSeconds must be 8 when using {', '.join(eight_second_reasons)}, got {duration}.", model
            )

    person_generation: Final = parameters.get("personGeneration")
    if person_generation is not None and (has_image or has_reference_images) and person_generation != "allow_adult":
        raise _bad_request(
            f"personGeneration must be 'allow_adult' for image-to-video, interpolation and "
            f"referenceImages requests, got '{person_generation}'.",
            model,
        )


def _json_payload(raw_response: httpx.Response) -> object:
    """Read an HTTP response body as an opaque JSON payload."""
    return raw_response.json()


def _usage_video_resolution_from_parameters(
    parameters: Mapping[str, object],
) -> str | None:
    """Normalize Veo ``parameters.resolution`` for usage and cost tracking."""
    res: Final = parameters.get("resolution")
    if res is None or res == "":
        return None
    return str(res).strip().lower()


class GeminiVideoConfig(BaseVideoConfig):
    """
    Configuration class for Gemini (Veo) video generation.

    Veo uses a long-running operation model:
    1. POST to :predictLongRunning returns operation name
    2. Poll operation until done=true
    3. Extract video URI from response
    4. Download video using file API
    """

    _OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO: dict[str, str] = {
        "1280x720": "16:9",
        "1920x1080": "16:9",
        "720x1280": "9:16",
        "1080x1920": "9:16",
    }

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported OpenAI parameters for Veo video generation.
        Veo supports minimal parameters compared to OpenAI.
        """
        return ["model", "prompt", "input_reference", "seconds", "size"]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        """
        Map OpenAI-style parameters to Veo format.

        Mappings:
        - prompt → prompt
        - input_reference → image
        - size → aspectRatio (e.g., "1280x720" → "16:9")
        - size → resolution when inferable ("1280x720"/"720x1280" → "720p",
          "1920x1080"/"1080x1920" → "1080p"); skipped if ``resolution`` is already set
        - seconds → durationSeconds (defaults to 4 seconds if not provided)

        All other params are passed through as-is to support Gemini-specific parameters.
        """
        mapped_params: Final[dict[str, object]] = {}

        # Get supported OpenAI params (exclude "model" and "prompt" which are handled separately)
        supported_openai_params: Final = self.get_supported_openai_params(model)
        openai_params_to_map: Final = {param for param in supported_openai_params if param not in {"model", "prompt"}}

        # Map input_reference to image
        if "input_reference" in video_create_optional_params:
            mapped_params["image"] = video_create_optional_params["input_reference"]

        # Map size to aspectRatio
        if "size" in video_create_optional_params:
            size: Final = video_create_optional_params["size"]
            if size is not None:
                aspect_ratio: Final = self._convert_size_to_aspect_ratio(size)
                if aspect_ratio:
                    mapped_params["aspectRatio"] = aspect_ratio
                if not video_create_optional_params.get("resolution"):
                    inferred_resolution: Final = self._convert_size_to_resolution(size)
                    if inferred_resolution is not None:
                        mapped_params["resolution"] = inferred_resolution

        # Map seconds to durationSeconds, default to 4 seconds (matching OpenAI)
        if "seconds" in video_create_optional_params:
            seconds: Final = video_create_optional_params["seconds"]
            try:
                duration: Final = int(seconds) if isinstance(seconds, str) else seconds
                if duration is not None:
                    mapped_params["durationSeconds"] = duration
            except (ValueError, TypeError):
                # If conversion fails, use default
                pass

        # Pass through any other params that weren't mapped (Gemini-specific params)
        for key, value in video_create_optional_params.items():
            if key not in openai_params_to_map and key not in mapped_params:
                mapped_params[key] = value

        return mapped_params

    def _convert_size_to_aspect_ratio(self, size: str) -> str | None:
        """
        Convert OpenAI size format to Veo aspectRatio format.

        https://cloud.google.com/vertex-ai/generative-ai/docs/image/generate-videos

        Supported aspect ratios: 9:16 (portrait), 16:9 (landscape)
        """
        if not size:
            return None

        return self._OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO.get(size, "16:9")

    def _convert_size_to_resolution(self, size: str) -> str | None:
        """
        Map OpenAI ``size`` (WxH) to Veo ``resolution`` for presets in
        ``_OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO`` (720p / 1080p from the smaller edge).

        Unknown sizes return None so the API default applies (no forced resolution).
        """
        if not size or size not in self._OPENAI_VIDEO_SIZE_TO_ASPECT_RATIO:
            return None
        try:
            w_str, h_str = size.split("x", 1)
            smaller: Final = min(int(w_str), int(h_str))
        except (ValueError, TypeError):
            return None
        if smaller == 720:
            return "720p"
        if smaller == 1080:
            return "1080p"
        return None

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict:
        """
        Validate environment and add Gemini API key to headers.
        Gemini uses x-goog-api-key header for authentication.
        """
        # Use api_key from litellm_params if available, otherwise fall back to other sources
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key

        api_key = api_key or litellm.api_key or get_secret_str("GOOGLE_API_KEY") or get_secret_str("GEMINI_API_KEY")

        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Veo video generation. "
                "Set it via environment variable or pass it as api_key parameter."
            )

        headers.update(
            {
                "x-goog-api-key": api_key,
                "Content-Type": "application/json",
            }
        )
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: str | None,
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for Veo video generation.
        For video creation: returns full URL with :predictLongRunning
        For status/delete: returns base URL only
        """
        if api_base is None:
            api_base = get_secret_str("GEMINI_API_BASE") or "https://generativelanguage.googleapis.com"

        if not model or model == "":
            return api_base.rstrip("/")

        model_name: Final = model.replace("gemini/", "")
        url: Final = f"{api_base.rstrip('/')}/v1beta/models/{model_name}:predictLongRunning"

        return url

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[dict, RequestFiles, str]:
        """
        Transform the video creation request for Veo API.

        Veo expects:
        {
            "instances": [
                {
                    "prompt": "A cat playing with a ball of yarn",
                    "image": {
                        "bytesBase64Encoded": "...",
                        "mimeType": "image/jpeg"
                    },
                    "lastFrame": {...},        # interpolation, requires image
                    "referenceImages": [...],  # up to 3 asset references
                    "video": {...}             # extension of a Veo-generated video
                }
            ],
            "parameters": {
                "aspectRatio": "16:9",
                "durationSeconds": 8,
                "resolution": "720p"
            }
        }

        Media inputs (image, lastFrame, referenceImages, video) belong in
        instances[0]; parameters only carries generation config.
        """
        params_copy: Final = video_create_optional_request_params.copy()
        image: Final = params_copy.pop("image", None)
        last_frame: Final = params_copy.pop("lastFrame", None)
        reference_images: Final = params_copy.pop("referenceImages", None)
        video: Final = params_copy.pop("video", None)

        if reference_images is not None and not isinstance(reference_images, (list, tuple)):
            raise _bad_request("referenceImages must be a list.", model)

        try:
            instance: Final = GeminiVideoGenerationInstance.model_validate(
                MappingProxyType(
                    {
                        "prompt": prompt,
                        "image": _to_gemini_media(image),
                        "lastFrame": _to_gemini_media(last_frame),
                        "referenceImages": tuple(_to_gemini_reference_image(r) for r in reference_images)
                        if reference_images
                        else None,
                        "video": _to_gemini_media(video, is_video=True),
                    }
                )
            )
        except ValidationError as e:
            raise _bad_request(f"invalid media input: {e}", model) from e

        _validate_veo_request(model=model, instance=instance, parameters=params_copy)

        parameters: Final = GeminiVideoGenerationParameters(**params_copy)

        request_body_obj: Final = GeminiVideoGenerationRequest(instances=[instance], parameters=parameters)

        request_data: Final = request_body_obj.model_dump(mode="json", exclude_none=True)

        return request_data, [], api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: dict | None = None,
    ) -> VideoObject:
        """
        Transform the Veo video creation response.

        Veo returns:
        {
            "name": "operations/generate_1234567890",
            "metadata": {...},
            "done": false,
            "error": {...}
        }

        We return this as a VideoObject with:
        - id: operation name (used for polling)
        - status: "processing"
        - usage: includes duration_seconds and optional video_resolution for cost calculation
        """
        response_data: Final = _json_payload(raw_response)

        # Parse response using Pydantic model for type safety
        try:
            operation_response: Final = GeminiLongRunningOperationResponse.model_validate(response_data)
        except Exception as e:
            raise ValueError(f"Failed to parse operation response: {e}")

        operation_name: Final = operation_response.name
        if not operation_name:
            raise ValueError(f"No operation name in Veo response: {response_data}")

        if custom_llm_provider:
            video_id = encode_video_id_with_provider(operation_name, custom_llm_provider, model)
        else:
            video_id = operation_name

        video_obj: Final = VideoObject(
            id=video_id,
            object="video",
            status="processing",
            model=model,
        )

        usage_data: Final[dict[str, float | str]] = {}
        if request_data:
            parameters: Final = request_data.get("parameters", {})
            duration: Final = parameters.get("durationSeconds") or DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS
            if duration is not None:
                try:
                    usage_data["duration_seconds"] = float(duration)
                except (ValueError, TypeError):
                    pass
            video_resolution: Final = _usage_video_resolution_from_parameters(parameters)
            if video_resolution is not None:
                usage_data["video_resolution"] = video_resolution
            video_count: Final = veo_video_count_from_parameters(parameters)
            if video_count is not None:
                usage_data["video_count"] = video_count

        video_obj.usage = usage_data
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """
        Transform the video status retrieve request for Veo API.

        Veo polls operations at:
        GET https://generativelanguage.googleapis.com/v1beta/{operation_name}
        """
        operation_name: Final = extract_original_video_id(video_id)
        url: Final = f"{api_base.rstrip('/')}/v1beta/{operation_name}"
        params: Final[dict[str, object]] = {}

        return url, params

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        client: "HTTPHandler | None" = None,
    ) -> VideoObject:
        """
        Transform the Veo operation status response.

        Veo returns:
        {
            "name": "operations/generate_1234567890",
            "done": false  # or true when complete
        }

        When done=true:
        {
            "name": "operations/generate_1234567890",
            "done": true,
            "response": {
                "generateVideoResponse": {
                    "generatedSamples": [
                        {
                            "video": {
                                "uri": "files/abc123..."
                            }
                        }
                    ]
                }
            }
        }
        """
        response_data: Final = _json_payload(raw_response)
        # Parse response using Pydantic model for type safety
        operation_response: Final = GeminiLongRunningOperationResponse.model_validate(response_data)

        operation_name: Final = operation_response.name
        is_done: Final = operation_response.done

        if custom_llm_provider:
            video_id = encode_video_id_with_provider(operation_name, custom_llm_provider, None)
        else:
            video_id = operation_name

        video_obj: Final = VideoObject(
            id=video_id,
            object="video",
            status="processing" if not is_done else "completed",
        )
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        variant: str | None = None,
    ) -> tuple[str, dict]:
        """
        Transform the video content request for Veo API.

        For Veo, we need to:
        1. Get operation status to extract video URI
        2. Return download URL for the video
        """
        operation_name: Final = extract_original_video_id(video_id)

        status_url: Final = f"{api_base.rstrip('/')}/v1beta/{operation_name}"
        client: Final = litellm.module_level_client
        status_response: Final = client.get(url=status_url, headers=headers)
        status_response.raise_for_status()
        response_data: Final = _json_payload(status_response)

        operation_response: Final = GeminiLongRunningOperationResponse.model_validate(response_data)

        if not operation_response.done:
            raise ValueError(
                "Video generation is not complete yet. Please check status with video_status() before downloading."
            )

        if not operation_response.response:
            raise ValueError("No response data in completed operation")

        generated_samples: Final = operation_response.response.generateVideoResponse.generatedSamples
        download_url: Final = generated_samples[0].video.uri

        params: Final[dict[str, object]] = {}

        return download_url, params

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """
        Transform the Veo video content download response.
        Returns the video bytes directly.
        """
        return raw_response.content

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Mapping[str, object] | None = None,
    ) -> tuple[str, dict]:
        """
        Video remix is not supported by Veo API.
        """
        raise NotImplementedError(
            "Video remix is not supported by Google Veo. Please use video_generation() to create new videos."
        )

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        """Video remix is not supported."""
        raise NotImplementedError("Video remix is not supported by Google Veo.")

    def transform_video_list_request(
        self,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: str | None = None,
        limit: int | None = None,
        order: str | None = None,
        extra_query: Mapping[str, object] | None = None,
    ) -> tuple[str, dict]:
        """
        Video list is not supported by Veo API.
        """
        raise NotImplementedError(
            "Video list is not supported by Google Veo. "
            "Use the operations endpoint directly if you need to list operations."
        )

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> dict[str, str]:
        """Video list is not supported."""
        raise NotImplementedError("Video list is not supported by Google Veo.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> tuple[str, dict]:
        """
        Video delete is not supported by Veo API.
        """
        raise NotImplementedError(
            "Video delete is not supported by Google Veo. Videos are automatically cleaned up by Google."
        )

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """Video delete is not supported."""
        raise NotImplementedError("Video delete is not supported by Google Veo.")

    def transform_video_create_character_request(self, name, video: object, api_base, litellm_params, headers):
        raise NotImplementedError("video create character is not supported for Gemini")

    def transform_video_create_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video create character is not supported for Gemini")

    def transform_video_get_character_request(self, character_id, api_base, litellm_params, headers):
        raise NotImplementedError("video get character is not supported for Gemini")

    def transform_video_get_character_response(self, raw_response, logging_obj):
        raise NotImplementedError("video get character is not supported for Gemini")

    def transform_video_edit_request(
        self,
        prompt,
        video_id,
        api_base,
        litellm_params,
        headers,
        video_file=None,
        extra_body=None,
        prefetched_source_data=None,
    ):
        raise NotImplementedError("video edit is not supported for Gemini")

    def transform_video_edit_response(
        self,
        raw_response,
        logging_obj,
        custom_llm_provider=None,
        request_data=None,
    ):
        raise NotImplementedError("video edit is not supported for Gemini")

    def transform_video_extension_request(
        self,
        prompt,
        video_id,
        seconds,
        api_base,
        litellm_params,
        headers,
        extra_body=None,
    ):
        raise NotImplementedError("video extension is not supported for Gemini")

    def transform_video_extension_response(self, raw_response, logging_obj, custom_llm_provider=None):
        raise NotImplementedError("video extension is not supported for Gemini")

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        from ..common_utils import GeminiError

        return GeminiError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
