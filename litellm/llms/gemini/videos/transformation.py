from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union
import base64

import httpx
from httpx._types import RequestFiles

from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.router import GenericLiteLLMParams
from litellm.secret_managers.main import get_secret_str
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)
from litellm.images.utils import ImageEditRequestUtils
import litellm
from litellm.types.llms.gemini import GeminiLongRunningOperationResponse, GeminiVideoGenerationInstance, GeminiVideoGenerationParameters, GeminiVideoGenerationRequest
from litellm.constants import DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any


def _convert_image_to_gemini_format(image_file) -> Dict[str, str]:
    """
    Convert image file to Gemini format with base64 encoding and MIME type.
    
    Args:
        image_file: File-like object opened in binary mode (e.g., open("path", "rb"))
    
    Returns:
        Dict with bytesBase64Encoded and mimeType
    """
    mime_type = ImageEditRequestUtils.get_image_content_type(image_file)
    
    if hasattr(image_file, 'seek'):
        image_file.seek(0)
    image_bytes = image_file.read()
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")
    
    return {
        "bytesBase64Encoded": base64_encoded,
        "mimeType": mime_type
    }


class GeminiVideoConfig(BaseVideoConfig):
    """
    Configuration class for Gemini (Veo) video generation.
    
    Veo uses a long-running operation model:
    1. POST to :predictLongRunning returns operation name
    2. Poll operation until done=true
    3. Extract video URI from response
    4. Download video using file API
    """

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported OpenAI parameters for Veo video generation.
        Veo supports minimal parameters compared to OpenAI.
        """
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size"
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict[str, Any]:
        """
        Map OpenAI-style parameters to Veo format.
        
        Mappings:
        - prompt → prompt
        - input_reference → image
        - size → aspectRatio (e.g., "1280x720" → "16:9")
        - seconds → durationSeconds (defaults to 4 seconds if not provided)
        
        All other params are passed through as-is to support Gemini-specific parameters.
        """
        mapped_params: Dict[str, Any] = {}
        
        # Get supported OpenAI params (exclude "model" and "prompt" which are handled separately)
        supported_openai_params = self.get_supported_openai_params(model)
        openai_params_to_map = {
            param for param in supported_openai_params 
            if param not in {"model", "prompt"}
        }
        
        # Map input_reference to image
        if "input_reference" in video_create_optional_params:
            mapped_params["image"] = video_create_optional_params["input_reference"]
        
        # Map size to aspectRatio
        if "size" in video_create_optional_params:
            size = video_create_optional_params["size"]
            if size is not None:
                aspect_ratio = self._convert_size_to_aspect_ratio(size)
                if aspect_ratio:
                    mapped_params["aspectRatio"] = aspect_ratio
        
        # Map seconds to durationSeconds, default to 4 seconds (matching OpenAI)
        if "seconds" in video_create_optional_params:
            seconds = video_create_optional_params["seconds"]
            try:
                duration = int(seconds) if isinstance(seconds, str) else seconds
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
    
    def _convert_size_to_aspect_ratio(self, size: str) -> Optional[str]:
        """
        Convert OpenAI size format to Veo aspectRatio format.
        
        https://cloud.google.com/vertex-ai/generative-ai/docs/image/generate-videos
        
        Supported aspect ratios: 9:16 (portrait), 16:9 (landscape)
        """
        if not size:
            return None
        
        aspect_ratio_map = {
            "1280x720": "16:9",
            "1920x1080": "16:9",
            "720x1280": "9:16",
            "1080x1920": "9:16",
        }
        
        return aspect_ratio_map.get(size, "16:9")


    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
        litellm_params: Optional[GenericLiteLLMParams] = None,
    ) -> dict:
        """
        Validate environment and add Gemini API key to headers.
        Gemini uses x-goog-api-key header for authentication.
        """
        # Use api_key from litellm_params if available, otherwise fall back to other sources
        if litellm_params and litellm_params.api_key:
            api_key = api_key or litellm_params.api_key
        
        api_key = (
            api_key
            or litellm.api_key
            or get_secret_str("GOOGLE_API_KEY")
            or get_secret_str("GEMINI_API_KEY")
        )
        
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY or GOOGLE_API_KEY is required for Veo video generation. "
                "Set it via environment variable or pass it as api_key parameter."
            )
        
        headers.update({
            "x-goog-api-key": api_key,
            "Content-Type": "application/json",
        })
        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
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
            return api_base.rstrip('/')
        
        model_name = model.replace("gemini/", "")
        url = f"{api_base.rstrip('/')}/v1beta/models/{model_name}:predictLongRunning"
        
        return url

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles, str]:
        """
        Transform the video creation request for Veo API.
        
        Veo expects:
        {
            "instances": [
                {
                    "prompt": "A cat playing with a ball of yarn"
                }
            ],
            "parameters": {
                "aspectRatio": "16:9",
                "durationSeconds": 8,
                "resolution": "720p"
            }
        }
        """
        instance = GeminiVideoGenerationInstance(prompt=prompt)
        
        params_copy = video_create_optional_request_params.copy()
        
        if "image" in params_copy and params_copy["image"] is not None:
            image_data = _convert_image_to_gemini_format(params_copy["image"])
            params_copy["image"] = image_data
        
        parameters = GeminiVideoGenerationParameters(**params_copy)
        
        request_body_obj = GeminiVideoGenerationRequest(
            instances=[instance],
            parameters=parameters
        )
        
        request_data = request_body_obj.model_dump(exclude_none=True)
        
        return request_data, [], api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
        request_data: Optional[Dict] = None,
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
        - usage: includes duration_seconds for cost calculation
        """        
        response_data = raw_response.json()
        
        # Parse response using Pydantic model for type safety
        try:
            operation_response = GeminiLongRunningOperationResponse(**response_data)
        except Exception as e:
            raise ValueError(f"Failed to parse operation response: {e}")
        
        operation_name = operation_response.name
        if not operation_name:
            raise ValueError(f"No operation name in Veo response: {response_data}")
        
        if custom_llm_provider:
            video_id = encode_video_id_with_provider(operation_name, custom_llm_provider, model)
        else:
            video_id = operation_name
        
        video_obj = VideoObject(
            id=video_id,
            object="video",
            status="processing",
            model=model,
        )
        
        usage_data = {}
        if request_data:
            parameters = request_data.get("parameters", {})
            duration = parameters.get("durationSeconds") or DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS
            if duration is not None:
                try:
                    usage_data["duration_seconds"] = float(duration)
                except (ValueError, TypeError):
                    pass
        
        video_obj.usage = usage_data
        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video status retrieve request for Veo API.
        
        Veo polls operations at:
        GET https://generativelanguage.googleapis.com/v1beta/{operation_name}
        """
        operation_name = extract_original_video_id(video_id)
        url = f"{api_base.rstrip('/')}/v1beta/{operation_name}"
        params: Dict[str, Any] = {}
        
        return url, params

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
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
        response_data = raw_response.json()
        # Parse response using Pydantic model for type safety
        operation_response = GeminiLongRunningOperationResponse(**response_data)
        
        operation_name = operation_response.name
        is_done = operation_response.done
        
        if custom_llm_provider:
            video_id = encode_video_id_with_provider(operation_name, custom_llm_provider, None)
        else:
            video_id = operation_name
                   
        video_obj = VideoObject(
            id=video_id,
            object="video",
            status="processing" if not is_done else "completed"
        )
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video content request for Veo API.
        
        For Veo, we need to:
        1. Get operation status to extract video URI
        2. Return download URL for the video
        """        
        operation_name = extract_original_video_id(video_id)
        
        status_url = f"{api_base.rstrip('/')}/v1beta/{operation_name}"
        client = litellm.module_level_client
        status_response = client.get(url=status_url, headers=headers)
        status_response.raise_for_status()
        response_data = status_response.json()
        
        operation_response = GeminiLongRunningOperationResponse(**response_data)

        if not operation_response.done:
            raise ValueError(
                "Video generation is not complete yet. "
                "Please check status with video_status() before downloading."
            )
        
        if not operation_response.response:
            raise ValueError("No response data in completed operation")
        
        generated_samples = operation_response.response.generateVideoResponse.generatedSamples
        download_url = generated_samples[0].video.uri
        
        params: Dict[str, Any] = {}
        
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
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """
        Video remix is not supported by Veo API.
        """
        raise NotImplementedError(
            "Video remix is not supported by Google Veo. "
            "Please use video_generation() to create new videos."
        )

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        """Video remix is not supported."""
        raise NotImplementedError("Video remix is not supported by Google Veo.")

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
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        """Video list is not supported."""
        raise NotImplementedError("Video list is not supported by Google Veo.")

    def transform_video_delete_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Video delete is not supported by Veo API.
        """
        raise NotImplementedError(
            "Video delete is not supported by Google Veo. "
            "Videos are automatically cleaned up by Google."
        )

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """Video delete is not supported."""
        raise NotImplementedError("Video delete is not supported by Google Veo.")

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ..common_utils import GeminiError

        return GeminiError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

