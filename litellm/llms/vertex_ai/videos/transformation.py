"""
Vertex AI Video Generation Transformation

Handles transformation of requests/responses for Vertex AI's Veo video generation API.
Based on: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/model-reference/veo-video-generation
"""

import base64
import time
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union, cast

import httpx
from httpx._types import RequestFiles

from litellm.constants import DEFAULT_GOOGLE_VIDEO_DURATION_SECONDS
from litellm.images.utils import ImageEditRequestUtils
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.llms.vertex_ai.common_utils import (
    _convert_vertex_datetime_to_openai_datetime,
    get_vertex_base_url,
)
from litellm.llms.vertex_ai.vertex_llm_base import VertexBase
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import (
    encode_video_id_with_provider,
    extract_original_video_id,
)

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.llms.base_llm.chat.transformation import (
        BaseLLMException as _BaseLLMException,
    )

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseLLMException = Any


def _convert_image_to_vertex_format(image_file) -> Dict[str, str]:
    """
    Convert image file to Vertex AI format with base64 encoding and MIME type.

    Args:
        image_file: File-like object opened in binary mode (e.g., open("path", "rb"))

    Returns:
        Dict with bytesBase64Encoded and mimeType
    """
    mime_type = ImageEditRequestUtils.get_image_content_type(image_file)

    if hasattr(image_file, "seek"):
        image_file.seek(0)
    image_bytes = image_file.read()
    base64_encoded = base64.b64encode(image_bytes).decode("utf-8")

    return {"bytesBase64Encoded": base64_encoded, "mimeType": mime_type}


class VertexAIVideoConfig(BaseVideoConfig, VertexBase):
    """
    Configuration class for Vertex AI (Veo) video generation.

    Veo uses a long-running operation model:
    1. POST to :predictLongRunning returns operation name
    2. Poll operation using :fetchPredictOperation until done=true
    3. Extract video data (base64) from response
    """

    def __init__(self):
        BaseVideoConfig.__init__(self)
        VertexBase.__init__(self)

    @staticmethod
    def extract_model_from_operation_name(operation_name: str) -> Optional[str]:
        """
        Extract the model name from a Vertex AI operation name.
        
        Args:
            operation_name: Operation name in format:
                projects/PROJECT/locations/LOCATION/publishers/google/models/MODEL/operations/OPERATION_ID
        
        Returns:
            Model name (e.g., "veo-2.0-generate-001") or None if extraction fails
        """
        parts = operation_name.split("/")
        # Model is at index 7 in the operation name format
        if len(parts) >= 8:
            return parts[7]
        return None

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
    ) -> Dict[str, Any]:
        """
        Map OpenAI-style parameters to Veo format.

        Mappings:
        - prompt → prompt (in instances)
        - input_reference → image (in instances)
        - size → aspectRatio (e.g., "1280x720" → "16:9")
        - seconds → durationSeconds (defaults to 4 seconds if not provided)
        """
        mapped_params: Dict[str, Any] = {}

        # Map input_reference to image (will be processed in transform_video_create_request)
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

        return mapped_params

    def _convert_size_to_aspect_ratio(self, size: str) -> Optional[str]:
        """
        Convert OpenAI size format to Veo aspectRatio format.

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
        litellm_params: Optional[Union[GenericLiteLLMParams, dict]] = None,
    ) -> dict:
        """
        Validate environment and return headers for Vertex AI OCR.
        
        Vertex AI uses Bearer token authentication with access token from credentials.
        """
        # Extract Vertex AI parameters using safe helpers from VertexBase
        # Use safe_get_* methods that don't mutate litellm_params dict
        # Ensure litellm_params is a dict for type checking
        params_dict: Dict[str, Any] = cast(Dict[str, Any], litellm_params) if litellm_params is not None else {}
        
        vertex_project = VertexBase.safe_get_vertex_ai_project(litellm_params=params_dict)
        vertex_credentials = VertexBase.safe_get_vertex_ai_credentials(litellm_params=params_dict)
        
        # Get access token from Vertex credentials
        access_token, project_id = self.get_access_token(
            credentials=vertex_credentials,
            project_id=vertex_project,
        )

        headers = {
            "Authorization": f"Bearer {access_token}",
            "Content-Type": "application/json",
            **headers,
        }

        return headers

    def get_complete_url(
        self,
        model: str,
        api_base: Optional[str],
        litellm_params: dict,
    ) -> str:
        """
        Get the complete URL for Veo video generation.

        Returns URL for :predictLongRunning endpoint:
        https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/MODEL:predictLongRunning
        """
        vertex_project = VertexBase.safe_get_vertex_ai_project(litellm_params)
        vertex_location = VertexBase.safe_get_vertex_ai_location(litellm_params)

        if not vertex_project:
            raise ValueError(
                "vertex_project is required for Vertex AI video generation. "
                "Set it via environment variable VERTEXAI_PROJECT or pass as parameter."
            )

        # Default to us-central1 if no location specified
        vertex_location = vertex_location or "us-central1"

        # Extract model name (remove vertex_ai/ prefix if present)
        model_name = model.replace("vertex_ai/", "")

        # Construct the URL
        if api_base:
            base_url = api_base.rstrip("/")
        else:
            base_url = get_vertex_base_url(vertex_location)

        url = f"{base_url}/v1/projects/{vertex_project}/locations/{vertex_location}/publishers/google/models/{model_name}"

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
                    "prompt": "A cat playing with a ball of yarn",
                    "image": {
                        "bytesBase64Encoded": "...",
                        "mimeType": "image/jpeg"
                    }
                }
            ],
            "parameters": {
                "aspectRatio": "16:9",
                "durationSeconds": 8
            }
        }
        """
        # Build instance with prompt
        instance_dict: Dict[str, Any] = {"prompt": prompt}
        params_copy = video_create_optional_request_params.copy()


        # Check if user wants to provide full instance dict
        if "instances" in params_copy and isinstance(params_copy["instances"], dict):
            # Replace/merge with user-provided instance
            instance_dict.update(params_copy["instances"])
            params_copy.pop("instances")
        elif "image" in params_copy and params_copy["image"] is not None:
            image_data = _convert_image_to_vertex_format(params_copy["image"])
            instance_dict["image"] = image_data
            params_copy.pop("image")

        # Build request data directly (TypedDict doesn't have model_dump)
        request_data: Dict[str, Any] = {"instances": [instance_dict]}

        # Only add parameters if there are any
        if params_copy:
            request_data["parameters"] = params_copy

        # Append :predictLongRunning endpoint to api_base
        url = f"{api_base}:predictLongRunning"

        # No files needed - everything is in JSON
        return request_data, [], url

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
            "name": "projects/PROJECT_ID/locations/LOCATION/publishers/google/models/MODEL/operations/OPERATION_ID"
        }

        We return this as a VideoObject with:
        - id: operation name (used for polling)
        - status: "processing"
        - usage: includes duration_seconds for cost calculation
        """
        response_data = raw_response.json()

        operation_name = response_data.get("name")
        if not operation_name:
            raise ValueError(f"No operation name in Veo response: {response_data}")

        if custom_llm_provider:
            video_id = encode_video_id_with_provider(
                operation_name, custom_llm_provider, model
            )
        else:
            video_id = operation_name


        video_obj = VideoObject(
            id=video_id,
            object="video",
            status="processing",
            model=model
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

        Veo polls operations using :fetchPredictOperation endpoint with POST request.
        """
        operation_name = extract_original_video_id(video_id)
        model = self.extract_model_from_operation_name(operation_name)
        
        if not model:
            raise ValueError(
                f"Invalid operation name format: {operation_name}. "
                "Expected format: projects/PROJECT/locations/LOCATION/publishers/google/models/MODEL/operations/OPERATION_ID"
            )

        # Construct the full URL including model ID
        # URL format: https://LOCATION-aiplatform.googleapis.com/v1/projects/PROJECT/locations/LOCATION/publishers/google/models/MODEL:fetchPredictOperation
        # Strip trailing slashes from api_base and append model
        url = f"{api_base.rstrip('/')}/{model}:fetchPredictOperation"

        # Request body contains the operation name
        params = {"operationName": operation_name}

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
            "name": "projects/.../operations/OPERATION_ID",
            "done": false  # or true when complete
        }

        When done=true:
        {
            "name": "projects/.../operations/OPERATION_ID",
            "done": true,
            "response": {
                "@type": "type.googleapis.com/cloud.ai.large_models.vision.GenerateVideoResponse",
                "raiMediaFilteredCount": 0,
                "videos": [
                    {
                        "bytesBase64Encoded": "...",
                        "mimeType": "video/mp4"
                    }
                ]
            }
        }
        """
        response_data = raw_response.json()

        operation_name = response_data.get("name", "")
        is_done = response_data.get("done", False)
        error_data = response_data.get("error")

        # Extract model from operation name
        model = self.extract_model_from_operation_name(operation_name)

        if custom_llm_provider:
            video_id = encode_video_id_with_provider(
                operation_name, custom_llm_provider, model
            )
        else:
            video_id = operation_name

        # Convert createTime to Unix timestamp
        create_time_str = response_data.get("metadata", {}).get("createTime")
        if create_time_str:
            try:
                created_at = _convert_vertex_datetime_to_openai_datetime(
                    create_time_str
                )
            except Exception:
                created_at = int(time.time())
        else:
            created_at = int(time.time())

        if error_data:
            status = "failed"
        elif is_done:
            status = "completed"
        else:
            status = "processing"

        video_obj = VideoObject(
            id=video_id,
            object="video",
            status=status,
            model=model,
            created_at=created_at,
            error=error_data,
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
        1. Poll the operation status to ensure it's complete
        2. Extract the base64 video data from the response
        3. Return it for decoding

        Since we need to make an HTTP call here, we'll use the same fetchPredictOperation
        approach as status retrieval.
        """
        return self.transform_video_status_retrieve_request(video_id, api_base, litellm_params, headers)

    def transform_video_content_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """
        Transform the Veo video content download response.

        Extracts the base64 encoded video from the response and decodes it to bytes.
        """
        response_data = raw_response.json()

        if not response_data.get("done", False):
            raise ValueError(
                "Video generation is not complete yet. "
                "Please check status with video_status() before downloading."
            )

        try:
            video_response = response_data.get("response", {})
            videos = video_response.get("videos", [])

            if not videos or len(videos) == 0:
                raise ValueError("No video data found in completed operation")

            # Get the first video
            video_data = videos[0]
            base64_encoded = video_data.get("bytesBase64Encoded")

            if not base64_encoded:
                raise ValueError("No base64 encoded video data found")

            # Decode base64 to bytes
            video_bytes = base64.b64decode(base64_encoded)
            return video_bytes

        except (KeyError, IndexError) as e:
            raise ValueError(f"Failed to extract video data: {e}")

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
            "Video remix is not supported by Vertex AI Veo. "
            "Please use video_generation() to create new videos."
        )

    def transform_video_remix_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> VideoObject:
        """Video remix is not supported."""
        raise NotImplementedError("Video remix is not supported by Vertex AI Veo.")

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
            "Video list is not supported by Vertex AI Veo. "
            "Use the operations endpoint directly if you need to list operations."
        )

    def transform_video_list_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: Optional[str] = None,
    ) -> Dict[str, str]:
        """Video list is not supported."""
        raise NotImplementedError("Video list is not supported by Vertex AI Veo.")

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
            "Video delete is not supported by Vertex AI Veo. "
            "Videos are automatically cleaned up by Google."
        )

    def transform_video_delete_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """Video delete is not supported."""
        raise NotImplementedError("Video delete is not supported by Vertex AI Veo.")

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from litellm.llms.vertex_ai.common_utils import VertexAIError

        return VertexAIError(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

