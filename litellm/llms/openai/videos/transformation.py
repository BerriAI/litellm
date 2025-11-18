from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from io import BufferedReader
from typing import cast
import httpx
from httpx._types import RequestFiles

from litellm.types.videos.main import VideoCreateOptionalRequestParams
from litellm.types.llms.openai import CreateVideoRequest
from litellm.types.router import GenericLiteLLMParams
from litellm.secret_managers.main import get_secret_str
from litellm.types.videos.main import VideoObject
import litellm
from litellm.llms.openai.image_edit.transformation import ImageEditRequestUtils
if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    from ...base_llm.videos.transformation import BaseVideoConfig as _BaseVideoConfig
    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseVideoConfig = _BaseVideoConfig
    BaseLLMException = _BaseLLMException
else:
    LiteLLMLoggingObj = Any
    BaseVideoConfig = Any
    BaseLLMException = Any


class OpenAIVideoConfig(BaseVideoConfig):
    """
    Configuration class for OpenAI video generation.
    """

    def __init__(self):
        super().__init__()

    def get_supported_openai_params(self, model: str) -> list:
        """
        Get the list of supported OpenAI parameters for video generation.
        """
        return [
            "model",
            "prompt",
            "input_reference",
            "seconds",
            "size",
            "user",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> Dict:
        """No mapping applied since inputs are in OpenAI spec already"""
        return dict(video_create_optional_params)

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        api_key = (
            api_key
            or litellm.api_key
            or litellm.openai_key
            or get_secret_str("OPENAI_API_KEY")
        )
        headers.update(
            {
                "Authorization": f"Bearer {api_key}",
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
        Get the complete URL for OpenAI video generation.
        """
        if api_base is None:
            api_base = "https://api.openai.com/v1"
        
        return f"{api_base.rstrip('/')}/videos"

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        video_create_optional_request_params: Dict,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[Dict, RequestFiles]:
        """
        Transform the video creation request for OpenAI API.
        """
        # Remove model and extra_headers from optional params as they're handled separately
        video_create_optional_request_params = {
            k: v for k, v in video_create_optional_request_params.items()
            if k not in ["model", "extra_headers"]
        }
        
        # Create the request data
        video_create_request = CreateVideoRequest(
            model=model,
            prompt=prompt,
            **video_create_optional_request_params
        )
        request_dict = cast(Dict, video_create_request)

        # Handle input_reference parameter if provided
        _input_reference = video_create_optional_request_params.get("input_reference")
        data_without_files = {
            k: v for k, v in request_dict.items() if k not in ["input_reference"]
        }
        files_list: List[Tuple[str, Any]] = []

        # Handle input_reference parameter
        if _input_reference is not None:
            self._add_image_to_files(
                files_list=files_list,
                image=_input_reference,
                field_name="input_reference",
            )
        # Convert to dict for JSON serialization
        return data_without_files, files_list

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the OpenAI video creation response.
        """
        response_data = raw_response.json()
        
        # Transform the response data
    
        video_obj = VideoObject(**response_data)  # type: ignore[arg-type]
        
        # Create usage object with duration information for cost calculation
        # Video generation API doesn't provide usage, so we create one with duration
        usage_data = {}
        if video_obj:
            if hasattr(video_obj, 'seconds') and video_obj.seconds:
                try:
                    usage_data["duration_seconds"] = float(video_obj.seconds)
                except (ValueError, TypeError):
                    pass
        # Create the response
        video_obj.usage = usage_data

        
        return video_obj

    def transform_video_content_request(
        self,
        video_id: str,
        model: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video content request for OpenAI API.
        
        OpenAI API expects the following request:
        - GET /v1/videos/{video_id}/content
        """
        # Construct the URL for video content download
        url = f"{api_base.rstrip('/')}/{video_id}/content"
        
        # Add video_id as query parameter
        params = {"video_id": video_id}
        
        return url, params

    def transform_video_remix_request(
        self,
        video_id: str,
        prompt: str,
        model: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """
        Transform the video remix request for OpenAI API.
        
        OpenAI API expects the following request:
        - POST /v1/videos/{video_id}/remix
        """
        # Construct the URL for video remix
        url = f"{api_base.rstrip('/')}/{video_id}/remix"
        
        # Prepare the request data
        data = {"prompt": prompt}
        
        # Add any extra body parameters
        if extra_body:
            data.update(extra_body)
        
        return url, data
    
    def transform_video_content_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> bytes:
        """
        Transform the OpenAI video content download response.
        Returns raw video content as bytes.
        """
        # For video content download, return the raw content as bytes
        return raw_response.content

    def transform_video_remix_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the OpenAI video remix response.
        """
        response_data = raw_response.json()
        
        # Transform the response data
        video_obj = VideoObject(**response_data)  # type: ignore[arg-type]
        
        # Create usage object with duration information for cost calculation
        # Video remix API doesn't provide usage, so we create one with duration
        usage_data = {}
        if video_obj:
            if hasattr(video_obj, 'seconds') and video_obj.seconds:
                try:
                    usage_data["duration_seconds"] = float(video_obj.seconds)
                except (ValueError, TypeError):
                    pass
        # Create the response
        video_obj.usage = usage_data

        return video_obj

    def transform_video_list_request(
        self,
        model: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
        after: Optional[str] = None,
        limit: Optional[int] = None,
        order: Optional[str] = None,
        extra_query: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, Dict]:
        """
        Transform the video list request for OpenAI API.
        
        OpenAI API expects the following request:
        - GET /v1/videos
        """
        # Use the api_base directly for video list
        url = api_base
        
        # Prepare query parameters
        params = {}
        if after is not None:
            params["after"] = after
        if limit is not None:
            params["limit"] = str(limit)
        if order is not None:
            params["order"] = order
        
        # Add any extra query parameters
        if extra_query:
            params.update(extra_query)
        
        return url, params

    def transform_video_list_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> Dict[str,str]:
        return raw_response.json()

    def transform_video_delete_request(
        self,
        video_id: str,
        model: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the video delete request for OpenAI API.
        
        OpenAI API expects the following request:
        - DELETE /v1/videos/{video_id}
        """
        # Construct the URL for video delete
        url = f"{api_base.rstrip('/')}/{video_id}"
        
        # No data needed for DELETE request
        data: Dict[str, Any] = {}
        
        return url, data

    def transform_video_delete_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the OpenAI video delete response.
        """
        response_data = raw_response.json()
        
        # Transform the response data
        video_obj = VideoObject(**response_data)  # type: ignore[arg-type]  # type: ignore[arg-type]

        return video_obj

    def transform_video_status_retrieve_request(
        self,
        video_id: str,
        model: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: dict,
    ) -> Tuple[str, Dict]:
        """
        Transform the OpenAI video retrieve request.
        """
        # For video retrieve, we just need to construct the URL
        url = f"{api_base.rstrip('/')}/{video_id}"
        
        # No additional data needed for GET request
        data: Dict[str, Any] = {}
        
        return url, data

    def transform_video_status_retrieve_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoObject:
        """
        Transform the OpenAI video retrieve response.
        """
        response_data = raw_response.json()
        # Transform the response data
        video_obj = VideoObject(**response_data)  # type: ignore[arg-type]

        return video_obj

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )

    def _add_image_to_files(
        self,
        files_list: List[Tuple[str, Any]],
        image: Any,
        field_name: str,
    ) -> None:
        """Add an image to the files list with appropriate content type"""
        image_content_type = ImageEditRequestUtils.get_image_content_type(image)

        if isinstance(image, BufferedReader):
            files_list.append((field_name, (image.name, image, image_content_type)))
        else:
            files_list.append((field_name, ("input_reference.png", image, image_content_type)))