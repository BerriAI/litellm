from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union, IO
from io import BufferedReader

import httpx
from httpx._types import RequestFiles

from litellm.types.videos.main import VideoCreateOptionalRequestParams
from litellm.types.llms.openai import CreateVideoRequest
from litellm.types.videos.main import VideoResponse
from litellm.types.router import GenericLiteLLMParams
from litellm.secret_managers.main import get_secret_str
from litellm.types.videos.main import VideoObject
import litellm

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
        
        # Handle file uploads
        files_list: List[Tuple[str, Tuple[str, Union[IO[bytes], bytes, str], str]]] = []
        
        # Handle input_reference parameter if provided
        _input_reference = video_create_optional_request_params.get("input_reference")
        if _input_reference is not None:
            if isinstance(_input_reference, BufferedReader):
                files_list.append(
                    ("input_reference", (_input_reference.name, _input_reference, "image/png"))
                )
            elif isinstance(_input_reference, str):
                # Handle file path - open the file
                try:
                    with open(_input_reference, "rb") as f:
                        files_list.append(
                            ("input_reference", (f.name, f.read(), "image/png"))
                        )
                except Exception as e:
                    raise ValueError(f"Could not open input_reference file {_input_reference}: {e}")
            else:
                # Handle file-like object
                files_list.append(
                    ("input_reference", ("input_reference.png", _input_reference, "image/png"))
                )
        
        # Convert to dict for JSON serialization
        data = dict(video_create_request)
        
        return data, files_list

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
    
        video_obj = VideoObject(**response_data)
        
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
        video_obj = VideoObject(**response_data)
        
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

    def transform_video_list_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoResponse:
        """
        Transform the OpenAI video list response.
        """
        response_data = raw_response.json()
        response_data = VideoResponse(**response_data)
        # The response should already be in the correct format
        # Just return it as-is since it matches the expected structure
        return response_data

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
        video_obj = VideoObject(**response_data)

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
