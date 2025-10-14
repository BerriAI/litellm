import types
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union
from io import BufferedReader

import httpx
from httpx._types import RequestFiles

from litellm.types.videos.main import VideoCreateOptionalRequestParams
from litellm.types.llms.openai import CreateVideoRequest, OpenAIVideoObject
from litellm.types.videos.main import VideoResponse
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.types.videos.main import VideoResponse as _VideoResponse

    from ...base_llm.videos_generation.transformation import BaseVideoGenerationConfig as _BaseVideoGenerationConfig
    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseVideoGenerationConfig = _BaseVideoGenerationConfig
    BaseLLMException = _BaseLLMException
    VideoResponse = _VideoResponse
else:
    LiteLLMLoggingObj = Any
    BaseVideoGenerationConfig = Any
    BaseLLMException = Any
    VideoResponse = Any


class OpenAIVideoGenerationConfig(BaseVideoGenerationConfig):
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
        """
        Validate the environment for OpenAI video generation.
        """
        if api_key is None:
            raise ValueError("OpenAI API key is required for video generation")
        
        headers["Authorization"] = f"Bearer {api_key}"
        headers["Content-Type"] = "application/json"
        
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
        files_list: RequestFiles = []
        
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
        data = video_create_request.model_dump(exclude_none=True)
        
        return data, files_list

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoResponse:
        """
        Transform the OpenAI video creation response.
        """
        response_data = raw_response.json()
        
        # Transform the response data
        video_objects = []
        for video_data in response_data.get("data", []):
            video_obj = OpenAIVideoObject(**video_data)
            video_objects.append(video_obj)
        
        # Create the response
        response = VideoResponse(
            data=video_objects,
            usage=response_data.get("usage", {}),
            hidden_params={},
        )
        
        return response

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
