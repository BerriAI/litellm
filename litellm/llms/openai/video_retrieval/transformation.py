import types
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import httpx

from litellm.types.llms.openai import OpenAIVideoObject
from litellm.types.videos.main import VideoResponse
from litellm.types.router import GenericLiteLLMParams

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj
    from litellm.types.videos.main import VideoResponse as _VideoResponse

    from ...base_llm.video_retrieval.transformation import BaseVideoRetrievalConfig as _BaseVideoRetrievalConfig
    from ...base_llm.chat.transformation import BaseLLMException as _BaseLLMException

    LiteLLMLoggingObj = _LiteLLMLoggingObj
    BaseVideoRetrievalConfig = _BaseVideoRetrievalConfig
    BaseLLMException = _BaseLLMException
    VideoResponse = _VideoResponse
else:
    LiteLLMLoggingObj = Any
    BaseVideoRetrievalConfig = Any
    BaseLLMException = Any
    VideoResponse = Any


class OpenAIVideoRetrievalConfig(BaseVideoRetrievalConfig):
    """
    Configuration class for OpenAI video retrieval operations.
    """

    def __init__(self):
        super().__init__()

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: Optional[str] = None,
    ) -> dict:
        """
        Validate the environment for OpenAI video retrieval.
        """
        if api_key is None:
            raise ValueError("OpenAI API key is required for video retrieval")
        
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
        Get the complete URL for OpenAI video retrieval.
        """
        if api_base is None:
            api_base = "https://api.openai.com/v1"
        
        return f"{api_base.rstrip('/')}/videos"

    def transform_video_retrieve_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> VideoResponse:
        """
        Transform the OpenAI video retrieval response.
        """
        response_data = raw_response.json()
        
        # Transform the response data
        video_obj = OpenAIVideoObject(**response_data)
        
        # Create the response
        response = VideoResponse(
            data=[video_obj],
            usage={},
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
