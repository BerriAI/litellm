from typing import TYPE_CHECKING, Any, Optional, Union

import httpx

from litellm.secret_managers.main import get_secret_str
import litellm

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
        Get the complete URL for OpenAI video operations.
        For video content download, this returns the base videos URL.
        """
        if api_base is None:
            api_base = "https://api.openai.com/v1"
        
        return f"{api_base.rstrip('/')}/videos"

    def transform_video_retrieve_response(
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

    def get_error_class(
        self, error_message: str, status_code: int, headers: Union[dict, httpx.Headers]
    ) -> BaseLLMException:
        from ...base_llm.chat.transformation import BaseLLMException

        raise BaseLLMException(
            status_code=status_code,
            message=error_message,
            headers=headers,
        )
