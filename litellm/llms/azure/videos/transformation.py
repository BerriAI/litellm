from typing import TYPE_CHECKING, Any, Dict, Optional

from litellm.types.videos.main import VideoCreateOptionalRequestParams
from litellm.secret_managers.main import get_secret_str
from litellm.llms.azure.common_utils import BaseAzureLLM
import litellm
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
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


class AzureVideoConfig(OpenAIVideoConfig):
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
            or litellm.azure_key
            or get_secret_str("AZURE_OPENAI_API_KEY")
            or get_secret_str("AZURE_API_KEY")
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
        Constructs a complete URL for the API request.
        """
        return BaseAzureLLM._get_base_azure_url(
            api_base=api_base,
            litellm_params=litellm_params,
            route="/openai/v1/videos",
            default_api_version="",
        )