from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, List, Optional, Union

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ModelResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class RecraftImageGenerationConfig(BaseImageGenerationConfig):
    DEFAULT_BASE_URL: str = "https://api.recraft.ai/v1"
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        https://www.recraft.ai/docs#generate-image
        """
        return [
            "n",
            "response_format",
            "size",
            "style"
        ]

    def get_complete_url(
        self,
        api_base: Optional[str],
        api_key: Optional[str],
        model: str,
        optional_params: dict,
        litellm_params: dict,
        stream: Optional[bool] = None,
    ) -> str:
        """
        Get the complete url for the request

        Some providers need `model` in `api_base`
        """
        complete_url: str = (
            api_base 
            or get_secret_str("RECRAFT_API_BASE") 
            or self.DEFAULT_BASE_URL
        )

        complete_url = complete_url.rstrip("/")
        complete_url = f"{complete_url}/v1/images/generations"
        return complete_url

    def validate_environment(
        self,
        headers: dict,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
    ) -> dict:
        final_api_key: Optional[str] = (
            api_key or 
            get_secret_str("RECRAFT_API_KEY")
        )
        if not final_api_key:
            raise ValueError("RECRAFT_API_KEY is not set")
        
        headers["Authorization"] = f"Bearer {final_api_key}"        
        return headers



    def transform_request(
        self,
        model: str,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        return {}

    def transform_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ModelResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        messages: List[AllMessageValues],
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ModelResponse:
        return model_response
