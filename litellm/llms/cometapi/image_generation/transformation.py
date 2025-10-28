from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import (
    AllMessageValues,
    OpenAIImageGenerationOptionalParams,
)
from litellm.types.utils import ImageObject, ImageResponse

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class CometAPIImageGenerationConfig(BaseImageGenerationConfig):
    DEFAULT_BASE_URL: str = "https://api.cometapi.com"
    IMAGE_GENERATION_ENDPOINT: str = "v1/images/generations"
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        https://api.cometapi.com/v1/images/generations
        """
        return [
            "n",
            "quality",
            "response_format",
            "size",
            "style",
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        
        for k in non_default_params.keys():
            if k not in optional_params.keys():
                if k in supported_params:
                    # CometAPI uses OpenAI-compatible parameters, so we can pass them directly
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

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
        """
        complete_url: str = (
            api_base 
            or get_secret_str("COMETAPI_BASE_URL")
            or get_secret_str("COMETAPI_API_BASE")
            or self.DEFAULT_BASE_URL
        )

        complete_url = complete_url.rstrip("/")
        complete_url = f"{complete_url}/{self.IMAGE_GENERATION_ENDPOINT}"
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
            get_secret_str("COMETAPI_KEY") or
            get_secret_str("COMETAPI_API_KEY")
        )
        if not final_api_key:
            raise ValueError("COMETAPI_KEY or COMETAPI_API_KEY is not set")
        
        headers["Authorization"] = f"Bearer {final_api_key}"
        headers["Content-Type"] = "application/json"        
        return headers

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to the CometAPI image generation request body

        https://api.cometapi.com/v1/images/generations
        """
        # CometAPI uses OpenAI-compatible format
        request_body = {
            "prompt": prompt,
            "model": model,
            **optional_params,
        }
        return request_body

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: dict,
        optional_params: dict,
        litellm_params: dict,
        encoding: Any,
        api_key: Optional[str] = None,
        json_mode: Optional[bool] = None,
    ) -> ImageResponse:
        """
        Transform the image generation response to the litellm image response

        https://api.cometapi.com/v1/images/generations
        """
        try:
            response_data = raw_response.json()
        except Exception as e:
            raise self.get_error_class(
                error_message=f"Error transforming image generation response: {e}",
                status_code=raw_response.status_code,
                headers=raw_response.headers,
            )
            
        if not model_response.data:
            model_response.data = []
        
        # CometAPI returns OpenAI-compatible format
        # Expected format: {"created": timestamp, "data": [{"url": "...", "b64_json": "..."}]}
        if "data" in response_data:
            for image_data in response_data["data"]:
                image_obj = ImageObject(
                    b64_json=image_data.get("b64_json"),
                    url=image_data.get("url"),
                )
                model_response.data.append(image_obj)
        
        return model_response
