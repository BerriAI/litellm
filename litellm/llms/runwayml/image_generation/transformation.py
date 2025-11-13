from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.constants import RUNWAYML_DEFAULT_API_VERSION
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


class RunwayMLImageGenerationConfig(BaseImageGenerationConfig):
    """
    Configuration for RunwayML image generation models.
    """
    DEFAULT_BASE_URL: str = "https://api.dev.runwayml.com"
    IMAGE_GENERATION_ENDPOINT: str = "v1/text_to_image"

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
            or get_secret_str("RUNWAYML_API_BASE") 
            or self.DEFAULT_BASE_URL
        )

        complete_url = complete_url.rstrip("/")
        if self.IMAGE_GENERATION_ENDPOINT:
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
            get_secret_str("RUNWAYML_API_SECRET") or
            get_secret_str("RUNWAYML_API_KEY")
        )
        if not final_api_key:
            raise ValueError("RUNWAYML_API_SECRET or RUNWAYML_API_KEY is not set")
        
        headers["Authorization"] = f"Bearer {final_api_key}"    
        headers["X-Runway-Version"] = RUNWAYML_DEFAULT_API_VERSION
        return headers

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
        
        # Handle RunwayML response format
        # Response contains task.output[0] with image URL(s)
        output = response_data.get("output", [])
        if isinstance(output, list):
            for image_url in output:
                if isinstance(image_url, str):
                    model_response.data.append(ImageObject(
                        url=image_url,
                        b64_json=None,
                    ))
                elif isinstance(image_url, dict):
                    # Handle if output contains dict with url/b64_json
                    model_response.data.append(ImageObject(
                        url=image_url.get("url", None),
                        b64_json=image_url.get("b64_json", None),
                    ))
        
        return model_response
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Get supported OpenAI parameters for RunwayML image generation
        """
        return [
            "size",
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        
        # Map OpenAI 'size' parameter to RunwayML 'ratio' parameter
        if "size" in non_default_params:
            size = non_default_params["size"]
            # Map common OpenAI sizes to RunwayML ratios
            size_to_ratio_map = {
                "1024x1024": "1024:1024",
                "1792x1024": "1792:1024",
                "1024x1792": "1024:1792",
                "1920x1080": "1920:1080",
                "1080x1920": "1080:1920",
            }
            optional_params["ratio"] = size_to_ratio_map.get(size, "1920:1080")
        
        for k in non_default_params.keys():
            if k not in optional_params.keys():
                if k in supported_params:
                    optional_params[k] = non_default_params[k]
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to the RunwayML image generation request body
        
        RunwayML expects:
        - model: The model to use (e.g., 'gen4_image')
        - promptText: The text prompt
        - ratio: The aspect ratio (e.g., '1920:1080', '1080:1920', '1024:1024')
        """
        runwayml_request_body = {
            "model": model or "gen4_image",
            "promptText": prompt,
        }
        
        # Add any RunwayML-specific parameters
        if "ratio" in optional_params:
            runwayml_request_body["ratio"] = optional_params["ratio"]
        else:
            # Set default ratio if not provided
            runwayml_request_body["ratio"] = "1920:1080"

        
        # Add any other optional parameters
        for k, v in optional_params.items():
            if k not in runwayml_request_body and k not in ["size"]:
                runwayml_request_body[k] = v
        
        return runwayml_request_body

