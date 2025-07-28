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


class GoogleImageGenConfig(BaseImageGenerationConfig):
    DEFAULT_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Google AI Imagen API supported parameters
        https://ai.google.dev/gemini-api/docs/imagen
        """
        return [
            "n",
            "size",
            "response_format"
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        supported_params = self.get_supported_openai_params(model)
        mapped_params = {}
        
        for k, v in non_default_params.items():
            if k not in optional_params.keys():
                if k in supported_params:
                    # Map OpenAI parameters to Google format
                    if k == "n":
                        mapped_params["sampleCount"] = v
                    elif k == "size":
                        # Map OpenAI size format to Google aspectRatio
                        aspect_ratio_map = {
                            "1024x1024": "1:1",
                            "1792x1024": "16:9", 
                            "1024x1792": "9:16",
                            "1280x896": "4:3",
                            "896x1280": "3:4"
                        }
                        mapped_params["aspectRatio"] = aspect_ratio_map.get(v, "1:1")
                    else:
                        mapped_params[k] = v
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        # Set default values
        if "sampleCount" not in mapped_params:
            mapped_params["sampleCount"] = 4
        if "aspectRatio" not in mapped_params:
            mapped_params["aspectRatio"] = "1:1"
            
        # Add default person generation policy
        mapped_params["personGeneration"] = "allow_adult"
        
        return mapped_params

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
        
        Google AI API format: https://generativelanguage.googleapis.com/v1beta/models/{model}:predict
        """
        complete_url: str = (
            api_base 
            or get_secret_str("GEMINI_API_BASE") 
            or self.DEFAULT_BASE_URL
        )

        complete_url = complete_url.rstrip("/")
        complete_url = f"{complete_url}/models/{model}:predict"
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
            get_secret_str("GEMINI_API_KEY")
        )
        if not final_api_key:
            raise ValueError("GEMINI_API_KEY is not set")
        
        headers["x-goog-api-key"] = final_api_key
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
        Transform the image generation request to Google AI Imagen format
        
        Google AI API format:
        {
          "instances": [
            {
              "prompt": "Robot holding a red skateboard"
            }
          ],
          "parameters": {
            "sampleCount": 4,
            "aspectRatio": "1:1",
            "personGeneration": "allow_adult"
          }
        }
        """
        request_body = {
            "instances": [
                {
                    "prompt": prompt
                }
            ],
            "parameters": optional_params
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
        Transform Google AI Imagen response to litellm ImageResponse format
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
        
        # Google AI returns predictions with generated images
        predictions = response_data.get("predictions", [])
        for prediction in predictions:
            # Google AI returns base64 encoded images in the prediction
            generated_images = prediction.get("generatedImages", [])
            for image_data in generated_images:
                model_response.data.append(ImageObject(
                    b64_json=image_data.get("bytesBase64Encoded", None),
                    url=None,  # Google AI returns base64, not URLs
                ))
        
        return model_response