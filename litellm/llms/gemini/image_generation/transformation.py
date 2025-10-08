from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.llms.base_llm.image_generation.transformation import (
    BaseImageGenerationConfig,
)
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.gemini import GeminiImageGenerationRequest
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
            "size"
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
                        mapped_params["aspectRatio"] = self._map_size_to_aspect_ratio(v)
                    else:
                        mapped_params[k] = v        
        return mapped_params
    

    def _map_size_to_aspect_ratio(self, size: str) -> str:
        """
        https://ai.google.dev/gemini-api/docs/image-generation

        """
        aspect_ratio_map = {
            "1024x1024": "1:1",
            "1792x1024": "16:9", 
            "1024x1792": "9:16",
            "1280x896": "4:3",
            "896x1280": "3:4"
        }
        return aspect_ratio_map.get(size, "1:1")

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

        Gemini 2.5 Flash Image Preview: :generateContent
        Other Imagen models: :predict
        """
        complete_url: str = (
            api_base
            or get_secret_str("GEMINI_API_BASE")
            or self.DEFAULT_BASE_URL
        )

        complete_url = complete_url.rstrip("/")

        # Gemini 2.5 Flash Image Preview uses generateContent endpoint
        if "2.5-flash-image-preview" in model:
            complete_url = f"{complete_url}/models/{model}:generateContent"
        else:
            # All other Imagen models use predict endpoint
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
        Transform the image generation request to Gemini format

        For Gemini 2.5 Flash Image Preview, use the standard Gemini format with response_modalities:
        {
          "contents": [
            {
              "parts": [
                {"text": "Generate an image of..."}
              ]
            }
          ],
          "generationConfig": {
            "response_modalities": ["IMAGE", "TEXT"]
          }
        }
        """
        # For Gemini 2.5 Flash Image Preview, use standard Gemini format
        if "2.5-flash-image-preview" in model:
            request_body: dict = {
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "response_modalities": ["IMAGE", "TEXT"]
                }
            }
            return request_body
        else:
            # For other Imagen models, use the original Imagen format
            from litellm.types.llms.gemini import (
                GeminiImageGenerationInstance,
                GeminiImageGenerationParameters,
            )
            request_body_obj: GeminiImageGenerationRequest = GeminiImageGenerationRequest(
                instances=[
                    GeminiImageGenerationInstance(
                        prompt=prompt
                    )
                ],
                parameters=GeminiImageGenerationParameters(**optional_params)
            )
            return request_body_obj.model_dump(exclude_none=True)

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

        # Handle different response formats based on model
        if "2.5-flash-image-preview" in model:
            # Gemini 2.5 Flash Image Preview returns in candidates format
            candidates = response_data.get("candidates", [])
            for candidate in candidates:
                content = candidate.get("content", {})
                parts = content.get("parts", [])
                for part in parts:
                    # Look for inlineData with image
                    if "inlineData" in part:
                        inline_data = part["inlineData"]
                        if "data" in inline_data:
                            model_response.data.append(ImageObject(
                                b64_json=inline_data["data"],
                                url=None,
                            ))
        else:
            # Original Imagen format - predictions with generated images
            predictions = response_data.get("predictions", [])
            for prediction in predictions:
                # Google AI returns base64 encoded images in the prediction
                model_response.data.append(ImageObject(
                    b64_json=prediction.get("bytesBase64Encoded", None),
                    url=None,  # Google AI returns base64, not URLs
                ))
        return model_response