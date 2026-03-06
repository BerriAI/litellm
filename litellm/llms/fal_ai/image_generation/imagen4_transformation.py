from typing import TYPE_CHECKING, Any, List, Optional

import httpx

from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from .transformation import FalAIBaseConfig

if TYPE_CHECKING:
    from litellm.litellm_core_utils.litellm_logging import Logging as _LiteLLMLoggingObj

    LiteLLMLoggingObj = _LiteLLMLoggingObj
else:
    LiteLLMLoggingObj = Any


class FalAIImagen4Config(FalAIBaseConfig):
    """
    Configuration for Fal AI Imagen4 model.
    
    Google's highest quality image generation model available through Fal AI.
    
    Model variants:
    - fal-ai/imagen4/preview (Standard): $0.05 per image
    - fal-ai/imagen4/preview/fast (Fast): $0.02 per image
    - fal-ai/imagen4/preview/ultra (Ultra): $0.06 per image
    
    Documentation: https://fal.ai/models/fal-ai/imagen4/preview
    """
    IMAGE_GENERATION_ENDPOINT: str = "fal-ai/imagen4/preview"
    
    def get_supported_openai_params(
        self, model: str
    ) -> List[OpenAIImageGenerationOptionalParams]:
        """
        Get supported OpenAI parameters for Imagen4.
        """
        return [
            "n",
            "response_format",
            "size",
        ]
    
    def map_openai_params(
        self,
        non_default_params: dict,
        optional_params: dict,
        model: str,
        drop_params: bool,
    ) -> dict:
        """
        Map OpenAI parameters to Imagen4 parameters.
        
        Mappings:
        - n -> num_images (1-4, default 1)
        - size -> aspect_ratio (1:1, 16:9, 9:16, 3:4, 4:3)
        - response_format -> ignored (Imagen4 returns URLs)
        """
        supported_params = self.get_supported_openai_params(model)
        
        # Map OpenAI params to Imagen4 params
        param_mapping = {
            "n": "num_images",
            "size": "aspect_ratio",
        }
        
        for k in non_default_params.keys():
            if k not in optional_params.keys():
                if k in supported_params:
                    # Use mapped parameter name if exists
                    mapped_key = param_mapping.get(k, k)
                    mapped_value = non_default_params[k]
                    
                    # Transform specific parameters
                    if k == "response_format":
                        # Imagen4 always returns URLs, so we can ignore this
                        continue
                    elif k == "size":
                        # Map OpenAI size format to Imagen4 aspect ratio
                        mapped_value = self._map_aspect_ratio(mapped_value)
                    
                    optional_params[mapped_key] = mapped_value
                elif drop_params:
                    pass
                else:
                    raise ValueError(
                        f"Parameter {k} is not supported for model {model}. Supported parameters are {supported_params}. Set drop_params=True to drop unsupported parameters."
                    )

        return optional_params

    def _map_aspect_ratio(self, size: str) -> str:
        """
        Map OpenAI size format to Imagen4 aspect ratio format.
        
        OpenAI format: "1024x1024", "1792x1024", etc.
        Imagen4 format: "1:1", "16:9", "9:16", "3:4", "4:3"
        
        Available aspect ratios:
        - 1:1 (default)
        - 16:9
        - 9:16
        - 3:4
        - 4:3
        """
        # Map common OpenAI sizes to Imagen4 aspect ratios
        size_to_aspect_ratio = {
            "1024x1024": "1:1",
            "512x512": "1:1",
            "1792x1024": "16:9",
            "1024x1792": "9:16",
            "1024x768": "4:3",
            "768x1024": "3:4",
        }
        
        if size in size_to_aspect_ratio:
            return size_to_aspect_ratio[size]
        
        # Parse custom size format "WIDTHxHEIGHT" and calculate aspect ratio
        if "x" in size:
            try:
                width_str, height_str = size.split("x")
                width = int(width_str)
                height = int(height_str)
                
                # Calculate aspect ratio and find closest match
                ratio = width / height
                
                # Map to closest supported aspect ratio
                if 0.95 <= ratio <= 1.05:  # Close to 1:1
                    return "1:1"
                elif ratio >= 1.7:  # Close to 16:9
                    return "16:9"
                elif ratio <= 0.6:  # Close to 9:16
                    return "9:16"
                elif ratio >= 1.2:  # Close to 4:3
                    return "4:3"
                elif ratio <= 0.8:  # Close to 3:4
                    return "3:4"
            except (ValueError, AttributeError, ZeroDivisionError):
                pass
        
        # Default to 1:1
        return "1:1"

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> dict:
        """
        Transform the image generation request to Imagen4 request body.
        
        Required parameters:
        - prompt: The text prompt describing what you want to see
        
        Optional parameters:
        - aspect_ratio: "1:1", "16:9", "9:16", "3:4", "4:3" (default: "1:1")
        - num_images: Number of images (1-4, default: 1)
        - resolution: "1K" or "2K" (default: "1K")
        - seed: Random seed for reproducibility
        - negative_prompt: Description of what to discourage (default: "")
        """
        imagen4_request_body = {
            "prompt": prompt,
            **optional_params,
        }
        
        return imagen4_request_body

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
        Transform the Imagen4 response to litellm ImageResponse format.
        
        Expected response format:
        {
            "images": [
                {
                    "url": "https://...",
                    "content_type": "image/png",
                    "file_name": "z9RV14K95DvU.png",
                    "file_size": 4404019
                }
            ],
            "seed": 42
        }
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
        
        # Handle Imagen4 response format
        images = response_data.get("images", [])
        if isinstance(images, list):
            for image_data in images:
                if isinstance(image_data, dict):
                    model_response.data.append(
                        ImageObject(
                            url=image_data.get("url", None),
                            b64_json=None,  # Imagen4 returns URLs only
                        )
                    )
                elif isinstance(image_data, str):
                    # If images is just a list of URLs
                    model_response.data.append(
                        ImageObject(
                            url=image_data,
                            b64_json=None,
                        )
                    )
        
        # Add seed metadata from Imagen4 response
        if hasattr(model_response, "_hidden_params"):
            if "seed" in response_data:
                model_response._hidden_params["seed"] = response_data["seed"]
        
        return model_response

